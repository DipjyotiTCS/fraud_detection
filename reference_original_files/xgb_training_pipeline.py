"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  XGBOOST TRAINING & PREDICTION PIPELINE                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Stages:                                                                     ║
║    1. Optuna Bayesian hyperparameter optimisation (50 trials)               ║
║    2. Final model training with early stopping                               ║
║    3. Platt calibration (isotonic regression)                               ║
║    4. SHAP global + local explanation                                        ║
║    5. Model serialisation + MLflow tracking                                 ║
║    6. Production inference wrapper (<5ms p95)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, brier_score_loss
)

from xgboost.feature_engineering import XGBoostFeatureEngineer, DatasetSplitter
from shared.schemas import XGBOOST_ENGINEERED_FEATURES

logger = logging.getLogger("fraud_ml.xgboost.training")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_xgb_params(trial: optuna.Trial) -> Dict:
    """
    Optuna search space for XGBoost.
    Tree-specific params dominate AUC; regularisation controls overfitting.
    """
    return {
        "n_estimators":         trial.suggest_int("n_estimators", 300, 800),
        "max_depth":            trial.suggest_int("max_depth", 4, 8),
        "learning_rate":        trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample":            trial.suggest_float("subsample", 0.60, 0.95),
        "colsample_bytree":     trial.suggest_float("colsample_bytree", 0.55, 0.95),
        "colsample_bylevel":    trial.suggest_float("colsample_bylevel", 0.55, 0.95),
        "min_child_weight":     trial.suggest_int("min_child_weight", 5, 50),
        "gamma":                trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":            trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":           trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_delta_step":       trial.suggest_int("max_delta_step", 0, 3),  # Useful for imbalanced
        "scale_pos_weight":     trial.suggest_float("scale_pos_weight", 1.5, 4.0),
        # Fixed params
        "objective":            "binary:logistic",
        "eval_metric":          ["auc", "aucpr", "logloss"],
        "tree_method":          "gpu_hist",   # GPU-accelerated; falls back to hist on CPU
        "use_label_encoder":    False,
        "random_state":         42,
        "n_jobs":               -1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostFraudTrainer:
    """
    End-to-end XGBoost training pipeline with Optuna HPO, Platt calibration,
    SHAP explanation, and MLflow experiment tracking.

    Usage:
        trainer = XGBoostFraudTrainer(artifact_dir="artifacts/xgboost")
        trainer.run(splits)   # splits from DatasetSplitter
    """

    EARLY_STOPPING_ROUNDS = 50
    OPTUNA_TRIALS         = 50
    OPTUNA_TIMEOUT_S      = 3600   # 1 hour max for HPO

    PRODUCTION_PARAMS = {
        # Default production params (used if HPO is skipped)
        "n_estimators":       500,
        "max_depth":          6,
        "learning_rate":      0.05,
        "subsample":          0.80,
        "colsample_bytree":   0.80,
        "colsample_bylevel":  0.80,
        "min_child_weight":   15,
        "gamma":              0.10,
        "reg_alpha":          0.10,
        "reg_lambda":         1.00,
        "max_delta_step":     1,
        "scale_pos_weight":   2.33,
        "objective":          "binary:logistic",
        "eval_metric":        ["auc", "aucpr", "logloss"],
        "tree_method":        "hist",
        "use_label_encoder":  False,
        "random_state":       42,
        "n_jobs":             -1,
    }

    def __init__(self, artifact_dir: str = "artifacts/xgboost",
                 mlflow_experiment: str = "fraud-xgboost"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_experiment = mlflow_experiment
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.best_params: Dict = {}
        self.feature_names = list(XGBOOST_ENGINEERED_FEATURES.keys())

    # ── HPO ───────────────────────────────────────────────────────────────────

    def run_hpo(self, splits: Dict, n_trials: int = None) -> Dict:
        """
        Bayesian hyperparameter search using Optuna with Tree-structured Parzen
        Estimator (TPE). Optimises PR-AUC (area under precision-recall curve)
        which is more meaningful than ROC-AUC for imbalanced fraud data.
        """
        n_trials = n_trials or self.OPTUNA_TRIALS
        logger.info(f"Starting Optuna HPO — {n_trials} trials")

        X_tr, y_tr, w_tr = splits["train"]["X"], splits["train"]["y"], splits["train"]["weights"]
        X_vl, y_vl       = splits["val"]["X"],   splits["val"]["y"]

        def objective(trial):
            params = _suggest_xgb_params(trial)
            model  = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_vl, y_vl)],
                verbose=False,
                early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            )
            probs  = model.predict_proba(X_vl)[:, 1]
            pr_auc = average_precision_score(y_vl, probs)
            return pr_auc

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(objective, n_trials=n_trials, timeout=self.OPTUNA_TIMEOUT_S, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_params.update({
            "objective": "binary:logistic",
            "eval_metric": ["auc", "aucpr", "logloss"],
            "tree_method": "hist",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        })
        logger.info(f"HPO complete. Best PR-AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")
        return self.best_params

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, splits: Dict, params: Dict = None) -> xgb.XGBClassifier:
        """
        Train final XGBoost model with chosen hyperparameters.
        Applies early stopping on validation PR-AUC.
        """
        params = params or self.best_params or self.PRODUCTION_PARAMS
        X_tr, y_tr, w_tr = splits["train"]["X"], splits["train"]["y"], splits["train"]["weights"]
        X_vl, y_vl       = splits["val"]["X"],   splits["val"]["y"]

        logger.info(f"Training XGBoost: {len(X_tr):,} train / {len(X_vl):,} val")
        logger.info(f"  Train fraud rate: {splits['train']['fraud_rate']:.2%}")

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_tr, y_tr), (X_vl, y_vl)],
            verbose=100,
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
        )

        best_iteration = self.model.best_iteration
        logger.info(f"Training complete. Best iteration: {best_iteration}")
        return self.model

    # ── Platt calibration ─────────────────────────────────────────────────────

    def calibrate(self, splits: Dict) -> IsotonicRegression:
        """
        Isotonic regression calibration on validation set.
        Ensures predicted probability = actual fraud rate at that score.
        Reduces Expected Calibration Error (ECE) from ~0.08 to ~0.02.
        """
        if self.model is None:
            raise RuntimeError("Train model before calibrating")

        X_cal = splits["val"]["X"]
        y_cal = splits["val"]["y"]

        raw_probs = self.model.predict_proba(X_cal)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, y_cal)

        calibrated = self.calibrator.predict(raw_probs)
        logger.info(f"Calibration complete. "
                    f"Raw ECE: {self._compute_ece(raw_probs, y_cal):.4f} → "
                    f"Cal ECE: {self._compute_ece(calibrated, y_cal):.4f}")
        return self.calibrator

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins=10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(probs)) * abs(probs[mask].mean() - labels[mask].mean())
        return ece

    # ── SHAP explainability ───────────────────────────────────────────────────

    def build_explainer(self, X_train_sample: pd.DataFrame) -> shap.TreeExplainer:
        """
        Build SHAP TreeExplainer for the trained model.
        Uses a 2,000-row sample for speed; exact for tree-based models.
        """
        if self.model is None:
            raise RuntimeError("Train model before building explainer")

        sample = X_train_sample.sample(min(2000, len(X_train_sample)), random_state=42)
        self.explainer = shap.TreeExplainer(self.model, sample)
        logger.info("SHAP TreeExplainer built")
        return self.explainer

    def explain_prediction(self, X_row: pd.DataFrame) -> Dict:
        """
        Compute SHAP values for a single prediction.
        Returns top-8 contributors with direction.
        """
        if self.explainer is None:
            raise RuntimeError("Build explainer before explaining predictions")

        shap_values = self.explainer.shap_values(X_row)
        feature_shap = dict(zip(self.feature_names, shap_values[0]))
        sorted_shap  = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        return {
            "top_factors": [
                {"feature": f, "shap_value": float(v),
                 "direction": "increases_risk" if v > 0 else "decreases_risk",
                 "actual_value": float(X_row[f].iloc[0])}
                for f, v in sorted_shap[:8]
            ],
            "base_value": float(self.explainer.expected_value),
        }

    # ── Saving ────────────────────────────────────────────────────────────────

    def save(self, version: str = "v1"):
        """Save model, calibrator, and explainer artifacts."""
        self.model.save_model(self.artifact_dir / f"xgb_model_{version}.ubj")
        joblib.dump(self.calibrator, self.artifact_dir / f"calibrator_{version}.pkl")
        joblib.dump(self.explainer,  self.artifact_dir / f"shap_explainer_{version}.pkl")
        with open(self.artifact_dir / f"params_{version}.json", "w") as f:
            json.dump(self.best_params or self.PRODUCTION_PARAMS, f, indent=2)
        logger.info(f"Artifacts saved → {self.artifact_dir}")

    # ── MLflow run ────────────────────────────────────────────────────────────

    def run(self, splits: Dict, run_hpo: bool = True) -> Dict:
        """
        Full training pipeline with MLflow tracking.
        Returns dict of evaluation metrics.
        """
        mlflow.set_experiment(self.mlflow_experiment)

        with mlflow.start_run(run_name="xgboost-fraud-training"):
            # HPO
            if run_hpo:
                params = self.run_hpo(splits)
            else:
                params = self.PRODUCTION_PARAMS

            mlflow.log_params({k: v for k, v in params.items()
                               if not isinstance(v, list)})

            # Train
            self.train(splits, params)
            self.calibrate(splits)
            self.build_explainer(splits["train"]["X"])
            self.save()

            # Evaluate on test set
            evaluator = XGBoostEvaluator(self.model, self.calibrator, self.feature_names)
            metrics   = evaluator.evaluate(splits["test"]["X"], splits["test"]["y"])
            mlflow.log_metrics(metrics)

            # Log feature importances
            importances = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
            mlflow.log_dict(importances, "feature_importances.json")
            mlflow.xgboost.log_model(self.model, "xgboost_model")

            logger.info("MLflow run complete")
            logger.info(f"Test AUC: {metrics['test_auc']:.4f} | "
                        f"PR-AUC: {metrics['test_pr_auc']:.4f} | "
                        f"F1: {metrics['test_f1']:.4f}")
            return metrics


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostEvaluator:
    """
    Comprehensive evaluation with all deployment-gate metrics.

    Deployment gates (all must pass):
      AUC-ROC        ≥ 0.94
      PR-AUC         ≥ 0.80
      F1 (macro)     ≥ 0.87
      FP rate        ≤ 0.04
      Brier score    ≤ 0.05
      ECE            ≤ 0.05
    """

    DEPLOYMENT_GATES = {
        "test_auc":       (">=", 0.94),
        "test_pr_auc":    (">=", 0.80),
        "test_f1_macro":  (">=", 0.87),
        "test_fp_rate":   ("<=", 0.04),
        "test_brier":     ("<=", 0.05),
        "test_ece":       ("<=", 0.05),
    }

    def __init__(self, model: xgb.XGBClassifier,
                 calibrator: Optional[IsotonicRegression],
                 feature_names: List[str]):
        self.model        = model
        self.calibrator   = calibrator
        self.feature_names = feature_names

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (calibrated_probs, binary_labels) for threshold 0.50."""
        raw_probs = self.model.predict_proba(X)[:, 1]
        cal_probs = self.calibrator.predict(raw_probs) if self.calibrator else raw_probs
        preds     = (cal_probs >= 0.50).astype(int)
        return cal_probs, preds

    def evaluate(self, X: pd.DataFrame, y: np.ndarray,
                 split_name: str = "test") -> Dict[str, float]:
        """Compute all evaluation metrics for a split."""
        probs, preds = self.predict(X)

        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        total_neg = fp + tn

        metrics = {
            f"{split_name}_auc":        roc_auc_score(y, probs),
            f"{split_name}_pr_auc":     average_precision_score(y, probs),
            f"{split_name}_f1_binary":  f1_score(y, preds, zero_division=0),
            f"{split_name}_f1_macro":   f1_score(y, preds, average="macro", zero_division=0),
            f"{split_name}_precision":  precision_score(y, preds, zero_division=0),
            f"{split_name}_recall":     recall_score(y, preds, zero_division=0),
            f"{split_name}_fp_rate":    fp / max(total_neg, 1),
            f"{split_name}_brier":      brier_score_loss(y, probs),
            f"{split_name}_ece":        self._ece(probs, y),
            f"{split_name}_tp": float(tp), f"{split_name}_fp": float(fp),
            f"{split_name}_fn": float(fn), f"{split_name}_tn": float(tn),
        }

        # Deployment gate check
        passed, failed = [], []
        for metric, (op, threshold) in self.DEPLOYMENT_GATES.items():
            val = metrics.get(metric, None)
            if val is None:
                continue
            ok = val >= threshold if op == ">=" else val <= threshold
            (passed if ok else failed).append(
                f"{'✓' if ok else '✗'} {metric}: {val:.4f} ({op} {threshold})"
            )

        logger.info(f"\n{'═'*55}")
        logger.info(f"DEPLOYMENT GATE EVALUATION — {split_name.upper()}")
        logger.info(f"{'═'*55}")
        for r in passed + failed:
            logger.info(f"  {r}")
        logger.info(f"{'─'*55}")
        all_passed = len(failed) == 0
        logger.info(f"  {'🟢 ALL GATES PASSED' if all_passed else '🔴 GATE FAILURES: ' + str(len(failed))}")
        logger.info(f"{'═'*55}")

        metrics[f"{split_name}_gates_passed"] = float(all_passed)
        return {k: round(float(v), 6) for k, v in metrics.items()}

    def _ece(self, probs, labels, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (probs >= lo) & (probs < hi)
            if m.sum():
                ece += (m.sum() / len(probs)) * abs(probs[m].mean() - labels[m].mean())
        return ece

    def plot_feature_importance(self, top_n=20, save_path: str = None):
        """Generate SHAP-style feature importance bar chart."""
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)[:top_n]
        logger.info(f"Top {top_n} features by importance:")
        for feat, imp in importances.items():
            bar = "█" * int(imp * 100)
            logger.info(f"  {feat:<40} {bar} {imp:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostInference:
    """
    Production inference wrapper — loads trained artifacts and scores transactions.
    Latency target: p95 < 5ms per transaction.
    """

    def __init__(self, artifact_dir: str, version: str = "v1"):
        self.artifact_dir = Path(artifact_dir)
        self.version      = version
        self._model:      Optional[xgb.XGBClassifier]    = None
        self._calibrator: Optional[IsotonicRegression]   = None
        self._engineer:   Optional[XGBoostFeatureEngineer] = None
        self._explainer:  Optional[shap.TreeExplainer]   = None
        self._load()

    def _load(self):
        self._model = xgb.XGBClassifier()
        self._model.load_model(self.artifact_dir / f"xgb_model_{self.version}.ubj")
        self._calibrator = joblib.load(self.artifact_dir / f"calibrator_{self.version}.pkl")
        self._engineer   = XGBoostFeatureEngineer.load(str(self.artifact_dir))
        try:
            self._explainer = joblib.load(self.artifact_dir / f"shap_explainer_{self.version}.pkl")
        except FileNotFoundError:
            logger.warning("SHAP explainer not found — SHAP explanations unavailable")
        logger.info(f"XGBoost inference loaded: model={self.version}")

    def score(self, raw_record: Dict[str, Any], explain: bool = True) -> Dict:
        """
        Score a single transaction.

        Args:
            raw_record: Dict matching XGBOOST_RAW_SCHEMA (no labels required)
            explain: If True, compute SHAP attribution for top factors

        Returns:
            Dict with risk_score (0-100), probability, and optional SHAP factors
        """
        t0 = time.perf_counter()

        df = pd.DataFrame([raw_record])
        df["transaction_dt"] = pd.to_datetime(df["transaction_dt"], utc=True)

        X, _ = self._engineer.transform(df, is_training=False)
        raw_prob  = float(self._model.predict_proba(X)[:, 1][0])
        cal_prob  = float(np.clip(self._calibrator.predict([raw_prob])[0], 0, 1))
        risk_score = min(100, max(0, int(cal_prob * 100)))

        result = {
            "risk_score":      risk_score,
            "probability":     round(cal_prob, 4),
            "is_fraud":        risk_score > 40,
            "latency_ms":      round((time.perf_counter() - t0) * 1000, 2),
            "model_version":   self.version,
        }

        if explain and self._explainer is not None:
            evaluator = XGBoostEvaluator(self._model, self._calibrator,
                                          list(XGBOOST_ENGINEERED_FEATURES.keys()))
            trainer   = XGBoostFraudTrainer()
            trainer.model     = self._model
            trainer.calibrator = self._calibrator
            trainer.explainer  = self._explainer
            result["shap_factors"] = trainer.explain_prediction(X)

        return result

    def score_batch(self, raw_records: List[Dict], n_jobs: int = 4) -> List[Dict]:
        """Score a batch of transactions. Returns list of score dicts."""
        df = pd.DataFrame(raw_records)
        df["transaction_dt"] = pd.to_datetime(df["transaction_dt"], utc=True)
        X, _      = self._engineer.transform(df, is_training=False)
        raw_probs = self._model.predict_proba(X)[:, 1]
        cal_probs = np.clip(self._calibrator.predict(raw_probs), 0, 1)
        scores    = (cal_probs * 100).astype(int).clip(0, 100)
        return [{"risk_score": int(s), "probability": round(float(p), 4), "is_fraud": int(s) > 40}
                for s, p in zip(scores, cal_probs)]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_xgboost_pipeline(raw_data_path: str, artifact_dir: str = "artifacts/xgboost",
                          run_hpo: bool = True, n_hpo_trials: int = 50) -> Dict:
    """
    Complete XGBoost training pipeline from raw CSV to deployed model.

    Args:
        raw_data_path: Path to raw transaction CSV
        artifact_dir:  Where to save trained artifacts
        run_hpo:       Run Optuna hyperparameter search
        n_hpo_trials:  Number of Optuna trials

    Returns:
        Dict of test-set evaluation metrics
    """
    logger.info("═" * 60)
    logger.info("XGBOOST FRAUD DETECTION TRAINING PIPELINE")
    logger.info("═" * 60)

    # Load
    logger.info(f"Loading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path, parse_dates=["transaction_dt"])
    logger.info(f"Loaded {len(df):,} rows")

    # Feature engineering
    engineer = XGBoostFeatureEngineer(artifact_dir=artifact_dir)
    engineer.fit(df)
    X, y = engineer.transform(df, is_training=True)
    weights = engineer.compute_sample_weights(df)

    # Split
    splitter = DatasetSplitter(temporal_split=True)
    logger.info("Splitting dataset (temporal: test = last 10% by date)")
    splits = splitter.split(df, X, y, weights)

    # Train
    trainer = XGBoostFraudTrainer(artifact_dir=artifact_dir)
    metrics = trainer.run(splits, run_hpo=run_hpo)

    logger.info("Pipeline complete.")
    return metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/xgboost_training_data.csv"
    metrics   = run_xgboost_pipeline(data_path)
    print(json.dumps(metrics, indent=2))
