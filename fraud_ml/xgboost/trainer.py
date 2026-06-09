"""Robust XGBoost training service used by the FastAPI application."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_ml.shared.schemas import XGBOOST_ENGINEERED_FEATURES
from fraud_ml.xgboost.feature_engineering import XGBoostFeatureEngineer, DatasetSplitter

logger = logging.getLogger("fraud_ml.xgboost.trainer")


DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.055,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_delta_step": 1,
    "scale_pos_weight": 2.33,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 1,
}


class XGBoostTrainingService:
    """Train, calibrate, evaluate, and persist an XGBoost fraud model."""

    def __init__(self, artifact_dir: str = "artifacts/xgboost"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(XGBOOST_ENGINEERED_FEATURES.keys())

    def train(
        self,
        raw_data_path: str,
        model_version: str = "v1",
        run_hpo: bool = False,
        n_hpo_trials: int = 10,
        temporal_split: bool = True,
    ) -> Dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_data_path = str(raw_data_path)
        df = pd.read_csv(raw_data_path, parse_dates=["transaction_dt"])

        engineer = XGBoostFeatureEngineer(artifact_dir=str(self.artifact_dir))
        engineer.fit(df)
        X, y = engineer.transform(df, is_training=True)
        weights = engineer.compute_sample_weights(df)

        splitter = DatasetSplitter(temporal_split=temporal_split)
        splits = splitter.split(df, X, y, weights)

        params = self._select_params(splits, run_hpo=run_hpo, n_trials=n_hpo_trials)
        model = xgb.XGBClassifier(**params)
        model.fit(
            splits["train"]["X"],
            splits["train"]["y"],
            sample_weight=splits["train"]["weights"],
            eval_set=[(splits["val"]["X"], splits["val"]["y"])],
            verbose=False,
        )

        raw_val_probs = model.predict_proba(splits["val"]["X"])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_val_probs, splits["val"]["y"])

        metrics = self._evaluate(model, calibrator, splits["test"]["X"], splits["test"]["y"], split_name="test")
        metrics.update({
            "train_rows": int(splits["train"]["n_total"]),
            "val_rows": int(splits["val"]["n_total"]),
            "test_rows": int(splits["test"]["n_total"]),
            "train_fraud_rate": float(splits["train"]["fraud_rate"]),
            "val_fraud_rate": float(splits["val"]["fraud_rate"]),
            "test_fraud_rate": float(splits["test"]["fraud_rate"]),
        })

        model_path = self.artifact_dir / f"xgb_model_{model_version}.ubj"
        calibrator_path = self.artifact_dir / f"calibrator_{model_version}.pkl"
        feature_engineer_path = self.artifact_dir / "feature_engineer.pkl"
        feature_names_path = self.artifact_dir / f"feature_names_{model_version}.json"
        metrics_path = self.artifact_dir / f"metrics_{model_version}.json"
        metadata_path = self.artifact_dir / f"training_metadata_{model_version}.json"
        importance_path = self.artifact_dir / f"feature_importance_{model_version}.csv"

        model.save_model(model_path)
        joblib.dump(calibrator, calibrator_path)
        # engineer.fit already saves feature_engineer.pkl into artifact_dir.
        feature_names_path.write_text(json.dumps(self.feature_names, indent=2))
        metrics_path.write_text(json.dumps(metrics, indent=2))

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(importance_path, index=False)

        metadata = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_data_path": raw_data_path,
            "model_version": model_version,
            "artifact_dir": str(self.artifact_dir),
            "params": params,
            "metrics": metrics,
            "artifacts": {
                "model_path": str(model_path),
                "calibrator_path": str(calibrator_path),
                "feature_engineer_path": str(feature_engineer_path),
                "feature_names_path": str(feature_names_path),
                "metrics_path": str(metrics_path),
                "metadata_path": str(metadata_path),
                "feature_importance_path": str(importance_path),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        return metadata

    def _select_params(self, splits: Dict[str, Any], run_hpo: bool, n_trials: int) -> Dict[str, Any]:
        if not run_hpo:
            return dict(DEFAULT_XGB_PARAMS)
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("run_hpo=True requires optuna. Install optuna or set run_hpo=false.") from exc

        X_tr, y_tr, w_tr = splits["train"]["X"], splits["train"]["y"], splits["train"]["weights"]
        X_vl, y_vl = splits["val"]["X"], splits["val"]["y"]

        def objective(trial):
            params = dict(DEFAULT_XGB_PARAMS)
            params.update({
                "n_estimators": trial.suggest_int("n_estimators", 150, 700),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
                "subsample": trial.suggest_float("subsample", 0.60, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 40),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            })
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            probs = model.predict_proba(X_vl)[:, 1]
            return average_precision_score(y_vl, probs)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=max(1, int(n_trials)), show_progress_bar=False)
        params = dict(DEFAULT_XGB_PARAMS)
        params.update(study.best_params)
        return params

    def _evaluate(self, model: xgb.XGBClassifier, calibrator: Optional[IsotonicRegression], X: pd.DataFrame, y: np.ndarray, split_name: str) -> Dict[str, float]:
        raw_probs = model.predict_proba(X)[:, 1]
        probs = calibrator.predict(raw_probs) if calibrator is not None else raw_probs
        probs = np.clip(probs, 0, 1)
        preds = (probs >= 0.50).astype(int)

        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        total_neg = max(fp + tn, 1)

        return {
            f"{split_name}_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0,
            f"{split_name}_pr_auc": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0,
            f"{split_name}_f1_binary": float(f1_score(y, preds, zero_division=0)),
            f"{split_name}_f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
            f"{split_name}_precision": float(precision_score(y, preds, zero_division=0)),
            f"{split_name}_recall": float(recall_score(y, preds, zero_division=0)),
            f"{split_name}_fp_rate": float(fp / total_neg),
            f"{split_name}_brier": float(brier_score_loss(y, probs)),
            f"{split_name}_ece": float(self._ece(probs, y)),
            f"{split_name}_tp": float(tp),
            f"{split_name}_fp": float(fp),
            f"{split_name}_fn": float(fn),
            f"{split_name}_tn": float(tn),
        }

    def _ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum():
                ece += (mask.sum() / len(probs)) * abs(probs[mask].mean() - labels[mask].mean())
        return float(ece)
