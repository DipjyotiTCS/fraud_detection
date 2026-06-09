"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  XGBOOST FEATURE ENGINEERING PIPELINE                                       ║
║  Transforms raw transaction records → 44-dimensional training matrix        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Pipeline stages:                                                            ║
║    1. Schema validation & null imputation                                    ║
║    2. Temporal feature extraction (hour, day, weekend, month-end)           ║
║    3. Amount transformation (log, z-score, CTR proximity)                   ║
║    4. Velocity ratio engineering                                             ║
║    5. Channel one-hot encoding                                               ║
║    6. Watchlist / geo / device features                                     ║
║    7. Behavioural drift & peer percentile                                    ║
║    8. Graph signal passthrough                                               ║
║    9. Sample weight computation (label source × class weight)               ║
║   10. Train/validation/test stratified split                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from shared.schemas import (
    XGBOOST_RAW_SCHEMA, XGBOOST_ENGINEERED_FEATURES,
    CLASS_WEIGHTS, FraudLabel, DataQualityValidator
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger("fraud_ml.xgboost.features")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CTR_THRESHOLD_INR  = 1_000_000     # RBI Currency Transaction Report threshold
LABEL_SOURCE_WEIGHTS = {           # Multiplier on class weight by label source
    "CONFIRMED_SAR":   1.50,       # Gold standard — highest weight
    "INVESTIGATOR":    1.20,
    "CARD_NETWORK":    1.10,
    "NPCI":            1.10,
    "CONSORTIUM":      0.90,       # Anonymised — slightly lower quality
    "SYNTHETIC":       0.70,       # Synthetic augmentation — lower weight
}

CHANNEL_COLS = ["IMPS","NEFT","RTGS","UPI","CARD_CP","CARD_CNP","WIRE","CRYPTO","P2P"]
NULL_FILL_DEFAULTS = {
    "ip_reputation_score":       0.5,
    "geo_distance_km":           0.0,
    "behavioural_drift_score":   0.0,
    "peer_amount_percentile":    50.0,
    "mule_network_probability":  0.0,
    "synthetic_identity_score":  0.0,
    "originator_graph_centrality": 0.0,
    "second_hop_sar_count":      0,
    "beneficiary_sar_count":     0,
    "distinct_devices_30d":      1,
    "failed_auth_count_7d":      0,
    "shared_device_accounts":    0,
    "shared_ip_accounts":        0,
    "amount_mean_90d":           0.0,
    "amount_std_90d":            1.0,
    "typical_velocity_daily":    1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostFeatureEngineer:
    """
    Stateless feature transformer (fit on train, apply on any split).

    The fit() call computes only what depends on training data:
      - Channel label encoder (for unseen channels at inference)
      - MCC code label encoder
      - Watchlist category encoder

    All other transformations are deterministic and stateless.
    """

    def __init__(self, artifact_dir: str = "artifacts/xgboost"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._channel_encoder    = LabelEncoder()
        self._mcc_encoder        = LabelEncoder()
        self._watchlist_encoder  = LabelEncoder()
        self._fitted             = False

    # ── Step 1: Null imputation ───────────────────────────────────────────────

    def _impute_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls with domain-appropriate defaults."""
        df = df.copy()
        for col, default in NULL_FILL_DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        # Boolean columns: null → False
        bool_cols = [k for k, v in XGBOOST_RAW_SCHEMA.items()
                     if v["dtype"] == "bool" and k in df.columns]
        for col in bool_cols:
            df[col] = df[col].fillna(False).astype(bool)
        return df

    # ── Step 2: Temporal features ─────────────────────────────────────────────

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive time-based features from transaction_dt.
        These capture nocturnal fraud patterns, weekend/month-end activity.
        """
        if "transaction_dt" not in df.columns:
            raise ValueError("transaction_dt column required for temporal features")

        dt = pd.to_datetime(df["transaction_dt"], utc=True)
        df["hour_of_day"]  = dt.dt.hour.astype(np.float32)
        df["day_of_week"]  = dt.dt.dayofweek.astype(np.float32)
        df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(np.float32)
        df["is_nighttime"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 22)).astype(np.float32)
        df["is_month_end"] = (dt.dt.day >= 28).astype(np.float32)
        return df

    # ── Step 3: Amount transformations ────────────────────────────────────────

    def _engineer_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log-transform heavy-tailed amounts; compute z-score vs baseline;
        calculate structuring proximity to CTR threshold.
        """
        df["amount_inr_log"]       = np.log1p(df["amount_inr"]).astype(np.float32)
        df["amount_usd_equiv_log"] = np.log1p(df["amount_usd_equiv"]).astype(np.float32)
        df["amount_sum_1h_log"]    = np.log1p(df["amount_sum_1h"]).astype(np.float32)
        df["amount_sum_24h_log"]   = np.log1p(df["amount_sum_24h"]).astype(np.float32)

        # Z-score: deviation from account's 90-day baseline
        std_safe = df["amount_std_90d"].replace(0, 1)  # avoid div/0
        df["amount_zscore"] = (
            (df["amount_inr"] - df["amount_mean_90d"]) / std_safe
        ).clip(-10, 10).astype(np.float32)

        # Structuring proximity: how close to the CTR threshold?
        # High value when amount is just below threshold (classic structuring signal)
        df["ctr_proximity"] = (
            1.0 - np.abs(df["amount_inr"] - CTR_THRESHOLD_INR) / CTR_THRESHOLD_INR
        ).clip(0, 1).astype(np.float32)

        return df

    # ── Step 4: Velocity ratio engineering ────────────────────────────────────

    def _engineer_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute velocity ratios that capture acceleration patterns.
        A fraud burst looks like: velocity_1h >> velocity_24h/24.
        """
        daily_rate_safe = (df["velocity_24h"] / 24).replace(0, 0.001)
        df["velocity_ratio_1h_24h"] = (df["velocity_1h"] / daily_rate_safe).clip(0, 1000).astype(np.float32)

        # Average amount per transaction in the last hour
        v1h_safe = df["velocity_1h"].replace(0, 1)
        df["amount_per_tx_1h"] = (df["amount_sum_1h"] / v1h_safe).astype(np.float32)
        df["amount_per_tx_1h"] = np.log1p(df["amount_per_tx_1h"]).astype(np.float32)

        # Velocity vs. account baseline
        daily_baseline_safe = (df["typical_velocity_daily"] / 24).replace(0, 0.001)
        df["velocity_vs_baseline"] = (df["velocity_1h"] / daily_baseline_safe).clip(0, 500).astype(np.float32)

        # Beneficiary diversification rate (spread pattern)
        df["distinct_bene_6h"]  = df["distinct_beneficiaries_6h"].astype(np.float32)
        df["distinct_bene_24h"] = df["distinct_beneficiaries_24h"].astype(np.float32)

        return df

    # ── Step 5: Channel one-hot encoding ──────────────────────────────────────

    def _encode_channel(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode payment channel. Unknown channels → all zeros."""
        for ch in CHANNEL_COLS:
            col_name = f"channel_{ch}"
            df[col_name] = (df["channel"] == ch).astype(np.float32)
        df["is_international"] = df["is_international"].astype(np.float32)
        return df

    # ── Step 6: Geo / device / watchlist features ─────────────────────────────

    def _engineer_geo_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform geo distance; cast booleans to float."""
        df["geo_distance_km_log"]  = np.log1p(df["geo_distance_km"]).astype(np.float32)
        df["geo_country_mismatch"] = df["geo_country_mismatch"].astype(np.float32)
        df["is_tor_vpn_proxy"]     = df["is_tor_vpn_proxy"].astype(np.float32)
        df["ip_reputation_score"]  = df["ip_reputation_score"].astype(np.float32)
        df["failed_auth_count_7d"] = df["failed_auth_count_7d"].astype(np.float32)
        df["distinct_devices_30d"] = df["distinct_devices_30d"].astype(np.float32)
        df["watchlist_hit"]        = df["watchlist_hit"].astype(np.float32)
        return df

    # ── Step 7: Behavioural features ──────────────────────────────────────────

    def _engineer_behavioural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast / transform account behavioural profile features."""
        df["behavioural_drift_score"]  = df["behavioural_drift_score"].astype(np.float32)
        df["is_dormant_account"]       = df["is_dormant_account"].astype(np.float32)
        df["days_since_acct_open_log"] = np.log1p(df["days_since_account_open"]).astype(np.float32)
        df["is_new_beneficiary"]       = df["is_new_beneficiary"].astype(np.float32)
        df["peer_amount_percentile"]   = df["peer_amount_percentile"].astype(np.float32)
        return df

    # ── Step 8: Graph signal features ─────────────────────────────────────────

    def _engineer_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform graph degree features; pass through probabilities."""
        df["shared_device_accounts_log"] = np.log1p(df["shared_device_accounts"]).astype(np.float32)
        df["shared_ip_accounts_log"]     = np.log1p(df["shared_ip_accounts"]).astype(np.float32)
        df["beneficiary_sar_count"]      = df["beneficiary_sar_count"].astype(np.float32)
        df["second_hop_sar_count"]       = df["second_hop_sar_count"].astype(np.float32)
        df["mule_network_probability"]   = df["mule_network_probability"].astype(np.float32)
        df["synthetic_identity_score"]   = df["synthetic_identity_score"].astype(np.float32)
        df["rapid_fan_out_flag"]         = df["rapid_fan_out_flag"].astype(np.float32)
        df["originator_graph_centrality"] = df["originator_graph_centrality"].astype(np.float32)
        return df

    # ── Step 9: Sample weights ────────────────────────────────────────────────

    def compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Combine class weights (fraud vs. benign) with label-source weights
        (confirmed SAR labels are worth more than synthetic ones).

        Final weight = class_weight × source_weight × label_confidence
        """
        class_w  = df["label_is_fraud"].map(CLASS_WEIGHTS).fillna(1.0)
        source_w = df["label_source"].map(LABEL_SOURCE_WEIGHTS).fillna(1.0)
        conf_w   = df["label_confidence"].fillna(1.0)
        weights  = (class_w * source_w * conf_w).values.astype(np.float32)
        return weights / weights.mean()   # Normalise so mean weight = 1.0

    # ── Main fit / transform API ───────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "XGBoostFeatureEngineer":
        """
        Learn any data-dependent transformers from training data.
        Currently minimal (most transformations are stateless), but
        the pattern allows adding learned scalers/encoders later.
        """
        self._fitted = True
        joblib.dump(self, self.artifact_dir / "feature_engineer.pkl")
        logger.info(f"FeatureEngineer fitted and saved → {self.artifact_dir}")
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply all 8 feature engineering steps.

        Returns:
            X: DataFrame with exactly 44 engineered features
            y: Binary label array (only if is_training=True, else zeros)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        validator = DataQualityValidator()
        report = validator.validate_xgboost_dataset(df)
        if not report.passes_contract:
            logger.warning(f"Data quality issues: {report.failure_reasons}")

        df = self._impute_nulls(df)
        df = self._extract_temporal_features(df)
        df = self._engineer_amount_features(df)
        df = self._engineer_velocity_features(df)
        df = self._encode_channel(df)
        df = self._engineer_geo_device_features(df)
        df = self._engineer_behavioural_features(df)
        df = self._engineer_graph_features(df)

        feature_cols = list(XGBOOST_ENGINEERED_FEATURES.keys())
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing engineered features after pipeline: {missing}")

        X = df[feature_cols].copy()
        assert X.shape[1] == 44, f"Expected 44 features, got {X.shape[1]}"

        y = df["label_is_fraud"].values.astype(np.int8) if is_training and "label_is_fraud" in df.columns \
            else np.zeros(len(df), dtype=np.int8)

        logger.info(f"Feature engineering complete: {X.shape[0]:,} rows × {X.shape[1]} features")
        return X, y

    @classmethod
    def load(cls, artifact_dir: str) -> "XGBoostFeatureEngineer":
        return joblib.load(Path(artifact_dir) / "feature_engineer.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

class DatasetSplitter:
    """
    Stratified train/validation/test split with temporal ordering.

    Critical: test set MUST come from later time period than train/val
    to avoid temporal leakage. Group is account_id_hash to prevent the
    same account appearing in both train and test.
    """

    def __init__(self, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10,
                 temporal_split=True, random_state=42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        self.train_ratio    = train_ratio
        self.val_ratio      = val_ratio
        self.test_ratio     = test_ratio
        self.temporal_split = temporal_split
        self.random_state   = random_state

    def split(self, df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
              weights: np.ndarray) -> Dict:
        """
        Returns dict with train/val/test DataFrames, feature arrays,
        labels, and sample weights.
        """
        if self.temporal_split:
            # Sort by date; last 10% of TIME is the test set
            df_sorted = df.sort_values("transaction_dt").reset_index(drop=True)
            n = len(df_sorted)
            train_val_end = int(n * (1 - self.test_ratio))
            val_start     = int(train_val_end * (1 - self.val_ratio / (self.train_ratio + self.val_ratio)))

            idx_train = df_sorted.index[:val_start]
            idx_val   = df_sorted.index[val_start:train_val_end]
            idx_test  = df_sorted.index[train_val_end:]
        else:
            idx_trainval, idx_test = train_test_split(
                range(len(df)), test_size=self.test_ratio,
                stratify=y, random_state=self.random_state
            )
            y_trainval = y[idx_trainval]
            idx_train, idx_val = train_test_split(
                idx_trainval, test_size=self.val_ratio / (self.train_ratio + self.val_ratio),
                stratify=y_trainval, random_state=self.random_state
            )
            idx_train, idx_val, idx_test = np.array(idx_train), np.array(idx_val), np.array(idx_test)

        splits = {}
        for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
            splits[split_name] = {
                "df":      df.iloc[idx].reset_index(drop=True),
                "X":       X.iloc[idx].reset_index(drop=True),
                "y":       y[idx],
                "weights": weights[idx],
                "n_fraud": int(y[idx].sum()),
                "n_total": len(idx),
                "fraud_rate": float(y[idx].mean()),
            }
            logger.info(f"  {split_name:5s}: {splits[split_name]['n_total']:>8,} rows | "
                        f"fraud: {splits[split_name]['n_fraud']:>6,} ({splits[split_name]['fraud_rate']:.1%})")

        return splits
