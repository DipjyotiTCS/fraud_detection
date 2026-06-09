"""Inference helper for trained XGBoost fraud model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_ml.xgboost.feature_engineering import XGBoostFeatureEngineer


class XGBoostInferenceService:
    def __init__(self, artifact_dir: str = "artifacts/xgboost", model_version: str = "v1"):
        self.artifact_dir = Path(artifact_dir)
        self.model_version = model_version
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.artifact_dir / f"xgb_model_{model_version}.ubj")
        self.calibrator = joblib.load(self.artifact_dir / f"calibrator_{model_version}.pkl")
        self.engineer = XGBoostFeatureEngineer.load(str(self.artifact_dir))

    def score_records(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df = pd.DataFrame(raw_records)
        df["transaction_dt"] = pd.to_datetime(df["transaction_dt"], utc=True)
        X, _ = self.engineer.transform(df, is_training=False)
        raw_probs = self.model.predict_proba(X)[:, 1]
        probs = np.clip(self.calibrator.predict(raw_probs), 0, 1)
        return [
            {"probability": round(float(p), 6), "risk_score": int(round(float(p) * 100)), "is_fraud": bool(p >= 0.5)}
            for p in probs
        ]
