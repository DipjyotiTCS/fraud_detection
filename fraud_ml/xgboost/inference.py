"""Inference helper for trained XGBoost fraud model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_ml.xgboost.feature_engineering import XGBoostFeatureEngineer


GNN_FINDING_COLUMNS = [
    "shared_device_accounts",
    "shared_ip_accounts",
    "beneficiary_sar_count",
    "second_hop_sar_count",
    "mule_network_probability",
    "synthetic_identity_score",
    "rapid_fan_out_flag",
    "originator_graph_centrality",
]

OPTIONAL_GNN_SCORE_COLUMNS = [
    "gnn_fraud_probability",
    "gnn_risk_score",
]

GRAPH_ENGINEERED_FEATURES = {
    "shared_device_accounts_log": "shared_device_accounts",
    "shared_ip_accounts_log": "shared_ip_accounts",
    "beneficiary_sar_count": "beneficiary_sar_count",
    "second_hop_sar_count": "second_hop_sar_count",
    "mule_network_probability": "mule_network_probability",
    "synthetic_identity_score": "synthetic_identity_score",
    "rapid_fan_out_flag": "rapid_fan_out_flag",
    "originator_graph_centrality": "originator_graph_centrality",
}


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

        # Keep both values explicit so downstream consumers can verify that the
        # returned probability is produced from predict_proba(), not from a hard
        # class prediction. The public probability remains calibrated.
        raw_probs = self.model.predict_proba(X)[:, 1]
        calibrated_probs = np.clip(self.calibrator.predict(raw_probs), 0, 1)
        contribution_rows = self._compute_feature_contributions(X)

        scores: List[Dict[str, Any]] = []
        for idx, (raw_p, calibrated_p) in enumerate(zip(raw_probs, calibrated_probs)):
            p = float(calibrated_p)
            contribution_detail = contribution_rows[idx]
            graph_contributions = self._extract_graph_contributions(contribution_detail)
            contribution_detail = dict(contribution_detail)
            contribution_detail.pop("graph_contributors", None)
            scores.append({
                "transaction_id": self._safe_value(df.iloc[idx].get("transaction_id")),
                "probability": round(p, 6),
                "probability_percentage": round(p * 100, 2),
                "raw_model_probability": round(float(raw_p), 6),
                "raw_model_probability_percentage": round(float(raw_p) * 100, 2),
                "probability_source": "xgboost.XGBClassifier.predict_proba calibrated with IsotonicRegression",
                "risk_score": int(round(p * 100)),
                "classification_threshold": 0.5,
                "is_fraud": bool(p >= 0.5),
                "gnn_findings": self._extract_gnn_findings(df.iloc[idx]),
                "feature_contributions": contribution_detail,
                "gnn_feature_contributions": graph_contributions,
                "top_risk_factors": [
                    item["feature"] for item in contribution_detail["top_positive_contributors"][:5]
                ],
                "top_risk_reducers": [
                    item["feature"] for item in contribution_detail["top_negative_contributors"][:5]
                ],
            })
        return scores

    def _compute_feature_contributions(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return local XGBoost contribution details per row.

        XGBoost's pred_contribs output is the TreeSHAP-style decomposition of
        the model margin: feature contributions plus the model bias term. These
        values explain the uncalibrated XGBoost margin; the final response also
        includes the calibrated probability returned by the API.
        """
        feature_names = list(X.columns)
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        contrib_matrix = self.model.get_booster().predict(dmatrix, pred_contribs=True)

        details: List[Dict[str, Any]] = []
        for row_idx, contribs in enumerate(contrib_matrix):
            feature_contribs = contribs[:-1]
            bias = float(contribs[-1])
            abs_total = float(np.sum(np.abs(feature_contribs))) or 1.0
            rows = []
            for feature, contribution in zip(feature_names, feature_contribs):
                contribution_f = float(contribution)
                rows.append({
                    "feature": feature,
                    "value": self._safe_value(X.iloc[row_idx][feature]),
                    "contribution": round(contribution_f, 6),
                    "absolute_contribution": round(abs(contribution_f), 6),
                    "contribution_percentage": round(abs(contribution_f) / abs_total * 100, 2),
                    "direction": "increases_risk" if contribution_f > 0 else "reduces_risk" if contribution_f < 0 else "neutral",
                })

            rows_by_abs = sorted(rows, key=lambda item: item["absolute_contribution"], reverse=True)
            graph_rows = []
            for item in rows:
                raw_gnn_field = GRAPH_ENGINEERED_FEATURES.get(item["feature"])
                if raw_gnn_field:
                    enriched = dict(item)
                    enriched["raw_gnn_field"] = raw_gnn_field
                    graph_rows.append(enriched)
            graph_rows = sorted(graph_rows, key=lambda item: item["absolute_contribution"], reverse=True)
            positive = sorted(
                [item for item in rows if item["contribution"] > 0],
                key=lambda item: item["contribution"],
                reverse=True,
            )
            negative = sorted(
                [item for item in rows if item["contribution"] < 0],
                key=lambda item: item["contribution"],
            )

            details.append({
                "basis": "xgboost_pred_contribs_raw_margin",
                "model_bias": round(bias, 6),
                "top_contributors": rows_by_abs[:10],
                "top_positive_contributors": positive[:10],
                "top_negative_contributors": negative[:10],
                "graph_contributors": graph_rows,
            })
        return details

    def _extract_gnn_findings(self, row: pd.Series) -> Dict[str, Any]:
        findings = {
            col: self._safe_value(row.get(col))
            for col in GNN_FINDING_COLUMNS
            if col in row.index
        }
        for col in OPTIONAL_GNN_SCORE_COLUMNS:
            if col in row.index:
                findings[col] = self._safe_value(row.get(col))
        return findings

    def _extract_graph_contributions(self, contribution_detail: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "basis": contribution_detail.get("basis"),
            "features": contribution_detail.get("graph_contributors", []),
        }

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
