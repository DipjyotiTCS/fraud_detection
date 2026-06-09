"""Utilities to inject trained GNN account scores into the XGBoost raw dataset."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class GNNScoreInjectionConfig:
    source_transaction_csv: str = "data/xgboost/xgboost_training_data.csv"
    gnn_score_path: str = "data/gnn/gnn_account_scores_v1.csv"
    output_dir: str = "data/xgboost"
    output_dataset_name: str = "xgboost_training_data_gnn_enriched"


class GNNScoreInjector:
    """Create an XGBoost raw dataset whose graph fields come from GNN scores."""

    def __init__(self, config: GNNScoreInjectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def inject(self) -> Dict[str, Any]:
        source = Path(self.config.source_transaction_csv)
        scores_path = Path(self.config.gnn_score_path)
        if not source.exists():
            raise FileNotFoundError(f"Source transaction CSV not found: {source}")
        if not scores_path.exists():
            raise FileNotFoundError(f"GNN score CSV not found: {scores_path}")
        df = pd.read_csv(source)
        scores = pd.read_csv(scores_path)
        if "account_id_hash" not in scores.columns or "gnn_fraud_probability" not in scores.columns:
            raise ValueError("GNN score CSV must contain account_id_hash and gnn_fraud_probability")
        score_map = scores.set_index("account_id_hash")["gnn_fraud_probability"].to_dict()
        p = df["account_id_hash"].map(score_map).fillna(0.0).astype(float).clip(0, 1)

        df["mule_network_probability"] = p.round(4)
        df["originator_graph_centrality"] = np.maximum(df.get("originator_graph_centrality", 0), np.sqrt(p) * 0.85).clip(0, 1).round(4)
        df["synthetic_identity_score"] = np.maximum(df.get("synthetic_identity_score", 0), p * 0.72).clip(0, 1).round(4)
        df["second_hop_sar_count"] = np.maximum(df.get("second_hop_sar_count", 0), np.round(p * 8)).astype(int)
        df["rapid_fan_out_flag"] = df.get("rapid_fan_out_flag", False) | (p >= 0.72)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{self.config.output_dataset_name}_{run_id}.csv"
        latest = self.output_dir / f"{self.config.output_dataset_name}.csv"
        df.to_csv(path, index=False)
        shutil.copyfile(path, latest)
        metadata = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_transaction_csv": str(source),
            "gnn_score_path": str(scores_path),
            "csv_path": str(path),
            "latest_csv_path": str(latest),
            "rows": int(len(df)),
            "mean_injected_gnn_probability": float(p.mean()),
            "max_injected_gnn_probability": float(p.max()),
        }
        metadata_path = self.output_dir / f"{self.config.output_dataset_name}_metadata_{run_id}.json"
        latest_metadata = self.output_dir / f"{self.config.output_dataset_name}_metadata_latest.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        latest_metadata.write_text(json.dumps(metadata, indent=2, default=str))
        metadata["metadata_path"] = str(metadata_path)
        metadata["latest_metadata_path"] = str(latest_metadata)
        return metadata
