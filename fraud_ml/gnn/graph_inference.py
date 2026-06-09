"""Batch inference for the account GraphSAGE model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from fraud_ml.gnn.graph_dataset_builder import ACCOUNT_FEATURE_COLUMNS
from fraud_ml.gnn.graph_trainer import AccountGraphSAGE, HAS_TORCH

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class GNNInferConfig:
    graph_data_dir: str = "data/gnn"
    dataset_name: str = "fraud_graph"
    artifact_dir: str = "artifacts/gnn"
    model_version: str = "v1"
    output_path: str = "data/gnn/gnn_account_scores_v1.csv"
    device: str = "auto"


class GNNInferenceService:
    def __init__(self, config: GNNInferConfig):
        if not HAS_TORCH:
            raise RuntimeError("GNN inference requires PyTorch. Install all dependencies with: pip install -r requirements.txt")
        self.config = config
        self.graph_data_dir = Path(config.graph_data_dir)
        self.artifact_dir = Path(config.artifact_dir)
        self.device = self._resolve_device(config.device)

    def infer(self) -> Dict[str, Any]:
        account_df, X, edge_index = self._load_graph_for_inference()
        cfg_path = self.artifact_dir / f"gnn_config_{self.config.model_version}.json"
        model_path = self.artifact_dir / f"gnn_model_{self.config.model_version}.pt"
        if not cfg_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"GNN artifacts not found under {self.artifact_dir} for version {self.config.model_version}")
        model_cfg = json.loads(cfg_path.read_text())
        model = AccountGraphSAGE(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            edge_t = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            probs = torch.softmax(model(X_t, edge_t), dim=1)[:, 1].detach().cpu().numpy()
        out = pd.DataFrame({
            "account_id_hash": account_df["node_id"].astype(str),
            "gnn_fraud_probability": np.clip(probs, 0, 1),
            "gnn_risk_score": np.round(np.clip(probs, 0, 1) * 100).astype(int),
        })
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        return {
            "output_path": str(output_path),
            "rows": int(len(out)),
            "mean_gnn_fraud_probability": float(out["gnn_fraud_probability"].mean()),
            "max_gnn_fraud_probability": float(out["gnn_fraud_probability"].max()),
        }

    def _load_graph_for_inference(self):
        account_path = self.graph_data_dir / f"{self.config.dataset_name}_account_nodes.csv"
        edge_path = self.graph_data_dir / f"{self.config.dataset_name}_edges_account_account.csv"
        shared_path = self.graph_data_dir / f"{self.config.dataset_name}_edges_account_shared_entity.csv"
        scaler_path = self.artifact_dir / f"gnn_scaler_{self.config.model_version}.pkl"
        if not account_path.exists():
            raise FileNotFoundError(f"Account node CSV not found: {account_path}")
        account_df = pd.read_csv(account_path)
        account_df = account_df.sort_values("node_id").reset_index(drop=True)
        for col in ACCOUNT_FEATURE_COLUMNS:
            if col not in account_df.columns:
                account_df[col] = 0.0
        scaler = joblib.load(scaler_path)
        X_raw = account_df[ACCOUNT_FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        X = scaler.transform(X_raw).astype(np.float32)
        id_map = {node_id: i for i, node_id in enumerate(account_df["node_id"].astype(str))}
        edge_index = self._load_edges(edge_path, shared_path, id_map, len(account_df))
        return account_df, X, edge_index

    def _load_edges(self, edge_path: Path, shared_path: Path, id_map: Dict[str, int], n: int) -> np.ndarray:
        def convert(path: Path) -> np.ndarray:
            if not path.exists():
                return np.empty((2, 0), dtype=np.int64)
            edges = pd.read_csv(path)
            if edges.empty:
                return np.empty((2, 0), dtype=np.int64)
            src = edges["src_node_id"].astype(str).map(id_map)
            dst = edges["dst_node_id"].astype(str).map(id_map)
            mask = src.notna() & dst.notna()
            if not mask.sum():
                return np.empty((2, 0), dtype=np.int64)
            return np.vstack([src[mask].astype(int).values, dst[mask].astype(int).values]).astype(np.int64)
        base = convert(edge_path)
        shared = convert(shared_path)
        pieces = [p for p in [base, shared] if p.size]
        loops = np.vstack([np.arange(n), np.arange(n)])
        if pieces:
            pairs = np.concatenate(pieces, axis=1)
            return np.concatenate([pairs, pairs[::-1], loops], axis=1)
        return loops

    def _resolve_device(self, device: str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
