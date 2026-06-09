"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GNN (GraphSAGE) TRAINING & PREDICTION PIPELINE                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Pipeline stages:                                                            ║
║    1. Neo4j → PyG HeteroData graph construction                             ║
║    2. Node feature engineering (account, device, IP, merchant)              ║
║    3. Edge feature engineering (TRANSACTS_WITH, USES_DEVICE, etc.)         ║
║    4. Inductive mini-batch training with NeighborSampler                   ║
║    5. GraphSAGE multi-relational message passing                            ║
║    6. Node classification head (account fraud / benign)                    ║
║    7. Temporal graph split (avoid future leakage)                          ║
║    8. Entity score batch inference → Redis cache                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import torch_geometric as pyg
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, HeteroConv, Linear, to_hetero
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.utils import add_self_loops
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    # Provide stub classes so the file is still importable and readable
    class HeteroData: pass
    class NeighborLoader: pass

logger = logging.getLogger("fraud_ml.gnn.pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# NODE FEATURE DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────

NODE_FEATURE_DIMS = {
    "account":  17,   # See GNN_NODE_SCHEMA["account"] — 17 numeric features
    "device":   6,
    "ip":       7,
    "merchant": 6,
}

EDGE_TYPES = [
    ("account",  "TRANSACTS_WITH",  "account"),
    ("account",  "USES_DEVICE",     "device"),
    ("account",  "ORIGINATES_FROM", "ip"),
    ("account",  "TRANSACTS_AT",    "merchant"),
    # Reverse edges (for bidirectional message passing)
    ("account",  "REV_TRANSACTS_WITH",  "account"),
    ("device",   "REV_USES_DEVICE",     "account"),
    ("ip",       "REV_ORIGINATES_FROM", "account"),
    ("merchant", "REV_TRANSACTS_AT",    "account"),
]


# ─────────────────────────────────────────────────────────────────────────────
# GNN FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class GNNFeatureEngineer:
    """
    Transforms raw Neo4j node and edge DataFrames into PyG HeteroData graph.

    Input:  node_dfs (dict of pd.DataFrame per node type)
            edge_dfs (dict of pd.DataFrame per edge type)
    Output: torch_geometric.data.HeteroData ready for training
    """

    def __init__(self, artifact_dir: str = "artifacts/gnn"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._node_id_maps: Dict[str, Dict[str, int]] = {}  # hash → integer index
        self._feature_stats: Dict[str, Dict] = {}            # mean/std for normalisation

    # ── Node feature preparation ──────────────────────────────────────────────

    def _prepare_account_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        17-dimensional account node feature vector.
        All features log-transformed where heavy-tailed; scaled 0-1 where bounded.
        """
        feats = pd.DataFrame()
        feats["account_age_log"]         = np.log1p(df["account_age_days"])
        feats["kyc_tier"]                = df["kyc_tier"].astype(float) / 3.0
        feats["account_type"]            = df["account_type"].astype(float) / 2.0
        feats["relationship_years_log"]  = np.log1p(df["relationship_years"])
        feats["lifetime_tx_log"]         = np.log1p(df["lifetime_tx_count"])
        feats["lifetime_amount_log"]     = df["lifetime_amount_log"]
        feats["prior_sar_count"]         = np.log1p(df["prior_sar_count"])
        feats["prior_alert_count"]       = np.log1p(df["prior_alert_count"])
        feats["prior_confirmed_fraud"]   = df["prior_confirmed_fraud"].astype(float)
        feats["monthly_avg_tx_log"]      = np.log1p(df["monthly_avg_tx_count"])
        feats["monthly_avg_amount_log"]  = df["monthly_avg_amount_log"]
        feats["geographic_risk_score"]   = df["geographic_risk_score"].clip(0, 1)
        feats["pep_flag"]                = df["pep_flag"].astype(float)
        feats["industry_risk_score"]     = df["industry_risk_score"].fillna(0.0).clip(0, 1)
        feats["distinct_devices_log"]    = np.log1p(df["distinct_devices_90d"])
        feats["avg_failed_auth_log"]     = np.log1p(df["avg_failed_auth_monthly"])
        feats["is_new_account"]          = (df["account_age_days"] < 90).astype(float)
        return torch.tensor(feats.values, dtype=torch.float32)

    def _prepare_device_features(self, df: pd.DataFrame) -> torch.Tensor:
        """6-dimensional device node feature vector."""
        feats = pd.DataFrame()
        feats["first_seen_log"]          = np.log1p(df["first_seen_days_ago"])
        feats["assoc_accounts_log"]      = np.log1p(df["associated_account_count"])
        feats["platform_android"]        = (df["platform"] == 0).astype(float)
        feats["platform_ios"]            = (df["platform"] == 1).astype(float)
        feats["device_risk_score"]       = df["device_risk_score"].clip(0, 1)
        feats["has_root_jailbreak"]      = df["has_root_jailbreak"].astype(float)
        return torch.tensor(feats.values, dtype=torch.float32)

    def _prepare_ip_features(self, df: pd.DataFrame) -> torch.Tensor:
        """7-dimensional IP node feature vector."""
        feats = pd.DataFrame()
        feats["reputation_score"]        = df["reputation_score"].clip(0, 1)
        feats["is_tor"]                  = df["is_tor"].astype(float)
        feats["is_vpn"]                  = df["is_vpn"].astype(float)
        feats["is_proxy"]                = df["is_proxy"].astype(float)
        feats["country_risk_score"]      = df["country_risk_score"].clip(0, 1)
        feats["assoc_accounts_log"]      = np.log1p(df["associated_account_count"])
        feats["risk_composite"]          = (df["is_tor"].astype(float) * 0.5 +
                                            df["is_vpn"].astype(float) * 0.3 +
                                            df["reputation_score"] * 0.2)
        return torch.tensor(feats.values, dtype=torch.float32)

    def _prepare_merchant_features(self, df: pd.DataFrame) -> torch.Tensor:
        """6-dimensional merchant node feature vector."""
        feats = pd.DataFrame()
        feats["merchant_age_log"]        = np.log1p(df["merchant_age_days"])
        feats["mcc_risk_score"]          = df["mcc_risk_score"].clip(0, 1)
        feats["chargeback_rate"]         = df["chargeback_rate_90d"].clip(0, 1)
        feats["dispute_count_log"]       = np.log1p(df["dispute_count_90d"])
        feats["is_new_merchant"]         = (df["merchant_age_days"] < 30).astype(float)
        feats["high_chargeback_flag"]    = (df["chargeback_rate_90d"] > 0.02).astype(float)
        return torch.tensor(feats.values, dtype=torch.float32)

    # ── Edge feature preparation ──────────────────────────────────────────────

    def _prepare_edge_features(self, edge_type: str, df: pd.DataFrame) -> torch.Tensor:
        """Prepare edge attribute tensors for each edge type."""
        if edge_type == "TRANSACTS_WITH":
            feats = pd.DataFrame()
            feats["tx_count_log"]        = np.log1p(df["transaction_count_30d"])
            feats["amount_log"]          = df["total_amount_30d_log"]
            feats["first_tx_recency"]    = np.log1p(df["first_transaction_days_ago"])
            feats["last_tx_recency"]     = np.log1p(df["last_transaction_hours_ago"] / 24)
            feats["avg_amount_log"]      = df["avg_amount_log"]
            feats["is_recurring"]        = df["is_recurring"].astype(float)
            feats["edge_risk_score"]     = df["edge_risk_score"].clip(0, 1)
        elif edge_type == "USES_DEVICE":
            feats = pd.DataFrame()
            feats["session_count_log"]   = np.log1p(df["session_count_30d"])
            feats["recency_log"]         = np.log1p(df["last_session_hours_ago"] / 24)
            feats["is_primary_device"]   = df["is_primary_device"].astype(float)
        elif edge_type == "ORIGINATES_FROM":
            feats = pd.DataFrame()
            feats["session_count_log"]   = np.log1p(df["session_count_7d"])
            feats["recency_log"]         = np.log1p(df["last_session_hours_ago"] / 24)
        elif edge_type == "TRANSACTS_AT":
            feats = pd.DataFrame()
            feats["tx_count_log"]        = np.log1p(df["transaction_count_90d"])
            feats["amount_log"]          = df["total_amount_90d_log"]
            feats["is_first_transaction"]= df["is_first_transaction"].astype(float)
        else:
            return None

        return torch.tensor(feats.values.astype(np.float32), dtype=torch.float32)

    # ── Graph construction ────────────────────────────────────────────────────

    def build_graph(self, node_dfs: Dict[str, pd.DataFrame],
                    edge_dfs: Dict[str, pd.DataFrame]) -> HeteroData:
        """
        Build PyG HeteroData graph from raw DataFrames.

        Args:
            node_dfs: {"account": df_accounts, "device": df_devices, ...}
            edge_dfs: {"TRANSACTS_WITH": df_edges, ...}
                      Each edge_df must have columns: src_node_id, dst_node_id

        Returns:
            HeteroData graph ready for training
        """
        if not HAS_PYG:
            raise ImportError("torch-geometric required: pip install torch-geometric")

        data = HeteroData()

        # Build node index maps
        for node_type, df in node_dfs.items():
            self._node_id_maps[node_type] = {
                nid: idx for idx, nid in enumerate(df["node_id"])
            }

        # Add node features and labels
        data["account"].x       = self._prepare_account_features(node_dfs["account"])
        data["device"].x        = self._prepare_device_features(node_dfs["device"])
        data["ip"].x            = self._prepare_ip_features(node_dfs["ip"])
        data["merchant"].x      = self._prepare_merchant_features(node_dfs["merchant"])

        # Labels on account nodes only
        if "node_label" in node_dfs["account"].columns:
            data["account"].y   = torch.tensor(
                node_dfs["account"]["node_label"].values, dtype=torch.long
            )

        # Add edges
        for edge_key, edge_df in edge_dfs.items():
            # Determine node types from edge_key (format: "SRC_TYPE__EDGE_TYPE__DST_TYPE")
            parts = edge_key.split("__")
            if len(parts) != 3:
                continue
            src_type, rel_type, dst_type = parts

            src_map = self._node_id_maps.get(src_type, {})
            dst_map = self._node_id_maps.get(dst_type, {})

            src_idx = edge_df["src_node_id"].map(src_map).dropna().astype(int)
            dst_idx = edge_df["dst_node_id"].map(dst_map).dropna().astype(int)

            valid_mask = (~edge_df["src_node_id"].map(src_map).isna() &
                          ~edge_df["dst_node_id"].map(dst_map).isna())
            edge_df_valid = edge_df[valid_mask]

            edge_index = torch.tensor(
                np.vstack([src_idx.values, dst_idx.values]), dtype=torch.long
            )
            data[(src_type, rel_type, dst_type)].edge_index = edge_index

            # Edge attributes
            edge_attr = self._prepare_edge_features(rel_type, edge_df_valid)
            if edge_attr is not None:
                data[(src_type, rel_type, dst_type)].edge_attr = edge_attr

            # Add reverse edges for bidirectional message passing
            rev_rel = f"REV_{rel_type}"
            data[(dst_type, rev_rel, src_type)].edge_index = edge_index.flip(0)

        logger.info(f"Graph built: {data}")
        return data

    def save_id_maps(self):
        """Persist node ID maps for inference-time node lookup."""
        import json
        for node_type, id_map in self._node_id_maps.items():
            with open(self.artifact_dir / f"id_map_{node_type}.json", "w") as f:
                json.dump(id_map, f)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHSAGE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HeteroGraphSAGE(nn.Module):
    """
    Multi-relational GraphSAGE for heterogeneous fraud detection graphs.

    Architecture:
      Layer 1: HeteroConv(SAGEConv) per edge type → 256-dim hidden
      Layer 2: HeteroConv(SAGEConv) per edge type → 128-dim hidden
      Layer 3: HeteroConv(SAGEConv) per edge type → 64-dim hidden
      Classification head: Linear(64 → 2) on account nodes
      Aggregation: mean (robust to variable node degrees)

    Key design choices:
      - Inductive (NeighborSampler) — scales to 15M+ node graphs
      - Layer-wise sampling: [15, 10, 5] neighbours per layer
      - Batch normalisation between layers for training stability
      - Dropout 0.3 for regularisation
      - Class-weighted cross-entropy for imbalanced labels
    """

    def __init__(self, hidden_channels: int = 256, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.num_layers      = num_layers
        self.dropout         = dropout
        self.hidden_channels = hidden_channels

        # Input projections (different feature dims per node type)
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_channels)
            for node_type, dim in NODE_FEATURE_DIMS.items()
        })

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            in_ch  = hidden_channels
            out_ch = hidden_channels // (2 ** layer) if layer < num_layers - 1 else hidden_channels // (2 ** (num_layers - 1))

            conv = HeteroConv(
                {
                    ("account",  "TRANSACTS_WITH",     "account"): SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("account",  "USES_DEVICE",         "device"):  SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("account",  "ORIGINATES_FROM",     "ip"):      SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("account",  "TRANSACTS_AT",        "merchant"):SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("account",  "REV_TRANSACTS_WITH",  "account"): SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("device",   "REV_USES_DEVICE",     "account"): SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("ip",       "REV_ORIGINATES_FROM", "account"): SAGEConv(in_ch, out_ch, aggr="mean"),
                    ("merchant", "REV_TRANSACTS_AT",    "account"): SAGEConv(in_ch, out_ch, aggr="mean"),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.ModuleDict({
                node_type: nn.BatchNorm1d(out_ch) for node_type in NODE_FEATURE_DIMS
            }))

        # Fraud classification head (account nodes only)
        final_dim = hidden_channels // (2 ** (num_layers - 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 2),   # Binary: fraud / benign
        )

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to common hidden dimension
        h = {
            node_type: F.relu(self.input_projections[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.input_projections
        }

        for i, (conv, bns) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index_dict)
            # Apply activation + batch norm per node type
            h = {}
            for node_type, emb in h_new.items():
                if node_type in bns:
                    emb = bns[node_type](emb)
                emb = F.relu(emb)
                if i < self.num_layers - 1:   # No dropout on last layer
                    emb = F.dropout(emb, p=self.dropout, training=self.training)
                h[node_type] = emb

        # Classify account nodes
        return self.classifier(h["account"])   # [N_accounts, 2]

    def get_embeddings(self, x_dict, edge_index_dict) -> torch.Tensor:
        """Extract penultimate-layer account node embeddings (for downstream use)."""
        h = {
            node_type: F.relu(self.input_projections[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.input_projections
        }
        for i, (conv, bns) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index_dict)
            h = {}
            for node_type, emb in h_new.items():
                if node_type in bns:
                    emb = bns[node_type](emb)
                emb = F.relu(emb)
                h[node_type] = emb
        return h["account"]   # Return raw embeddings


# ─────────────────────────────────────────────────────────────────────────────
# GNN TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class GNNFraudTrainer:
    """
    Inductive GraphSAGE training with mini-batch NeighborLoader.

    Configuration:
      Batch size:         2,048 account nodes per batch
      Neighbour sampling: [15, 10, 5] per layer (layer1=15, layer2=10, layer3=5)
      Optimiser:          AdamW, lr=1e-3, weight_decay=1e-4
      Scheduler:          Cosine annealing
      Loss:               Weighted cross-entropy (fraud weight = 2.33×)
      Epochs:             100 with early stopping (patience=10)
    """

    BATCH_SIZE          = 2048
    NUM_NEIGHBORS       = [15, 10, 5]   # One per GNN layer
    LEARNING_RATE       = 1e-3
    WEIGHT_DECAY        = 1e-4
    MAX_EPOCHS          = 100
    PATIENCE            = 10
    FRAUD_CLASS_WEIGHT  = 2.33          # 70/30 imbalance correction
    HIDDEN_CHANNELS     = 256
    NUM_LAYERS          = 3
    DROPOUT             = 0.30

    def __init__(self, artifact_dir: str = "artifacts/gnn",
                 mlflow_experiment: str = "fraud-gnn",
                 device: str = "auto"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_experiment = mlflow_experiment
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else
            "cpu" if device == "auto" else device
        )
        logger.info(f"GNN training device: {self.device}")
        self.model: Optional[HeteroGraphSAGE] = None
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0

    def _build_data_loaders(self, data: HeteroData, train_mask: torch.Tensor,
                             val_mask: torch.Tensor, test_mask: torch.Tensor):
        """Build NeighborLoader instances for mini-batch training."""
        common_kwargs = dict(
            data=data,
            num_neighbors={rel: self.NUM_NEIGHBORS for rel in data.edge_types},
            batch_size=self.BATCH_SIZE,
            input_node_type="account",
        )
        train_loader = NeighborLoader(
            **common_kwargs, input_nodes=("account", train_mask), shuffle=True
        )
        val_loader   = NeighborLoader(**common_kwargs, input_nodes=("account", val_mask))
        test_loader  = NeighborLoader(**common_kwargs, input_nodes=("account", test_mask))
        return train_loader, val_loader, test_loader

    def _temporal_split_masks(self, data: HeteroData,
                               account_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Temporal split: train on accounts with all transactions before cutoff,
        validate/test on accounts whose most recent transaction is after cutoff.
        """
        n = data["account"].num_nodes
        if "last_tx_date" in account_df.columns:
            dates    = pd.to_datetime(account_df["last_tx_date"])
            sorted_d = dates.sort_values()
            train_cutoff = sorted_d.iloc[int(n * 0.80)]
            val_cutoff   = sorted_d.iloc[int(n * 0.90)]

            train_mask = torch.tensor(dates <= train_cutoff, dtype=torch.bool)
            val_mask   = torch.tensor((dates > train_cutoff) & (dates <= val_cutoff), dtype=torch.bool)
            test_mask  = torch.tensor(dates > val_cutoff, dtype=torch.bool)
        else:
            # Random split fallback
            idx   = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            val_mask   = torch.zeros(n, dtype=torch.bool)
            test_mask  = torch.zeros(n, dtype=torch.bool)
            train_mask[idx[:int(n * 0.80)]] = True
            val_mask[idx[int(n * 0.80):int(n * 0.90)]] = True
            test_mask[idx[int(n * 0.90):]] = True

        logger.info(f"Split: train={train_mask.sum():,} | val={val_mask.sum():,} | test={test_mask.sum():,}")
        return train_mask, val_mask, test_mask

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            labels == 1,
            torch.tensor(self.FRAUD_CLASS_WEIGHT, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        return F.cross_entropy(logits, labels, reduction="none").mul(weights).mean()

    def _evaluate_loader(self, loader) -> Tuple[float, float, Dict]:
        """Evaluate model on a DataLoader split. Returns (AUC, F1, metrics_dict)."""
        from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x_dict, batch.edge_index_dict)
                # Only take seed nodes (first batch_size nodes in mini-batch)
                seed_mask = batch["account"].batch_size
                all_logits.append(logits[:seed_mask].cpu())
                all_labels.append(batch["account"].y[:seed_mask].cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels).numpy()
        probs  = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds  = (probs >= 0.5).astype(int)

        auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        f1     = f1_score(labels, preds, average="macro", zero_division=0)

        return auc, f1, {"auc": auc, "pr_auc": pr_auc, "f1_macro": f1,
                          "fraud_rate": float(labels.mean())}

    def train(self, data: HeteroData, account_df: pd.DataFrame) -> Dict:
        """
        Full GNN training loop with early stopping and model checkpointing.
        """
        mlflow.set_experiment(self.mlflow_experiment)

        with mlflow.start_run(run_name="graphsage-fraud-training"):
            # Setup
            train_mask, val_mask, test_mask = self._temporal_split_masks(data, account_df)
            train_loader, val_loader, test_loader = self._build_data_loaders(
                data, train_mask, val_mask, test_mask
            )

            self.model = HeteroGraphSAGE(
                hidden_channels=self.HIDDEN_CHANNELS,
                num_layers=self.NUM_LAYERS,
                dropout=self.DROPOUT,
            ).to(self.device)

            optimizer = AdamW(self.model.parameters(),
                              lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.MAX_EPOCHS, eta_min=1e-5)

            mlflow.log_params({
                "hidden_channels": self.HIDDEN_CHANNELS,
                "num_layers": self.NUM_LAYERS,
                "dropout": self.DROPOUT,
                "batch_size": self.BATCH_SIZE,
                "num_neighbors": str(self.NUM_NEIGHBORS),
                "learning_rate": self.LEARNING_RATE,
                "weight_decay": self.WEIGHT_DECAY,
                "fraud_class_weight": self.FRAUD_CLASS_WEIGHT,
            })

            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            best_model_state = None

            for epoch in range(1, self.MAX_EPOCHS + 1):
                # ── Training epoch ────────────────────────────────────────────
                self.model.train()
                total_loss, n_batches = 0.0, 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(batch.x_dict, batch.edge_index_dict)
                    seed_n = batch["account"].batch_size
                    loss   = self._compute_loss(
                        logits[:seed_n], batch["account"].y[:seed_n]
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches  += 1
                scheduler.step()

                avg_loss = total_loss / max(n_batches, 1)

                # ── Validation ───────────────────────────────────────────────
                val_auc, val_f1, val_metrics = self._evaluate_loader(val_loader)

                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "val_auc": val_auc,
                    "val_f1": val_f1,
                    "val_pr_auc": val_metrics["pr_auc"],
                    "lr": scheduler.get_last_lr()[0],
                }, step=epoch)

                logger.info(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                            f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

                # ── Early stopping ────────────────────────────────────────────
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.epochs_no_improve = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    torch.save(best_model_state, self.artifact_dir / "gnn_best.pt")
                    logger.info(f"  ✓ New best val AUC: {val_auc:.4f} — checkpoint saved")
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.PATIENCE:
                        logger.info(f"Early stopping at epoch {epoch} (patience={self.PATIENCE})")
                        break

            # ── Test evaluation ───────────────────────────────────────────────
            if best_model_state:
                self.model.load_state_dict(best_model_state)

            test_auc, test_f1, test_metrics = self._evaluate_loader(test_loader)
            logger.info(f"\nTest AUC: {test_auc:.4f} | Test F1: {test_f1:.4f} | "
                        f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
            mlflow.log_metrics({
                "test_auc": test_auc, "test_f1": test_f1, "test_pr_auc": test_metrics["pr_auc"]
            })

            return {"test_auc": test_auc, "test_f1_macro": test_f1, **test_metrics}

    def save(self, version: str = "v1"):
        torch.save(self.model.state_dict(), self.artifact_dir / f"gnn_model_{version}.pt")
        model_config = {
            "hidden_channels": self.HIDDEN_CHANNELS,
            "num_layers": self.NUM_LAYERS,
            "dropout": self.DROPOUT,
        }
        with open(self.artifact_dir / f"gnn_config_{version}.json", "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"GNN saved → {self.artifact_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH INFERENCE → REDIS CACHE
# ─────────────────────────────────────────────────────────────────────────────

class GNNBatchInference:
    """
    Hourly batch inference: run GraphSAGE on full graph, push scores to Redis.
    Each account node gets a pre-computed fraud risk score stored in Redis
    for <5ms lookup at transaction scoring time.

    Inference schedule: every hour (Airflow DAG)
    Full graph inference time: ~8 minutes (15M nodes, 85M edges, A100)
    """

    def __init__(self, artifact_dir: str, redis_client, version: str = "v1"):
        self.artifact_dir = Path(artifact_dir)
        self.redis        = redis_client
        self.version      = version
        self.model        = self._load_model(version)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_model(self, version: str) -> HeteroGraphSAGE:
        config_path = self.artifact_dir / f"gnn_config_{version}.json"
        with open(config_path) as f:
            config = json.load(f)
        model = HeteroGraphSAGE(**config)
        state = torch.load(self.artifact_dir / f"gnn_model_{version}.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def run_batch_inference(self, data: HeteroData,
                             account_id_hashes: List[str],
                             ttl_seconds: int = 7200) -> int:
        """
        Run full-graph inference and push all account scores to Redis.
        Returns number of scores written.
        """
        loader = NeighborLoader(
            data=data,
            num_neighbors={rel: [15, 10, 5] for rel in data.edge_types},
            batch_size=4096,
            input_node_type="account",
            input_nodes=("account", torch.arange(data["account"].num_nodes)),
            shuffle=False,
        )

        all_scores  = []
        all_node_ids = []

        with torch.no_grad():
            for batch in loader:
                batch  = batch.to(self.device)
                logits = self.model(batch.x_dict, batch.edge_index_dict)
                seed_n = batch["account"].batch_size
                probs  = torch.softmax(logits[:seed_n], dim=-1)[:, 1].cpu().numpy()
                orig_ids = batch["account"].n_id[:seed_n].numpy()

                all_scores.extend(probs.tolist())
                all_node_ids.extend(orig_ids.tolist())

        # Write to Redis
        pipe     = self.redis.pipeline(transaction=False)
        written  = 0
        for node_idx, score in zip(all_node_ids, all_scores):
            if node_idx < len(account_id_hashes):
                acct_hash = account_id_hashes[node_idx]
                redis_key = f"gnn:score:{acct_hash}"
                pipe.setex(redis_key, ttl_seconds, f"{score:.6f}")
                written  += 1

        pipe.execute()
        logger.info(f"GNN batch inference complete: {written:,} scores → Redis (TTL={ttl_seconds}s)")
        return written
