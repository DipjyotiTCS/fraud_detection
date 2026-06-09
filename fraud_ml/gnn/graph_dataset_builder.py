"""Graph dataset builder for the fraud GNN.

This builder converts the XGBoost raw transaction CSV into a graph-oriented
set of CSV files. It is designed for local development and for the synthetic
pipeline generated in this project. Later, the same output contract can be fed
from Neo4j or a production graph store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_ml.gnn.graph_dataset_builder")


@dataclass
class GraphDatasetBuildConfig:
    source_transaction_csv: str = "data/xgboost/xgboost_training_data.csv"
    output_dir: str = "data/gnn"
    dataset_name: str = "fraud_graph"
    seed: int = 42
    max_shared_entity_edges: int = 250_000


ACCOUNT_FEATURE_COLUMNS = [
    "account_age_days",
    "kyc_tier",
    "account_type",
    "relationship_years",
    "lifetime_tx_count",
    "lifetime_amount_log",
    "prior_sar_count",
    "prior_alert_count",
    "prior_confirmed_fraud",
    "monthly_avg_tx_count",
    "monthly_avg_amount_log",
    "geographic_risk_score",
    "pep_flag",
    "industry_risk_score",
    "distinct_devices_90d",
    "avg_failed_auth_monthly",
    "is_new_account",
]


class GraphDatasetBuilder:
    """Build account/device/IP/merchant nodes and graph edges from transactions."""

    def __init__(self, config: GraphDatasetBuildConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.seed)

    def build(self) -> Dict[str, Any]:
        source_path = Path(self.config.source_transaction_csv)
        if not source_path.exists():
            raise FileNotFoundError(f"Transaction CSV not found: {source_path}")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tx = pd.read_csv(source_path, parse_dates=["transaction_dt"])
        if tx.empty:
            raise ValueError("Source transaction CSV is empty")

        tx = tx.copy()
        tx["transaction_dt"] = pd.to_datetime(tx["transaction_dt"], utc=True)
        tx["ip_node_id"] = [self._hash(f"ip-{acct[:16]}-{int(i) % max(100, len(tx)//40)}") for i, acct in enumerate(tx["account_id_hash"].astype(str))]
        tx["merchant_node_id"] = "mcc_" + tx["merchant_category_code"].fillna("UNKNOWN").astype(str)

        account_nodes = self._build_account_nodes(tx)
        device_nodes = self._build_device_nodes(tx)
        ip_nodes = self._build_ip_nodes(tx)
        merchant_nodes = self._build_merchant_nodes(tx)
        account_edges = self._build_account_account_edges(tx)
        device_edges = self._build_account_device_edges(tx)
        ip_edges = self._build_account_ip_edges(tx)
        merchant_edges = self._build_account_merchant_edges(tx)
        derived_edges = self._build_shared_entity_account_edges(device_edges, ip_edges, merchant_edges)

        files = {
            "account_nodes": self._write_csv(account_nodes, f"{self.config.dataset_name}_account_nodes", run_id),
            "device_nodes": self._write_csv(device_nodes, f"{self.config.dataset_name}_device_nodes", run_id),
            "ip_nodes": self._write_csv(ip_nodes, f"{self.config.dataset_name}_ip_nodes", run_id),
            "merchant_nodes": self._write_csv(merchant_nodes, f"{self.config.dataset_name}_merchant_nodes", run_id),
            "account_account_edges": self._write_csv(account_edges, f"{self.config.dataset_name}_edges_account_account", run_id),
            "account_device_edges": self._write_csv(device_edges, f"{self.config.dataset_name}_edges_account_device", run_id),
            "account_ip_edges": self._write_csv(ip_edges, f"{self.config.dataset_name}_edges_account_ip", run_id),
            "account_merchant_edges": self._write_csv(merchant_edges, f"{self.config.dataset_name}_edges_account_merchant", run_id),
            "derived_account_edges": self._write_csv(derived_edges, f"{self.config.dataset_name}_edges_account_shared_entity", run_id),
        }

        metadata = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "source_transaction_csv": str(source_path),
            "counts": {
                "transactions": int(len(tx)),
                "account_nodes": int(len(account_nodes)),
                "device_nodes": int(len(device_nodes)),
                "ip_nodes": int(len(ip_nodes)),
                "merchant_nodes": int(len(merchant_nodes)),
                "account_account_edges": int(len(account_edges)),
                "derived_account_edges": int(len(derived_edges)),
            },
            "files": files,
        }
        metadata_path = self.output_dir / f"{self.config.dataset_name}_metadata_{run_id}.json"
        latest_metadata_path = self.output_dir / f"{self.config.dataset_name}_metadata_latest.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        latest_metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        metadata["metadata_path"] = str(metadata_path)
        metadata["latest_metadata_path"] = str(latest_metadata_path)
        return metadata

    def _build_account_nodes(self, tx: pd.DataFrame) -> pd.DataFrame:
        origin = tx.groupby("account_id_hash").agg(
            first_tx=("transaction_dt", "min"),
            last_tx=("transaction_dt", "max"),
            account_age_days=("days_since_account_open", "median"),
            lifetime_tx_count=("transaction_id", "count"),
            lifetime_amount=("amount_inr", "sum"),
            prior_sar_count=("beneficiary_sar_count", "max"),
            prior_alert_count=("second_hop_sar_count", "max"),
            prior_confirmed_fraud=("label_is_fraud", "max"),
            monthly_avg_tx_count=("typical_velocity_daily", "mean"),
            monthly_avg_amount=("amount_mean_90d", "mean"),
            geographic_risk_score=("ip_reputation_score", "mean"),
            pep_flag=("watchlist_hit", "max"),
            industry_risk_score=("synthetic_identity_score", "mean"),
            distinct_devices_90d=("distinct_devices_30d", "max"),
            avg_failed_auth_monthly=("failed_auth_count_7d", "mean"),
            node_label=("label_is_fraud", "max"),
            label_source=("label_source", lambda s: s.mode().iloc[0] if len(s.mode()) else "SYNTHETIC"),
        ).reset_index().rename(columns={"account_id_hash": "node_id"})

        # Beneficiaries become account nodes too. They get less profile detail, but labels if linked to fraud.
        bene = tx.groupby("beneficiary_id_hash").agg(
            first_tx=("transaction_dt", "min"),
            last_tx=("transaction_dt", "max"),
            lifetime_tx_count=("transaction_id", "count"),
            lifetime_amount=("amount_inr", "sum"),
            node_label=("label_is_fraud", "max"),
            label_source=("label_source", lambda s: s.mode().iloc[0] if len(s.mode()) else "SYNTHETIC"),
        ).reset_index().rename(columns={"beneficiary_id_hash": "node_id"})
        for col in origin.columns:
            if col not in bene.columns:
                bene[col] = np.nan
        accounts = pd.concat([origin, bene[origin.columns]], ignore_index=True)
        accounts = accounts.sort_values("node_id").groupby("node_id", as_index=False).agg({
            "first_tx": "min",
            "last_tx": "max",
            "account_age_days": "median",
            "lifetime_tx_count": "sum",
            "lifetime_amount": "sum",
            "prior_sar_count": "max",
            "prior_alert_count": "max",
            "prior_confirmed_fraud": "max",
            "monthly_avg_tx_count": "mean",
            "monthly_avg_amount": "mean",
            "geographic_risk_score": "mean",
            "pep_flag": "max",
            "industry_risk_score": "mean",
            "distinct_devices_90d": "max",
            "avg_failed_auth_monthly": "mean",
            "node_label": "max",
            "label_source": lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) else "SYNTHETIC",
        })
        accounts["account_age_days"] = accounts["account_age_days"].fillna(365).clip(0, 36500)
        accounts["relationship_years"] = (accounts["account_age_days"] / 365.0).clip(0, 100)
        accounts["lifetime_amount_log"] = np.log1p(accounts["lifetime_amount"].fillna(0))
        accounts["monthly_avg_amount_log"] = np.log1p(accounts["monthly_avg_amount"].fillna(accounts["lifetime_amount"] / 12.0))
        accounts["prior_sar_count"] = accounts["prior_sar_count"].fillna(0).clip(0, 127)
        accounts["prior_alert_count"] = accounts["prior_alert_count"].fillna(0).clip(0, 32767)
        accounts["prior_confirmed_fraud"] = accounts["prior_confirmed_fraud"].fillna(0).astype(int)
        accounts["monthly_avg_tx_count"] = accounts["monthly_avg_tx_count"].fillna(accounts["lifetime_tx_count"] / 12.0)
        accounts["geographic_risk_score"] = accounts["geographic_risk_score"].fillna(0.1).clip(0, 1)
        accounts["pep_flag"] = accounts["pep_flag"].fillna(False).astype(int)
        accounts["industry_risk_score"] = accounts["industry_risk_score"].fillna(0).clip(0, 1)
        accounts["distinct_devices_90d"] = accounts["distinct_devices_90d"].fillna(1).clip(0, 10000)
        accounts["avg_failed_auth_monthly"] = accounts["avg_failed_auth_monthly"].fillna(0).clip(0, 10000)
        accounts["kyc_tier"] = np.where(accounts["account_age_days"] > 730, 3, np.where(accounts["account_age_days"] > 90, 2, 1))
        accounts["account_type"] = np.where(accounts["monthly_avg_amount_log"] > 12.5, 1, 0)
        accounts["is_new_account"] = (accounts["account_age_days"] < 90).astype(int)
        accounts["node_label"] = accounts["node_label"].fillna(0).astype(int)
        keep = ["node_id"] + ACCOUNT_FEATURE_COLUMNS + ["node_label", "label_source", "first_tx", "last_tx"]
        return accounts[keep]

    def _build_device_nodes(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.groupby("device_fingerprint_hash").agg(
            first_seen=("transaction_dt", "min"),
            associated_account_count=("account_id_hash", "nunique"),
            device_risk_score=("behavioural_drift_score", "mean"),
            has_root_jailbreak=("is_tor_vpn_proxy", "max"),
        ).reset_index().rename(columns={"device_fingerprint_hash": "node_id"})
        last_dt = tx["transaction_dt"].max()
        df["first_seen_days_ago"] = ((last_dt - df["first_seen"]).dt.total_seconds() / 86400).clip(0).astype(int)
        df["platform"] = [int(self._stable_int(v, 4)) for v in df["node_id"]]
        df["device_risk_score"] = df["device_risk_score"].clip(0, 1)
        df["has_root_jailbreak"] = df["has_root_jailbreak"].astype(int)
        return df[["node_id", "first_seen_days_ago", "associated_account_count", "platform", "device_risk_score", "has_root_jailbreak"]]

    def _build_ip_nodes(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.groupby("ip_node_id").agg(
            reputation_score=("ip_reputation_score", "mean"),
            is_tor=("is_tor_vpn_proxy", "max"),
            is_vpn=("is_tor_vpn_proxy", "max"),
            is_proxy=("is_tor_vpn_proxy", "max"),
            country_risk_score=("geo_country_mismatch", "mean"),
            associated_account_count=("account_id_hash", "nunique"),
        ).reset_index().rename(columns={"ip_node_id": "node_id"})
        for c in ["is_tor", "is_vpn", "is_proxy"]:
            df[c] = df[c].astype(int)
        df["country_risk_score"] = df["country_risk_score"].clip(0, 1)
        df["reputation_score"] = df["reputation_score"].clip(0, 1)
        return df

    def _build_merchant_nodes(self, tx: pd.DataFrame) -> pd.DataFrame:
        risk_by_mcc = tx.groupby("merchant_node_id")["label_is_fraud"].mean().to_dict()
        df = tx.groupby(["merchant_node_id", "merchant_category_code"]).agg(
            tx_count=("transaction_id", "count"),
            chargeback_rate_90d=("label_is_fraud", "mean"),
            dispute_count_90d=("second_hop_sar_count", "sum"),
        ).reset_index().rename(columns={"merchant_node_id": "node_id", "merchant_category_code": "mcc_code"})
        df["merchant_age_days"] = [30 + self._stable_int(v, 3650) for v in df["node_id"]]
        df["mcc_risk_score"] = df["node_id"].map(risk_by_mcc).fillna(0).clip(0, 1)
        df["chargeback_rate_90d"] = df["chargeback_rate_90d"].clip(0, 1)
        df["dispute_count_90d"] = df["dispute_count_90d"].clip(0, 32767).astype(int)
        return df[["node_id", "mcc_code", "merchant_age_days", "mcc_risk_score", "chargeback_rate_90d", "dispute_count_90d"]]

    def _build_account_account_edges(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.groupby(["account_id_hash", "beneficiary_id_hash"]).agg(
            transaction_count_30d=("transaction_id", "count"),
            total_amount_30d=("amount_inr", "sum"),
            first_tx=("transaction_dt", "min"),
            last_tx=("transaction_dt", "max"),
            avg_amount=("amount_inr", "mean"),
            edge_risk_score=("label_is_fraud", "mean"),
        ).reset_index().rename(columns={"account_id_hash": "src_node_id", "beneficiary_id_hash": "dst_node_id"})
        max_dt = tx["transaction_dt"].max()
        df["total_amount_30d_log"] = np.log1p(df["total_amount_30d"])
        df["first_transaction_days_ago"] = ((max_dt - df["first_tx"]).dt.total_seconds() / 86400).clip(0)
        df["last_transaction_hours_ago"] = ((max_dt - df["last_tx"]).dt.total_seconds() / 3600).clip(0)
        df["avg_amount_log"] = np.log1p(df["avg_amount"])
        df["is_recurring"] = (df["transaction_count_30d"] >= 2).astype(int)
        return df[["src_node_id", "dst_node_id", "transaction_count_30d", "total_amount_30d_log", "first_transaction_days_ago", "last_transaction_hours_ago", "avg_amount_log", "is_recurring", "edge_risk_score"]]

    def _build_account_device_edges(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.groupby(["account_id_hash", "device_fingerprint_hash"]).agg(
            session_count_30d=("transaction_id", "count"),
            last_session=("transaction_dt", "max"),
        ).reset_index().rename(columns={"account_id_hash": "src_node_id", "device_fingerprint_hash": "dst_node_id"})
        max_dt = tx["transaction_dt"].max()
        df["last_session_hours_ago"] = ((max_dt - df["last_session"]).dt.total_seconds() / 3600).clip(0)
        df["is_primary_device"] = df.groupby("src_node_id")["session_count_30d"].rank(method="first", ascending=False).eq(1).astype(int)
        return df[["src_node_id", "dst_node_id", "session_count_30d", "last_session_hours_ago", "is_primary_device"]]

    def _build_account_ip_edges(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.groupby(["account_id_hash", "ip_node_id"]).agg(
            session_count_7d=("transaction_id", "count"),
            last_session=("transaction_dt", "max"),
        ).reset_index().rename(columns={"account_id_hash": "src_node_id", "ip_node_id": "dst_node_id"})
        max_dt = tx["transaction_dt"].max()
        df["last_session_hours_ago"] = ((max_dt - df["last_session"]).dt.total_seconds() / 3600).clip(0)
        return df[["src_node_id", "dst_node_id", "session_count_7d", "last_session_hours_ago"]]

    def _build_account_merchant_edges(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.groupby(["account_id_hash", "merchant_node_id"]).agg(
            transaction_count_90d=("transaction_id", "count"),
            total_amount_90d=("amount_inr", "sum"),
            first_seen=("transaction_dt", "min"),
        ).reset_index().rename(columns={"account_id_hash": "src_node_id", "merchant_node_id": "dst_node_id"})
        df["total_amount_90d_log"] = np.log1p(df["total_amount_90d"])
        df["is_first_transaction"] = (df["transaction_count_90d"] == 1).astype(int)
        return df[["src_node_id", "dst_node_id", "transaction_count_90d", "total_amount_90d_log", "is_first_transaction"]]

    def _build_shared_entity_account_edges(self, device_edges: pd.DataFrame, ip_edges: pd.DataFrame, merchant_edges: pd.DataFrame) -> pd.DataFrame:
        if self.config.max_shared_entity_edges <= 0:
            return pd.DataFrame(columns=["src_node_id", "dst_node_id", "relation", "shared_entity_id"])
        frames = []
        for df, rel in [(device_edges, "SHARES_DEVICE"), (ip_edges, "SHARES_IP"), (merchant_edges, "SHARES_MERCHANT")]:
            if df.empty:
                continue
            pairs = []
            for entity_id, group in df.groupby("dst_node_id"):
                accounts = group["src_node_id"].drop_duplicates().tolist()
                if len(accounts) < 2:
                    continue
                # Limit clique expansion for high-degree entities.
                accounts = accounts[:60]
                for i in range(len(accounts) - 1):
                    pairs.append((accounts[i], accounts[i + 1], rel, entity_id))
                    if len(pairs) >= self.config.max_shared_entity_edges // 3:
                        break
                if len(pairs) >= self.config.max_shared_entity_edges // 3:
                    break
            if pairs:
                frames.append(pd.DataFrame(pairs, columns=["src_node_id", "dst_node_id", "relation", "shared_entity_id"]))
        if not frames:
            return pd.DataFrame(columns=["src_node_id", "dst_node_id", "relation", "shared_entity_id"])
        out = pd.concat(frames, ignore_index=True).drop_duplicates()
        return out.head(self.config.max_shared_entity_edges)

    def _write_csv(self, df: pd.DataFrame, base_name: str, run_id: str) -> Dict[str, Any]:
        path = self.output_dir / f"{base_name}_{run_id}.csv"
        latest = self.output_dir / f"{base_name}.csv"
        df.to_csv(path, index=False)
        shutil.copyfile(path, latest)
        return {"csv_path": str(path), "latest_csv_path": str(latest), "rows": int(len(df))}

    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _stable_int(self, value: str, modulo: int) -> int:
        return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8], 16) % modulo
