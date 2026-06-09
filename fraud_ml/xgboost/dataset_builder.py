"""Dataset builder for the XGBoost fraud model.

This module can generate a synthetic but schema-valid raw transaction dataset for
local development, demos, and smoke-testing the XGBoost training pipeline.

In production, keep the public interface of XGBoostDatasetBuilder and replace the
synthetic source with extractors from your transaction lake, customer profile
store, velocity feature store, enrichment services, graph store, and label store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from fraud_ml.shared.schemas import (
    FraudTypology,
    XGBOOST_ENGINEERED_FEATURES,
    XGBOOST_RAW_SCHEMA,
    DataQualityValidator,
)
from fraud_ml.xgboost.feature_engineering import XGBoostFeatureEngineer, DatasetSplitter

logger = logging.getLogger("fraud_ml.xgboost.dataset_builder")


@dataclass
class DatasetBuildConfig:
    n_rows: int = 10_000
    fraud_rate: float = 0.12
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42
    output_dir: str = "data/xgboost"
    dataset_name: str = "xgboost_training_data"
    generate_engineered_csv: bool = True
    generate_split_csvs: bool = True
    temporal_split: bool = True
    # synthetic: generate graph fields from fraud-aware distributions.
    # none: neutralize graph fields for a no-graph baseline.
    # gnn_scores: overwrite graph probability fields from a GNN score CSV.
    graph_feature_mode: str = "synthetic"
    graph_score_path: Optional[str] = None


class XGBoostDatasetBuilder:
    """Build raw, engineered, and split CSV datasets for XGBoost training."""

    CHANNELS = np.array(["IMPS", "NEFT", "RTGS", "UPI", "CARD_CP", "CARD_CNP", "WIRE", "CRYPTO", "P2P"])
    CURRENCIES = np.array(["INR", "USD", "EUR", "GBP", "AED", "SGD"])
    MCC_CODES = np.array(["5411", "5812", "5999", "6012", "6211", "6536", "7995", "4829", "5734", "7399"])
    WATCHLIST_CATEGORIES = np.array(["OFAC_SDN", "UN_CONSOLIDATED", "PEP_TIER1", "PEP_TIER2", "FIU_IND", "INTERNAL", "NONE"])
    LABEL_SOURCES = np.array(["CONFIRMED_SAR", "INVESTIGATOR", "CARD_NETWORK", "NPCI", "CONSORTIUM", "SYNTHETIC"])
    FRAUD_TYPOLOGIES = np.array([
        FraudTypology.ACCOUNT_TAKEOVER.value,
        FraudTypology.STRUCTURING.value,
        FraudTypology.MONEY_MULE.value,
        FraudTypology.CARD_NOT_PRESENT.value,
        FraudTypology.SYNTHETIC_ID.value,
        FraudTypology.BEC.value,
        FraudTypology.TRADE_BASED_ML.value,
        FraudTypology.CRYPTO_LAYERING.value,
        FraudTypology.ROMANCE_SCAM.value,
        FraudTypology.UPI_FRAUD.value,
        FraudTypology.FIRST_PARTY.value,
    ])

    def __init__(self, config: DatasetBuildConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.seed)

    def build(self) -> Dict[str, Any]:
        """Generate dataset bundle and return metadata/paths."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_df = self.generate_synthetic_raw_dataset()
        raw_df = self._apply_graph_feature_mode(raw_df)

        validator = DataQualityValidator()
        quality_report = validator.validate_xgboost_dataset(raw_df)

        raw_path = self.output_dir / f"{self.config.dataset_name}_{run_id}.csv"
        latest_path = self.output_dir / f"{self.config.dataset_name}.csv"
        raw_df.to_csv(raw_path, index=False)
        shutil.copyfile(raw_path, latest_path)

        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "row_count": int(len(raw_df)),
            "fraud_count": int(raw_df["label_is_fraud"].sum()),
            "fraud_rate": float(raw_df["label_is_fraud"].mean()),
            "raw_csv_path": str(raw_path),
            "latest_raw_csv_path": str(latest_path),
            "schema_columns": list(XGBOOST_RAW_SCHEMA.keys()),
            "quality_report": quality_report.__dict__,
        }

        if self.config.generate_engineered_csv or self.config.generate_split_csvs:
            engineer = XGBoostFeatureEngineer(artifact_dir=str(self.output_dir / "feature_engineering_artifacts"))
            engineer.fit(raw_df)
            X, y = engineer.transform(raw_df, is_training=True)
            weights = engineer.compute_sample_weights(raw_df)

            if self.config.generate_engineered_csv:
                engineered_df = X.copy()
                engineered_df.insert(0, "transaction_id", raw_df["transaction_id"].values)
                engineered_df["label_is_fraud"] = y
                engineered_path = self.output_dir / f"{self.config.dataset_name}_engineered_{run_id}.csv"
                engineered_latest_path = self.output_dir / f"{self.config.dataset_name}_engineered.csv"
                engineered_df.to_csv(engineered_path, index=False)
                shutil.copyfile(engineered_path, engineered_latest_path)
                metadata["engineered_csv_path"] = str(engineered_path)
                metadata["latest_engineered_csv_path"] = str(engineered_latest_path)
                metadata["engineered_feature_count"] = len(XGBOOST_ENGINEERED_FEATURES)

            if self.config.generate_split_csvs:
                splitter = DatasetSplitter(temporal_split=self.config.temporal_split)
                splits = splitter.split(raw_df, X, y, weights)
                split_paths = {}
                for split_name, split_data in splits.items():
                    split_df = split_data["df"].copy()
                    split_df.to_csv(self.output_dir / f"{self.config.dataset_name}_{split_name}_{run_id}.csv", index=False)
                    shutil.copyfile(
                        self.output_dir / f"{self.config.dataset_name}_{split_name}_{run_id}.csv",
                        self.output_dir / f"{self.config.dataset_name}_{split_name}.csv",
                    )
                    split_paths[split_name] = {
                        "csv_path": str(self.output_dir / f"{self.config.dataset_name}_{split_name}_{run_id}.csv"),
                        "latest_csv_path": str(self.output_dir / f"{self.config.dataset_name}_{split_name}.csv"),
                        "n_total": int(split_data["n_total"]),
                        "n_fraud": int(split_data["n_fraud"]),
                        "fraud_rate": float(split_data["fraud_rate"]),
                    }
                metadata["split_paths"] = split_paths

        metadata_path = self.output_dir / f"dataset_metadata_{run_id}.json"
        latest_metadata_path = self.output_dir / "dataset_metadata_latest.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        latest_metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        metadata["metadata_path"] = str(metadata_path)
        metadata["latest_metadata_path"] = str(latest_metadata_path)
        return metadata

    def _apply_graph_feature_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply graph feature mode before writing the XGBoost raw CSV."""
        mode = (self.config.graph_feature_mode or "synthetic").lower()
        df = df.copy()
        if mode == "synthetic":
            return df
        if mode == "none":
            df["shared_device_accounts"] = 0
            df["shared_ip_accounts"] = 0
            df["beneficiary_sar_count"] = 0
            df["second_hop_sar_count"] = 0
            df["mule_network_probability"] = 0.0
            df["synthetic_identity_score"] = 0.0
            df["rapid_fan_out_flag"] = False
            df["originator_graph_centrality"] = 0.0
            return df
        if mode == "gnn_scores":
            if not self.config.graph_score_path:
                raise ValueError("graph_score_path is required when graph_feature_mode='gnn_scores'")
            score_path = Path(self.config.graph_score_path)
            if not score_path.exists():
                raise FileNotFoundError(f"GNN score CSV not found: {score_path}")
            scores = pd.read_csv(score_path)
            if "account_id_hash" not in scores.columns or "gnn_fraud_probability" not in scores.columns:
                raise ValueError("GNN score CSV must contain account_id_hash and gnn_fraud_probability")
            score_map = scores.set_index("account_id_hash")["gnn_fraud_probability"].to_dict()
            p = df["account_id_hash"].map(score_map).fillna(0.0).astype(float).clip(0, 1)
            df["mule_network_probability"] = p.round(4)
            df["originator_graph_centrality"] = np.maximum(df["originator_graph_centrality"], np.sqrt(p) * 0.85).clip(0, 1).round(4)
            df["synthetic_identity_score"] = np.maximum(df["synthetic_identity_score"], p * 0.72).clip(0, 1).round(4)
            df["second_hop_sar_count"] = np.maximum(df["second_hop_sar_count"], np.round(p * 8)).astype(int)
            df["rapid_fan_out_flag"] = df["rapid_fan_out_flag"].astype(bool) | (p >= 0.72)
            return df
        raise ValueError("graph_feature_mode must be one of: synthetic, none, gnn_scores")

    def generate_synthetic_raw_dataset(self) -> pd.DataFrame:
        """Generate a schema-valid synthetic raw transaction DataFrame."""
        n = self.config.n_rows
        if n < 200:
            raise ValueError("n_rows should be at least 200 so train/val/test splits contain both classes")
        if not 0.01 <= self.config.fraud_rate <= 0.49:
            raise ValueError("fraud_rate must be between 0.01 and 0.49")

        start = pd.Timestamp(self.config.start_date, tz="UTC")
        end = pd.Timestamp(self.config.end_date, tz="UTC")
        if end <= start:
            raise ValueError("end_date must be after start_date")

        transaction_dt = self._random_timestamps(start, end, n)
        labels = self.rng.binomial(1, self.config.fraud_rate, n).astype(np.int8)
        fraud_mask = labels == 1
        benign_mask = ~fraud_mask

        account_ids = self._hashed_ids("acct", self.rng.integers(0, max(1, n // 4), n))
        beneficiary_ids = self._hashed_ids("bene", self.rng.integers(0, max(1, n // 3), n))
        device_ids = self._hashed_ids("device", self.rng.integers(0, max(1, n // 5), n))
        transaction_ids = [self._hash(f"txn-{self.config.seed}-{i}") for i in range(n)]

        # Amounts: benign mostly normal retail/banking; fraud includes bursts, threshold proximity, and high-value transfers.
        amount_inr = self.rng.lognormal(mean=10.0, sigma=0.9, size=n)
        amount_inr[fraud_mask] = self.rng.lognormal(mean=12.1, sigma=1.0, size=fraud_mask.sum())
        structuring_idx = np.where(fraud_mask & (self.rng.random(n) < 0.35))[0]
        if len(structuring_idx):
            amount_inr[structuring_idx] = self.rng.uniform(850_000, 999_500, len(structuring_idx))
        amount_inr = np.clip(amount_inr, 10, 95_000_000)
        amount_usd = np.clip(amount_inr / 83.0, 0.001, 1_190_000)

        channel = self._choice_by_mask(
            n,
            benign_probs=[0.14, 0.19, 0.04, 0.34, 0.10, 0.08, 0.04, 0.01, 0.06],
            fraud_probs=[0.09, 0.03, 0.03, 0.24, 0.03, 0.20, 0.15, 0.16, 0.07],
            fraud_mask=fraud_mask,
            values=self.CHANNELS,
        )

        is_international = self.rng.random(n) < np.where(fraud_mask, 0.28, 0.04)
        currency = np.where(is_international, self.rng.choice(self.CURRENCIES[1:], n), "INR")
        merchant_category_code = self.rng.choice(self.MCC_CODES, n)
        is_new_beneficiary = self.rng.random(n) < np.where(fraud_mask, 0.57, 0.12)
        days_since_account_open = self._integer_feature(n, benign_lambda=1600, fraud_lambda=380, max_value=36500, fraud_mask=fraud_mask)

        velocity_1h = self._integer_feature(n, 1.2, 6.5, 10000, fraud_mask)
        velocity_6h = velocity_1h + self._integer_feature(n, 2.5, 13.0, 50000, fraud_mask)
        velocity_24h = velocity_6h + self._integer_feature(n, 4.0, 28.0, 200000, fraud_mask)
        velocity_7d = velocity_24h + self._integer_feature(n, 10.0, 60.0, 1_000_000, fraud_mask)
        amount_sum_1h = np.clip(amount_inr * np.maximum(velocity_1h, 1) * self.rng.uniform(0.7, 1.6, n), 0, 9.9e9)
        amount_sum_24h = np.clip(amount_sum_1h + amount_inr * np.maximum(velocity_24h, 1) * self.rng.uniform(0.5, 1.4, n), 0, 9.9e10)
        distinct_beneficiaries_6h = np.minimum(velocity_6h, self._integer_feature(n, 1.0, 5.0, 5000, fraud_mask))
        distinct_beneficiaries_24h = np.minimum(velocity_24h, distinct_beneficiaries_6h + self._integer_feature(n, 1.5, 8.0, 20000, fraud_mask))

        ip_reputation_score = self._beta_feature(n, benign_a=1.0, benign_b=8.5, fraud_a=4.5, fraud_b=2.2, fraud_mask=fraud_mask)
        is_tor_vpn_proxy = self.rng.random(n) < np.where(fraud_mask, 0.38, 0.03)
        geo_distance_km = self.rng.exponential(np.where(fraud_mask, 2600.0, 120.0), n).clip(0, 40000)
        geo_country_mismatch = self.rng.random(n) < np.where(fraud_mask, 0.34, 0.025)
        watchlist_hit = self.rng.random(n) < np.where(fraud_mask, 0.12, 0.006)
        watchlist_category = np.where(watchlist_hit, self.rng.choice(self.WATCHLIST_CATEGORIES[:-1], n), "NONE")
        failed_auth_count_7d = self._integer_feature(n, 0.4, 5.5, 1000, fraud_mask)

        amount_mean_90d = np.clip(amount_inr * self.rng.uniform(0.25, 1.25, n), 0, 9.9e7)
        amount_std_90d = np.clip(amount_mean_90d * self.rng.uniform(0.10, 0.65, n), 1, 9.9e7)
        typical_velocity_daily = np.where(benign_mask, self.rng.gamma(2, 2, n), self.rng.gamma(2, 5, n)).clip(0.2, 9999)
        is_dormant_account = self.rng.random(n) < np.where(fraud_mask, 0.18, 0.03)
        distinct_devices_30d = self._integer_feature(n, 1.0, 4.0, 200, fraud_mask)
        behavioural_drift_score = self._beta_feature(n, 1.1, 8.0, 4.8, 2.0, fraud_mask)
        peer_group_id = np.array([f"PG-{x:03d}" for x in self.rng.integers(1, 36, n)])
        peer_amount_percentile = np.where(
            fraud_mask,
            self.rng.beta(6.0, 1.6, n) * 100,
            self.rng.beta(2.4, 2.8, n) * 100,
        ).clip(0, 100)

        # Graph-derived features. In production these should come from GNN/Neo4j/Redis.
        shared_device_accounts = self._integer_feature(n, 1.0, 9.0, 10000, fraud_mask)
        shared_ip_accounts = self._integer_feature(n, 1.2, 12.0, 10000, fraud_mask)
        beneficiary_sar_count = self._integer_feature(n, 0.05, 0.85, 1000, fraud_mask)
        second_hop_sar_count = self._integer_feature(n, 0.10, 2.2, 5000, fraud_mask)
        mule_network_probability = self._beta_feature(n, 0.8, 10.0, 4.5, 2.0, fraud_mask)
        synthetic_identity_score = self._beta_feature(n, 0.9, 11.0, 3.8, 2.3, fraud_mask)
        rapid_fan_out_flag = self.rng.random(n) < np.where(fraud_mask, 0.32, 0.015)
        originator_graph_centrality = self._beta_feature(n, 1.2, 9.0, 3.5, 3.0, fraud_mask)

        label_typology = np.where(labels == 0, FraudTypology.BENIGN.value, self.rng.choice(self.FRAUD_TYPOLOGIES, n))
        label_source = self._choice_by_mask(
            n,
            benign_probs=[0.02, 0.20, 0.15, 0.20, 0.25, 0.18],
            fraud_probs=[0.34, 0.24, 0.13, 0.12, 0.09, 0.08],
            fraud_mask=fraud_mask,
            values=self.LABEL_SOURCES,
        )
        label_confidence = np.where(
            label_source == "CONFIRMED_SAR",
            self.rng.uniform(0.92, 1.0, n),
            np.where(label_source == "SYNTHETIC", self.rng.uniform(0.62, 0.82, n), self.rng.uniform(0.78, 0.94, n)),
        ).clip(0.5, 1.0)

        df = pd.DataFrame({
            "transaction_id": transaction_ids,
            "account_id_hash": account_ids,
            "beneficiary_id_hash": beneficiary_ids,
            "device_fingerprint_hash": device_ids,
            "transaction_dt": transaction_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "amount_inr": amount_inr.round(2),
            "amount_usd_equiv": amount_usd.round(2),
            "channel": channel,
            "merchant_category_code": merchant_category_code,
            "is_international": is_international,
            "currency": currency,
            "is_new_beneficiary": is_new_beneficiary,
            "days_since_account_open": days_since_account_open.astype(np.int32),
            "velocity_1h": velocity_1h.astype(np.int32),
            "velocity_6h": velocity_6h.astype(np.int32),
            "velocity_24h": velocity_24h.astype(np.int32),
            "velocity_7d": velocity_7d.astype(np.int32),
            "amount_sum_1h": amount_sum_1h.round(2),
            "amount_sum_24h": amount_sum_24h.round(2),
            "distinct_beneficiaries_6h": distinct_beneficiaries_6h.astype(np.int32),
            "distinct_beneficiaries_24h": distinct_beneficiaries_24h.astype(np.int32),
            "ip_reputation_score": ip_reputation_score.round(4),
            "is_tor_vpn_proxy": is_tor_vpn_proxy,
            "geo_distance_km": geo_distance_km.round(3),
            "geo_country_mismatch": geo_country_mismatch,
            "watchlist_hit": watchlist_hit,
            "watchlist_category": watchlist_category,
            "failed_auth_count_7d": failed_auth_count_7d.astype(np.int32),
            "amount_mean_90d": amount_mean_90d.round(2),
            "amount_std_90d": amount_std_90d.round(2),
            "typical_velocity_daily": typical_velocity_daily.round(3),
            "is_dormant_account": is_dormant_account,
            "distinct_devices_30d": distinct_devices_30d.astype(np.int32),
            "behavioural_drift_score": behavioural_drift_score.round(4),
            "peer_group_id": peer_group_id,
            "peer_amount_percentile": peer_amount_percentile.round(3),
            "shared_device_accounts": shared_device_accounts.astype(np.int32),
            "shared_ip_accounts": shared_ip_accounts.astype(np.int32),
            "beneficiary_sar_count": beneficiary_sar_count.astype(np.int32),
            "second_hop_sar_count": second_hop_sar_count.astype(np.int32),
            "mule_network_probability": mule_network_probability.round(4),
            "synthetic_identity_score": synthetic_identity_score.round(4),
            "rapid_fan_out_flag": rapid_fan_out_flag,
            "originator_graph_centrality": originator_graph_centrality.round(4),
            "label_is_fraud": labels,
            "label_typology": label_typology,
            "label_source": label_source,
            "label_confidence": label_confidence.round(4),
        })

        # Strictly follow schema column order.
        return df[list(XGBOOST_RAW_SCHEMA.keys())]

    def _random_timestamps(self, start: pd.Timestamp, end: pd.Timestamp, n: int) -> pd.DatetimeIndex:
        start_ns = start.value
        end_ns = end.value
        values = self.rng.integers(start_ns, end_ns, n, dtype=np.int64)
        return pd.to_datetime(values, utc=True)

    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _hashed_ids(self, prefix: str, values: np.ndarray) -> list[str]:
        return [self._hash(f"{prefix}-{int(v)}") for v in values]

    def _integer_feature(self, n: int, benign_lambda: float, fraud_lambda: float, max_value: int, fraud_mask: np.ndarray) -> np.ndarray:
        out = self.rng.poisson(benign_lambda, n)
        out[fraud_mask] = self.rng.poisson(fraud_lambda, fraud_mask.sum())
        return np.clip(out, 0, max_value)

    def _beta_feature(self, n: int, benign_a: float, benign_b: float, fraud_a: float, fraud_b: float, fraud_mask: np.ndarray) -> np.ndarray:
        out = self.rng.beta(benign_a, benign_b, n)
        out[fraud_mask] = self.rng.beta(fraud_a, fraud_b, fraud_mask.sum())
        return out.clip(0, 1)

    def _choice_by_mask(self, n: int, benign_probs: list[float], fraud_probs: list[float], fraud_mask: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = self.rng.choice(values, n, p=np.array(benign_probs) / np.sum(benign_probs))
        out[fraud_mask] = self.rng.choice(values, fraud_mask.sum(), p=np.array(fraud_probs) / np.sum(fraud_probs))
        return out
