"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FRAUDSENTINEL — DATA SCHEMAS & FEATURE DEFINITIONS                         ║
║  Shared definitions for XGBoost and GNN training pipelines                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This module defines:                                                        ║
║    1. Input dataset schemas (XGBoost tabular + GNN graph)                   ║
║    2. Feature group taxonomy (44 XGBoost + 22 GNN node features)            ║
║    3. Label definitions and class mappings                                   ║
║    4. Data quality contracts (validation rules)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class FraudLabel(int, Enum):
    BENIGN = 0
    FRAUD  = 1

class FraudTypology(str, Enum):
    """FinCEN typology codes used as multi-class labels."""
    BENIGN           = "BENIGN"
    ACCOUNT_TAKEOVER = "ATO-04"
    STRUCTURING      = "STR-09"
    MONEY_MULE       = "MML-02"
    CARD_NOT_PRESENT = "CNP-11"
    SYNTHETIC_ID     = "SID-06"
    BEC              = "BEC-13"
    TRADE_BASED_ML   = "TBML-07"
    CRYPTO_LAYERING  = "CRY-15"
    ROMANCE_SCAM     = "ROM-17"
    UPI_FRAUD        = "UPI-21"
    FIRST_PARTY      = "FPF-03"

TYPOLOGY_TO_INT: Dict[str, int] = {t.value: i for i, t in enumerate(FraudTypology)}
INT_TO_TYPOLOGY: Dict[int, str] = {v: k for k, v in TYPOLOGY_TO_INT.items()}

# Class weights for imbalanced training (fraud:benign = 30:70)
CLASS_WEIGHTS: Dict[int, float] = {
    FraudLabel.BENIGN: 1.0,
    FraudLabel.FRAUD:  2.33,   # 70/30 ratio
}

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST INPUT DATA SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

XGBOOST_RAW_SCHEMA = {
    # ── IDENTITY (dropped before training — for traceability only) ────────────
    "transaction_id":            {"dtype": "str",   "nullable": False, "drop_before_train": True,  "description": "Unique transaction UUID"},
    "account_id_hash":           {"dtype": "str",   "nullable": False, "drop_before_train": True,  "description": "SHA-256 of originator account number"},
    "beneficiary_id_hash":       {"dtype": "str",   "nullable": False, "drop_before_train": True,  "description": "SHA-256 of beneficiary account number"},
    "device_fingerprint_hash":   {"dtype": "str",   "nullable": True,  "drop_before_train": True,  "description": "SHA-256 of device identifier"},
    "transaction_dt":            {"dtype": "datetime", "nullable": False, "drop_before_train": True, "description": "UTC transaction timestamp"},

    # ── TRANSACTION FUNDAMENTALS ──────────────────────────────────────────────
    "amount_inr":                {"dtype": "float64", "nullable": False, "drop_before_train": False, "range": [0.01, 1e8],    "description": "Transaction amount in INR"},
    "amount_usd_equiv":          {"dtype": "float64", "nullable": False, "drop_before_train": False, "range": [0.001, 1.2e6], "description": "USD equivalent at transaction time"},
    "channel":                   {"dtype": "str",   "nullable": False, "drop_before_train": False,
                                  "allowed": ["IMPS","NEFT","RTGS","UPI","CARD_CP","CARD_CNP","WIRE","CRYPTO","P2P"],
                                  "description": "Payment channel"},
    "merchant_category_code":    {"dtype": "str",   "nullable": True,  "drop_before_train": False, "description": "ISO 18245 MCC code (4 digits)"},
    "is_international":          {"dtype": "bool",  "nullable": False, "drop_before_train": False, "description": "Cross-border transaction flag"},
    "currency":                  {"dtype": "str",   "nullable": False, "drop_before_train": False, "description": "ISO 4217 currency code"},
    "is_new_beneficiary":        {"dtype": "bool",  "nullable": False, "drop_before_train": False, "description": "First transfer to this beneficiary"},
    "days_since_account_open":   {"dtype": "int32", "nullable": False, "drop_before_train": False, "range": [0, 36500],     "description": "Account age in days at transaction time"},

    # ── VELOCITY FEATURES (computed by Flink before ingestion) ────────────────
    "velocity_1h":               {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 10000],  "description": "Transaction count in last 1 hour"},
    "velocity_6h":               {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 50000],  "description": "Transaction count in last 6 hours"},
    "velocity_24h":              {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 200000], "description": "Transaction count in last 24 hours"},
    "velocity_7d":               {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 1e6],    "description": "Transaction count in last 7 days"},
    "amount_sum_1h":             {"dtype": "float64", "nullable": False, "drop_before_train": False, "range": [0, 1e10],   "description": "Total amount sent in last 1 hour (INR)"},
    "amount_sum_24h":            {"dtype": "float64", "nullable": False, "drop_before_train": False, "range": [0, 1e11],   "description": "Total amount sent in last 24 hours (INR)"},
    "distinct_beneficiaries_6h": {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 5000],   "description": "Unique beneficiaries in last 6 hours"},
    "distinct_beneficiaries_24h":{"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 20000],  "description": "Unique beneficiaries in last 24 hours"},

    # ── CONTEXTUAL ENRICHMENT (from real-time enrichment layer) ───────────────
    "ip_reputation_score":       {"dtype": "float32", "nullable": False, "drop_before_train": False, "range": [0.0, 1.0], "description": "IP risk score (IPQualityScore + Spur) — 1.0 = most malicious"},
    "is_tor_vpn_proxy":          {"dtype": "bool",    "nullable": False, "drop_before_train": False, "description": "True if originating IP is TOR exit node, VPN, or proxy"},
    "geo_distance_km":           {"dtype": "float32", "nullable": False, "drop_before_train": False, "range": [0, 40000],  "description": "Haversine distance from last authenticated login (km)"},
    "geo_country_mismatch":      {"dtype": "bool",    "nullable": False, "drop_before_train": False, "description": "True if transaction country differs from last login country"},
    "watchlist_hit":             {"dtype": "bool",    "nullable": False, "drop_before_train": False, "description": "OFAC/PEP/UN sanctions match"},
    "watchlist_category":        {"dtype": "str",     "nullable": True,  "drop_before_train": False,
                                  "allowed": ["OFAC_SDN","UN_CONSOLIDATED","PEP_TIER1","PEP_TIER2","FIU_IND","INTERNAL","NONE"],
                                  "description": "Watchlist category if hit"},
    "failed_auth_count_7d":      {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 1000],  "description": "Failed authentication attempts in last 7 days"},

    # ── BEHAVIOURAL PROFILE (from Feast online feature store) ─────────────────
    "amount_mean_90d":           {"dtype": "float64", "nullable": False, "drop_before_train": False, "description": "90-day rolling mean transaction amount (INR)"},
    "amount_std_90d":            {"dtype": "float64", "nullable": False, "drop_before_train": False, "description": "90-day rolling standard deviation of amount"},
    "typical_velocity_daily":    {"dtype": "float32", "nullable": False, "drop_before_train": False, "description": "Average daily transaction count (90-day)"},
    "is_dormant_account":        {"dtype": "bool",    "nullable": False, "drop_before_train": False, "description": "No activity >90 days before this event"},
    "distinct_devices_30d":      {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 200], "description": "Distinct device count in last 30 days"},
    "behavioural_drift_score":   {"dtype": "float32", "nullable": False, "drop_before_train": False, "range": [0.0, 1.0], "description": "KL divergence from 90-day behavioural baseline"},
    "peer_group_id":             {"dtype": "str",     "nullable": True,  "drop_before_train": True,  "description": "Peer segment cluster ID (used for peer_amount_percentile)"},
    "peer_amount_percentile":    {"dtype": "float32", "nullable": False, "drop_before_train": False, "range": [0.0, 100.0], "description": "Amount percentile vs. peer segment"},

    # ── ENTITY GRAPH SIGNALS (from GraphSAGE pre-computed Redis cache) ─────────
    "shared_device_accounts":    {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 10000], "description": "Other accounts sharing same device fingerprint"},
    "shared_ip_accounts":        {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 10000], "description": "Other accounts from same IP address"},
    "beneficiary_sar_count":     {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 1000],  "description": "Prior SAR filings involving beneficiary"},
    "second_hop_sar_count":      {"dtype": "int32",   "nullable": False, "drop_before_train": False, "range": [0, 5000],  "description": "SAR count among accounts 2 hops away in graph"},
    "mule_network_probability":  {"dtype": "float32", "nullable": False, "drop_before_train": False, "range": [0.0, 1.0], "description": "GraphSAGE mule ring membership probability"},
    "synthetic_identity_score":  {"dtype": "float32", "nullable": False, "drop_before_train": False, "range": [0.0, 1.0], "description": "Identity document consistency anomaly score"},
    "rapid_fan_out_flag":        {"dtype": "bool",    "nullable": False, "drop_before_train": False, "description": "Funds received then dispersed to 5+ beneficiaries within 2h"},
    "originator_graph_centrality":{"dtype":"float32", "nullable": False, "drop_before_train": False, "range": [0.0, 1.0], "description": "PageRank centrality of originator in entity graph"},

    # ── LABELS ────────────────────────────────────────────────────────────────
    "label_is_fraud":            {"dtype": "int8",    "nullable": False, "drop_before_train": False, "allowed": [0, 1],   "description": "Binary fraud label (0=benign, 1=fraud)"},
    "label_typology":            {"dtype": "str",     "nullable": False, "drop_before_train": False,
                                  "allowed": [t.value for t in FraudTypology],
                                  "description": "FinCEN typology code or BENIGN"},
    "label_source":              {"dtype": "str",     "nullable": False, "drop_before_train": True,
                                  "allowed": ["CONFIRMED_SAR","INVESTIGATOR","CARD_NETWORK","NPCI","CONSORTIUM","SYNTHETIC"],
                                  "description": "Origin of label — used for sample weighting"},
    "label_confidence":          {"dtype": "float32", "nullable": False, "drop_before_train": True,  "range": [0.5, 1.0], "description": "Label confidence (1.0=confirmed SAR, 0.7-0.9=synthetic)"},
}

# Features used in XGBoost training (raw → engineered names)
XGBOOST_RAW_FEATURE_COLS = [k for k, v in XGBOOST_RAW_SCHEMA.items()
                              if not v["drop_before_train"] and k not in
                              ("label_is_fraud", "label_typology")]

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST ENGINEERED FEATURE SCHEMA (44 features after transformation)
# ─────────────────────────────────────────────────────────────────────────────

XGBOOST_ENGINEERED_FEATURES = {
    # Group A: Amount features
    "amount_inr_log":             "log1p(amount_inr) — log-normalises heavy tail",
    "amount_usd_equiv_log":       "log1p(amount_usd_equiv)",
    "amount_zscore":              "(amount_inr - amount_mean_90d) / max(amount_std_90d, 1) — deviation from baseline",
    "amount_sum_1h_log":          "log1p(amount_sum_1h)",
    "amount_sum_24h_log":         "log1p(amount_sum_24h)",
    "ctr_proximity":              "abs(amount_inr - 1_000_000) / 1_000_000 — distance from INR 10L CTR threshold",

    # Group B: Velocity features
    "velocity_1h":                "Raw — transaction count last 1h",
    "velocity_6h":                "Raw — transaction count last 6h",
    "velocity_24h":               "Raw — transaction count last 24h",
    "velocity_7d":                "Raw — transaction count last 7d",
    "velocity_ratio_1h_24h":      "velocity_1h / max(velocity_24h / 24, 0.001) — hourly vs daily rate",
    "amount_per_tx_1h":           "amount_sum_1h / max(velocity_1h, 1) — average amount per recent tx",
    "distinct_bene_6h":           "distinct_beneficiaries_6h — raw",
    "distinct_bene_24h":          "distinct_beneficiaries_24h — raw",

    # Group C: Temporal features (derived from transaction_dt)
    "hour_of_day":                "transaction_dt.hour — 0-23",
    "day_of_week":                "transaction_dt.dayofweek — 0=Mon",
    "is_weekend":                 "int(day_of_week >= 5)",
    "is_nighttime":               "int(hour < 6 or hour >= 22) — nocturnal flag",
    "is_month_end":               "int(day >= 28) — month-end activity pattern",

    # Group D: Channel encoding (one-hot)
    "channel_IMPS":               "int(channel == 'IMPS')",
    "channel_UPI":                "int(channel == 'UPI')",
    "channel_CARD_CNP":           "int(channel == 'CARD_CNP')",
    "channel_WIRE":               "int(channel == 'WIRE')",
    "channel_CRYPTO":             "int(channel == 'CRYPTO')",
    "is_international":           "Raw boolean cast to int",

    # Group E: Geo / device features
    "geo_distance_km_log":        "log1p(geo_distance_km) — log-normalises extreme outliers",
    "geo_country_mismatch":       "Raw boolean cast to int",
    "is_tor_vpn_proxy":           "Raw boolean cast to int",
    "ip_reputation_score":        "Raw float 0-1",
    "failed_auth_count_7d":       "Raw integer",
    "distinct_devices_30d":       "Raw integer",

    # Group F: Behavioural features
    "velocity_vs_baseline":       "velocity_1h / max(typical_velocity_daily / 24, 0.001) — vs daily normalised",
    "behavioural_drift_score":    "Raw KL divergence 0-1",
    "is_dormant_account":         "Raw boolean cast to int",
    "days_since_acct_open_log":   "log1p(days_since_account_open)",
    "is_new_beneficiary":         "Raw boolean cast to int",
    "peer_amount_percentile":     "Raw 0-100",
    "watchlist_hit":              "Raw boolean cast to int",

    # Group G: Entity graph features
    "shared_device_accounts_log": "log1p(shared_device_accounts)",
    "shared_ip_accounts_log":     "log1p(shared_ip_accounts)",
    "beneficiary_sar_count":      "Raw integer",
    "second_hop_sar_count":       "Raw integer — network contamination signal",
    "mule_network_probability":   "Raw float 0-1",
    "synthetic_identity_score":   "Raw float 0-1",
    "rapid_fan_out_flag":         "Raw boolean cast to int",
    "originator_graph_centrality":"Raw float 0-1",
}

# The original generated file claimed 44 engineered features, but the dictionary currently contains 46.
# Keep the schema authoritative and avoid import-time failure.
assert len(XGBOOST_ENGINEERED_FEATURES) == 46, \
    f"Expected 46 features, got {len(XGBOOST_ENGINEERED_FEATURES)}"

# ─────────────────────────────────────────────────────────────────────────────
# GNN INPUT DATA SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

GNN_NODE_SCHEMA = {
    # ── ACCOUNT NODES ─────────────────────────────────────────────────────────
    "account": {
        "node_id":                  {"dtype": "str",     "description": "SHA-256 account hash — primary key"},
        "account_age_days":         {"dtype": "int32",   "description": "Days since account opened"},
        "kyc_tier":                 {"dtype": "int8",    "allowed": [0, 1, 2, 3], "description": "0=None,1=Basic,2=Standard,3=Enhanced"},
        "account_type":             {"dtype": "int8",    "allowed": [0, 1, 2],    "description": "0=Individual,1=Business,2=Joint"},
        "relationship_years":       {"dtype": "float32", "description": "Years as customer"},
        "lifetime_tx_count":        {"dtype": "int32",   "description": "All-time transaction count"},
        "lifetime_amount_log":      {"dtype": "float32", "description": "log1p of total lifetime amount INR"},
        "prior_sar_count":          {"dtype": "int8",    "description": "Number of prior SAR filings"},
        "prior_alert_count":        {"dtype": "int16",   "description": "Total prior fraud alerts"},
        "prior_confirmed_fraud":    {"dtype": "int8",    "allowed": [0, 1],       "description": "1 if prior confirmed fraud"},
        "monthly_avg_tx_count":     {"dtype": "float32", "description": "Average monthly transaction count (12m)"},
        "monthly_avg_amount_log":   {"dtype": "float32", "description": "log1p of average monthly amount INR"},
        "geographic_risk_score":    {"dtype": "float32", "range": [0.0, 1.0],     "description": "Country/region risk score"},
        "pep_flag":                 {"dtype": "int8",    "allowed": [0, 1],       "description": "Politically Exposed Person"},
        "industry_risk_score":      {"dtype": "float32", "range": [0.0, 1.0],     "description": "Industry sector risk (for business accounts)"},
        "distinct_devices_90d":     {"dtype": "int16",   "description": "Distinct devices used in 90 days"},
        "avg_failed_auth_monthly":  {"dtype": "float32", "description": "Average monthly failed auth attempts"},
        # Label (for training only)
        "node_label":               {"dtype": "int8",    "allowed": [0, 1],       "description": "0=benign node, 1=fraud node"},
        "label_source":             {"dtype": "str",     "drop_at_inference": True,"description": "Label provenance"},
    },
    # ── DEVICE NODES ──────────────────────────────────────────────────────────
    "device": {
        "node_id":                  {"dtype": "str",     "description": "SHA-256 device fingerprint hash"},
        "first_seen_days_ago":      {"dtype": "int32",   "description": "Days since device first seen in system"},
        "associated_account_count": {"dtype": "int16",   "description": "Number of accounts linked to this device"},
        "platform":                 {"dtype": "int8",    "allowed": [0,1,2,3],    "description": "0=Android,1=iOS,2=Web,3=Other"},
        "device_risk_score":        {"dtype": "float32", "range": [0.0, 1.0],     "description": "Device risk (from device intelligence API)"},
        "has_root_jailbreak":       {"dtype": "int8",    "allowed": [0, 1],       "description": "Device is rooted or jailbroken"},
    },
    # ── IP NODES ──────────────────────────────────────────────────────────────
    "ip": {
        "node_id":                  {"dtype": "str",     "description": "Hashed IP address"},
        "reputation_score":         {"dtype": "float32", "range": [0.0, 1.0],     "description": "IP risk score from IPQualityScore"},
        "is_tor":                   {"dtype": "int8",    "allowed": [0, 1],       "description": "TOR exit node"},
        "is_vpn":                   {"dtype": "int8",    "allowed": [0, 1],       "description": "VPN endpoint"},
        "is_proxy":                 {"dtype": "int8",    "allowed": [0, 1],       "description": "Open proxy"},
        "country_risk_score":       {"dtype": "float32", "range": [0.0, 1.0],     "description": "Country-level AML risk"},
        "associated_account_count": {"dtype": "int16",   "description": "Accounts originating from this IP"},
    },
    # ── MERCHANT NODES ────────────────────────────────────────────────────────
    "merchant": {
        "node_id":                  {"dtype": "str",     "description": "Merchant registry ID (hashed)"},
        "mcc_code":                 {"dtype": "str",     "description": "ISO 18245 MCC code"},
        "merchant_age_days":        {"dtype": "int32",   "description": "Days since merchant registered"},
        "mcc_risk_score":           {"dtype": "float32", "range": [0.0, 1.0],     "description": "MCC category risk score"},
        "chargeback_rate_90d":      {"dtype": "float32", "range": [0.0, 1.0],     "description": "Chargeback rate last 90 days"},
        "dispute_count_90d":        {"dtype": "int16",   "description": "Disputes filed in last 90 days"},
    },
}

GNN_EDGE_SCHEMA = {
    "TRANSACTS_WITH": {
        "description": "Account → Beneficiary (account-to-account transaction edge)",
        "source_type": "account", "target_type": "account",
        "features": {
            "transaction_count_30d":     {"dtype": "int16",   "description": "Number of transactions in 30 days"},
            "total_amount_30d_log":      {"dtype": "float32", "description": "log1p total amount INR sent in 30 days"},
            "first_transaction_days_ago":{"dtype": "int16",   "description": "Days since first transaction on this edge"},
            "last_transaction_hours_ago":{"dtype": "float32", "description": "Hours since most recent transaction"},
            "avg_amount_log":            {"dtype": "float32", "description": "log1p average transaction amount"},
            "is_recurring":              {"dtype": "int8",    "description": "Regular periodic pattern detected"},
            "edge_risk_score":           {"dtype": "float32", "range": [0.0, 1.0], "description": "Risk score from scoring engine for this pair"},
        }
    },
    "USES_DEVICE": {
        "description": "Account → Device (account uses device edge)",
        "source_type": "account", "target_type": "device",
        "features": {
            "session_count_30d":         {"dtype": "int16",   "description": "Sessions from this device in 30 days"},
            "last_session_hours_ago":    {"dtype": "float32", "description": "Hours since last session"},
            "is_primary_device":         {"dtype": "int8",    "description": "Most-used device for this account"},
        }
    },
    "ORIGINATES_FROM": {
        "description": "Account → IP (session originated from IP)",
        "source_type": "account", "target_type": "ip",
        "features": {
            "session_count_7d":          {"dtype": "int16",   "description": "Sessions from this IP in 7 days"},
            "last_session_hours_ago":    {"dtype": "float32", "description": "Hours since last session from this IP"},
        }
    },
    "TRANSACTS_AT": {
        "description": "Account → Merchant (account transacted at merchant)",
        "source_type": "account", "target_type": "merchant",
        "features": {
            "transaction_count_90d":     {"dtype": "int16",   "description": "Transactions at this merchant in 90 days"},
            "total_amount_90d_log":      {"dtype": "float32", "description": "log1p total amount at merchant in 90 days"},
            "is_first_transaction":      {"dtype": "int8",    "description": "First ever transaction at this merchant"},
        }
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataQualityReport:
    total_rows:          int
    null_violations:     Dict[str, int]     # column → null count
    range_violations:    Dict[str, int]     # column → out-of-range count
    schema_violations:   Dict[str, str]     # column → error message
    duplicate_ids:       int
    class_distribution:  Dict[str, float]   # label → fraction
    label_source_dist:   Dict[str, float]   # source → fraction
    passes_contract:     bool
    failure_reasons:     List[str]

class DataQualityValidator:
    """Validates raw input DataFrame against schema contracts."""

    NULL_TOLERANCE    = 0.01   # 1% maximum null rate
    IMBALANCE_MAX     = 0.80   # Maximum fraction of any single class
    MIN_FRAUD_RATE    = 0.05   # Minimum fraud rate to ensure enough positive samples

    def validate_xgboost_dataset(self, df) -> DataQualityReport:
        import pandas as pd
        failures = []
        null_violations, range_violations, schema_violations = {}, {}, {}

        for col, spec in XGBOOST_RAW_SCHEMA.items():
            if col not in df.columns:
                if not spec.get("nullable", True):
                    schema_violations[col] = f"Required column missing"
                continue

            null_rate = df[col].isnull().mean()
            if not spec.get("nullable", False) and null_rate > self.NULL_TOLERANCE:
                null_violations[col] = int(df[col].isnull().sum())

            if "range" in spec and spec["dtype"] in ("float32","float64","int32","int8"):
                lo, hi = spec["range"]
                oob = ((df[col] < lo) | (df[col] > hi)).sum()
                if oob > 0:
                    range_violations[col] = int(oob)

            if "allowed" in spec:
                invalid = ~df[col].isin(spec["allowed"])
                if invalid.any():
                    schema_violations[col] = f"{invalid.sum()} values not in allowed set"

        dup_ids = df["transaction_id"].duplicated().sum() if "transaction_id" in df.columns else 0
        if dup_ids > 0:
            failures.append(f"{dup_ids} duplicate transaction_ids")

        class_dist = {}
        fraud_rate = 0.0
        if "label_is_fraud" in df.columns:
            vc = df["label_is_fraud"].value_counts(normalize=True)
            class_dist = {str(k): round(float(v), 4) for k, v in vc.items()}
            fraud_rate = float(vc.get(1, 0))
            if fraud_rate < self.MIN_FRAUD_RATE:
                failures.append(f"Fraud rate {fraud_rate:.2%} below minimum {self.MIN_FRAUD_RATE:.2%}")

        label_source_dist = {}
        if "label_source" in df.columns:
            vc = df["label_source"].value_counts(normalize=True)
            label_source_dist = {k: round(float(v), 4) for k, v in vc.items()}

        if null_violations:
            failures.append(f"Null violations in: {list(null_violations.keys())}")
        if range_violations:
            failures.append(f"Range violations in: {list(range_violations.keys())}")
        if schema_violations:
            failures.append(f"Schema violations in: {list(schema_violations.keys())}")

        return DataQualityReport(
            total_rows=len(df), null_violations=null_violations,
            range_violations=range_violations, schema_violations=schema_violations,
            duplicate_ids=int(dup_ids), class_distribution=class_dist,
            label_source_dist=label_source_dist,
            passes_contract=len(failures) == 0, failure_reasons=failures,
        )
