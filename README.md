# FraudSentinel Hybrid Fraud ML FastAPI Project

This project now supports a local **hybrid fraud-detection ML flow**:

```text
XGBoost raw transaction dataset
    ↓
GNN graph dataset generation
    ↓
Account-level GraphSAGE training
    ↓
GNN account risk score export
    ↓
GNN-enriched XGBoost dataset
    ↓
Final XGBoost transaction-level fraud model
```

The GNN is implemented as a local PyTorch GraphSAGE-style account model. It uses graph CSVs derived from the transaction data, so you can run it without Neo4j. Later, the graph CSV generation layer can be replaced by Neo4j extractors.

---

## Project structure

```text
fraud_xgboost_fastapi_project/
  app/
    main.py                         # FastAPI endpoints
    api_models.py                   # Request models
  fraud_ml/
    shared/
      schemas.py                    # Shared schema definitions
    xgboost/
      dataset_builder.py            # XGBoost raw CSV generator
      feature_engineering.py        # XGBoost feature engineering
      trainer.py                    # XGBoost training/evaluation service
      inference.py                  # XGBoost scoring helper
    gnn/
      graph_dataset_builder.py      # Builds account/device/IP/merchant graph CSVs
      graph_trainer.py              # PyTorch GraphSAGE-style account GNN trainer
      graph_inference.py            # Batch GNN score export
      graph_feature_injection.py    # Injects GNN scores into XGBoost raw dataset
  data/xgboost/                     # Generated XGBoost CSVs
  data/gnn/                         # Generated graph node/edge CSVs and score CSVs
  artifacts/xgboost/                # Trained XGBoost artifacts
  artifacts/gnn/                    # Trained GNN artifacts
  reference_original_files/          # Original uploaded files
  requirements.txt
  Dockerfile
  run_local.sh
```

---

## Install and run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export FRAUD_ML_HOME=$(pwd)  # Windows PowerShell: $env:FRAUD_ML_HOME=(Get-Location)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

---

# XGBoost-only flow

## 1. Generate XGBoost dataset

```bash
curl -X POST "http://localhost:8000/api/datasets/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 100000,
    "fraud_rate": 0.08,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "seed": 42,
    "output_dir": "data/xgboost",
    "dataset_name": "xgboost_training_data",
    "generate_engineered_csv": true,
    "generate_split_csvs": true,
    "temporal_split": true,
    "graph_feature_mode": "synthetic"
  }'
```

## 2. Train XGBoost

```bash
curl -X POST "http://localhost:8000/api/xgboost/train" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_data_path": "data/xgboost/xgboost_training_data.csv",
    "artifact_dir": "artifacts/xgboost",
    "model_version": "v1",
    "run_hpo": false,
    "n_hpo_trials": 10,
    "temporal_split": true,
    "async_mode": false
  }'
```

---

# GNN + XGBoost hybrid flow

## 1. Generate base XGBoost transaction data

```bash
curl -X POST "http://localhost:8000/api/datasets/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 100000,
    "fraud_rate": 0.08,
    "output_dir": "data/xgboost",
    "dataset_name": "xgboost_base_data",
    "generate_engineered_csv": false,
    "generate_split_csvs": false,
    "graph_feature_mode": "synthetic"
  }'
```

This creates:

```text
data/xgboost/xgboost_base_data.csv
```

## 2. Generate GNN graph dataset from the transaction CSV

```bash
curl -X POST "http://localhost:8000/api/gnn/datasets/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "source_transaction_csv": "data/xgboost/xgboost_base_data.csv",
    "output_dir": "data/gnn",
    "dataset_name": "fraud_graph",
    "seed": 42,
    "max_shared_entity_edges": 250000
  }'
```

This creates files such as:

```text
data/gnn/fraud_graph_account_nodes.csv
data/gnn/fraud_graph_device_nodes.csv
data/gnn/fraud_graph_ip_nodes.csv
data/gnn/fraud_graph_merchant_nodes.csv
data/gnn/fraud_graph_edges_account_account.csv
data/gnn/fraud_graph_edges_account_device.csv
data/gnn/fraud_graph_edges_account_ip.csv
data/gnn/fraud_graph_edges_account_merchant.csv
data/gnn/fraud_graph_edges_account_shared_entity.csv
```

## 3. Train GNN

```bash
curl -X POST "http://localhost:8000/api/gnn/train" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_data_dir": "data/gnn",
    "dataset_name": "fraud_graph",
    "artifact_dir": "artifacts/gnn",
    "model_version": "v1",
    "device": "auto",
    "hidden_dim": 64,
    "epochs": 35,
    "learning_rate": 0.003,
    "dropout": 0.25,
    "patience": 6,
    "include_shared_entity_edges": true,
    "async_mode": true
  }'
```

Poll the job:

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

GNN artifacts:

```text
artifacts/gnn/gnn_model_v1.pt
artifacts/gnn/gnn_scaler_v1.pkl
artifacts/gnn/account_id_map_v1.json
artifacts/gnn/gnn_config_v1.json
artifacts/gnn/gnn_metrics_v1.json
artifacts/gnn/training_metadata_v1.json
```

## 4. Generate GNN account risk scores

```bash
curl -X POST "http://localhost:8000/api/gnn/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_data_dir": "data/gnn",
    "dataset_name": "fraud_graph",
    "artifact_dir": "artifacts/gnn",
    "model_version": "v1",
    "output_path": "data/gnn/gnn_account_scores_v1.csv",
    "device": "auto"
  }'
```

This creates:

```text
data/gnn/gnn_account_scores_v1.csv
```

with:

```text
account_id_hash, gnn_fraud_probability, gnn_risk_score
```

## 5. Create GNN-enriched XGBoost training dataset

```bash
curl -X POST "http://localhost:8000/api/datasets/enrich-with-gnn" \
  -H "Content-Type: application/json" \
  -d '{
    "source_transaction_csv": "data/xgboost/xgboost_base_data.csv",
    "gnn_score_path": "data/gnn/gnn_account_scores_v1.csv",
    "output_dir": "data/xgboost",
    "output_dataset_name": "xgboost_training_data_gnn_enriched"
  }'
```

This creates:

```text
data/xgboost/xgboost_training_data_gnn_enriched.csv
```

The injector maps GNN scores into graph fields used by XGBoost, especially:

```text
mule_network_probability
originator_graph_centrality
synthetic_identity_score
second_hop_sar_count
rapid_fan_out_flag
```

## 6. Train final XGBoost on the GNN-enriched dataset

```bash
curl -X POST "http://localhost:8000/api/xgboost/train" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_data_path": "data/xgboost/xgboost_training_data_gnn_enriched.csv",
    "artifact_dir": "artifacts/xgboost",
    "model_version": "v1",
    "run_hpo": false,
    "temporal_split": true,
    "async_mode": true
  }'
```

---

# One-call hybrid pipeline

You can also run the full flow with one API:

```bash
curl -X POST "http://localhost:8000/api/pipeline/train-hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 100000,
    "fraud_rate": 0.08,
    "base_dataset_name": "xgboost_base_data",
    "final_dataset_name": "xgboost_training_data_gnn_enriched",
    "graph_dataset_name": "fraud_graph",
    "gnn_version": "v1",
    "xgboost_version": "v1",
    "train_gnn": true,
    "train_xgboost": true,
    "xgboost_run_hpo": false,
    "async_mode": true
  }'
```

Poll:

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

---

# Inference

After XGBoost is trained, score raw transaction records:

```bash
curl -X POST "http://localhost:8000/api/xgboost/score" \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_dir": "artifacts/xgboost",
    "model_version": "v1",
    "records": [
      {
        "transaction_id": "sample-txn-1",
        "account_id_hash": "acct-hash",
        "beneficiary_id_hash": "bene-hash",
        "device_fingerprint_hash": "device-hash",
        "transaction_dt": "2025-06-01T10:00:00Z",
        "amount_inr": 950000,
        "amount_usd_equiv": 11465,
        "channel": "UPI",
        "merchant_category_code": "6012",
        "is_international": false,
        "currency": "INR",
        "is_new_beneficiary": true,
        "days_since_account_open": 45,
        "velocity_1h": 8,
        "velocity_6h": 20,
        "velocity_24h": 45,
        "velocity_7d": 80,
        "amount_sum_1h": 2500000,
        "amount_sum_24h": 6500000,
        "distinct_beneficiaries_6h": 5,
        "distinct_beneficiaries_24h": 9,
        "ip_reputation_score": 0.82,
        "is_tor_vpn_proxy": true,
        "geo_distance_km": 3000,
        "geo_country_mismatch": true,
        "watchlist_hit": false,
        "watchlist_category": "NONE",
        "failed_auth_count_7d": 6,
        "amount_mean_90d": 25000,
        "amount_std_90d": 8000,
        "typical_velocity_daily": 4,
        "is_dormant_account": false,
        "distinct_devices_30d": 6,
        "behavioural_drift_score": 0.75,
        "peer_group_id": "PG-001",
        "peer_amount_percentile": 98,
        "shared_device_accounts": 10,
        "shared_ip_accounts": 20,
        "beneficiary_sar_count": 1,
        "second_hop_sar_count": 3,
        "mule_network_probability": 0.78,
        "synthetic_identity_score": 0.42,
        "rapid_fan_out_flag": true,
        "originator_graph_centrality": 0.55
      }
    ]
  }'
```

---

## Notes

- XGBoost can still run without GNN by using `graph_feature_mode: synthetic` or `graph_feature_mode: none`.
- GNN training/inference requires PyTorch, which is now included in `requirements.txt`.
- The local GNN is account-level GraphSAGE. It uses account transfer edges and derived account-account edges from shared devices, IPs, and merchants.
- For production, replace `graph_dataset_builder.py` with a Neo4j-backed extractor and keep the downstream file contracts unchanged.
