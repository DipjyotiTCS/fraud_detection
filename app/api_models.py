from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class DatasetGenerateRequest(BaseModel):
    n_rows: int = Field(default=10000, ge=200, le=5_000_000)
    fraud_rate: float = Field(default=0.12, ge=0.01, le=0.49)
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42
    output_dir: str = "data/xgboost"
    dataset_name: str = "xgboost_training_data"
    generate_engineered_csv: bool = True
    generate_split_csvs: bool = True
    temporal_split: bool = True
    graph_feature_mode: Literal["synthetic", "none", "gnn_scores"] = "synthetic"
    graph_score_path: Optional[str] = None


class TrainXGBoostRequest(BaseModel):
    raw_data_path: Optional[str] = Field(
        default=None,
        description="Defaults to data/xgboost/xgboost_training_data.csv when omitted.",
    )
    artifact_dir: str = "artifacts/xgboost"
    model_version: str = "v1"
    run_hpo: bool = False
    n_hpo_trials: int = Field(default=10, ge=1, le=100)
    temporal_split: bool = True
    async_mode: bool = False


class ScoreRequest(BaseModel):
    records: List[Dict[str, Any]]
    artifact_dir: str = "artifacts/xgboost"
    model_version: str = "v1"


class GraphDatasetGenerateRequest(BaseModel):
    source_transaction_csv: str = "data/xgboost/xgboost_training_data.csv"
    output_dir: str = "data/gnn"
    dataset_name: str = "fraud_graph"
    seed: int = 42
    max_shared_entity_edges: int = Field(default=250000, ge=0, le=2_000_000)


class TrainGNNRequest(BaseModel):
    graph_data_dir: str = "data/gnn"
    dataset_name: str = "fraud_graph"
    artifact_dir: str = "artifacts/gnn"
    model_version: str = "v1"
    device: str = "auto"
    hidden_dim: int = Field(default=64, ge=8, le=512)
    epochs: int = Field(default=35, ge=1, le=500)
    learning_rate: float = Field(default=0.003, gt=0, le=0.1)
    weight_decay: float = Field(default=0.0005, ge=0, le=0.1)
    dropout: float = Field(default=0.25, ge=0, le=0.8)
    patience: int = Field(default=6, ge=1, le=100)
    seed: int = 42
    include_shared_entity_edges: bool = True
    async_mode: bool = False


class InferGNNRequest(BaseModel):
    graph_data_dir: str = "data/gnn"
    dataset_name: str = "fraud_graph"
    artifact_dir: str = "artifacts/gnn"
    model_version: str = "v1"
    output_path: str = "data/gnn/gnn_account_scores_v1.csv"
    device: str = "auto"


class EnrichXGBoostWithGNNRequest(BaseModel):
    source_transaction_csv: str = "data/xgboost/xgboost_training_data.csv"
    gnn_score_path: str = "data/gnn/gnn_account_scores_v1.csv"
    output_dir: str = "data/xgboost"
    output_dataset_name: str = "xgboost_training_data_gnn_enriched"


class HybridPipelineRequest(BaseModel):
    n_rows: int = Field(default=10000, ge=200, le=5_000_000)
    fraud_rate: float = Field(default=0.12, ge=0.01, le=0.49)
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42
    base_dataset_name: str = "xgboost_base_data"
    final_dataset_name: str = "xgboost_training_data_gnn_enriched"
    graph_dataset_name: str = "fraud_graph"
    gnn_version: str = "v1"
    xgboost_version: str = "v1"
    train_gnn: bool = True
    train_xgboost: bool = True
    xgboost_run_hpo: bool = False
    async_mode: bool = True
