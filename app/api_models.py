from __future__ import annotations

import os

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def _env_csv_list(name: str, default_csv: str) -> List[str]:
    value = os.getenv(name)
    raw = value if value not in (None, "") else default_csv
    return [item.strip() for item in raw.split(",") if item.strip()]


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



class LLMDatasetInspectRequest(BaseModel):
    dataset_dir: str = Field(default_factory=lambda: _env_str("LLM_SFT_DATASET_DIR", "data/llm/sft"))
    dataset_type: Literal["sft", "dpo"] = "sft"
    validation_ratio: float = Field(default_factory=lambda: _env_float("LLM_VALIDATION_RATIO", 0.1), ge=0, le=0.49)
    test_ratio: float = Field(default_factory=lambda: _env_float("LLM_TEST_RATIO", 0.1), ge=0, le=0.49)
    seed: int = Field(default_factory=lambda: _env_int("LLM_SEED", 42))


class DownloadLLMBaseModelRequest(BaseModel):
    model_id: str = Field(
        default_factory=lambda: _env_str("LLM_DEFAULT_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
        description="Use mistralai/Mistral-7B-v0.3 if you want the raw pretrained base model instead of the instruct model.",
    )
    output_dir: str = Field(default_factory=lambda: _env_str("LLM_BASE_MODEL_PATH", "artifacts/llm/base/mistral-7b-instruct-v0.3"))
    revision: str = Field(default_factory=lambda: _env_str("LLM_DEFAULT_MODEL_REVISION", "main"))
    allow_patterns: Optional[List[str]] = Field(
        default=None,
        description="Optional Hugging Face allow_patterns. Leave null to download the full model snapshot.",
    )
    ignore_patterns: Optional[List[str]] = None
    async_mode: bool = Field(default_factory=lambda: _env_bool("LLM_ASYNC_MODE", True))


class FineTuneLLMRequest(BaseModel):
    base_model_path: str = Field(default_factory=lambda: _env_str("LLM_BASE_MODEL_PATH", "artifacts/llm/base/mistral-7b-instruct-v0.3"))
    output_dir: str = Field(default_factory=lambda: _env_str("LLM_FINETUNED_OUTPUT_DIR", "artifacts/llm/finetuned/risk-report-mistral-7b"))
    sft_dataset_dir: str = Field(default_factory=lambda: _env_str("LLM_SFT_DATASET_DIR", "data/llm/sft"))
    dpo_dataset_dir: str = Field(default_factory=lambda: _env_str("LLM_DPO_DATASET_DIR", "data/llm/dpo"))
    run_sft: bool = Field(default_factory=lambda: _env_bool("LLM_RUN_SFT", True))
    run_dpo: bool = Field(default_factory=lambda: _env_bool("LLM_RUN_DPO", True))
    use_4bit: bool = Field(default_factory=lambda: _env_bool("LLM_USE_4BIT", True))
    max_seq_length: int = Field(default_factory=lambda: _env_int("LLM_MAX_SEQ_LENGTH", 2048), ge=256, le=8192)
    sft_epochs: float = Field(default_factory=lambda: _env_float("LLM_SFT_EPOCHS", 2.0), gt=0, le=20)
    dpo_epochs: float = Field(default_factory=lambda: _env_float("LLM_DPO_EPOCHS", 1.0), gt=0, le=20)
    per_device_train_batch_size: int = Field(default_factory=lambda: _env_int("LLM_TRAIN_BATCH_SIZE", 1), ge=1, le=16)
    gradient_accumulation_steps: int = Field(default_factory=lambda: _env_int("LLM_GRADIENT_ACCUMULATION_STEPS", 8), ge=1, le=128)
    learning_rate: float = Field(default_factory=lambda: _env_float("LLM_LEARNING_RATE", 2e-4), gt=0, le=1e-2)
    dpo_learning_rate: float = Field(default_factory=lambda: _env_float("LLM_DPO_LEARNING_RATE", 5e-6), gt=0, le=1e-3)
    logging_steps: int = Field(default_factory=lambda: _env_int("LLM_LOGGING_STEPS", 5), ge=1, le=1000)
    save_steps: int = Field(default_factory=lambda: _env_int("LLM_SAVE_STEPS", 100), ge=1, le=10000)
    save_total_limit: int = Field(default_factory=lambda: _env_int("LLM_SAVE_TOTAL_LIMIT", 2), ge=1, le=20)
    seed: int = Field(default_factory=lambda: _env_int("LLM_SEED", 42))
    validation_ratio: float = Field(default_factory=lambda: _env_float("LLM_VALIDATION_RATIO", 0.1), ge=0, le=0.49)
    test_ratio: float = Field(default_factory=lambda: _env_float("LLM_TEST_RATIO", 0.1), ge=0, le=0.49)
    lora_r: int = Field(default_factory=lambda: _env_int("LLM_LORA_R", 16), ge=1, le=256)
    lora_alpha: int = Field(default_factory=lambda: _env_int("LLM_LORA_ALPHA", 32), ge=1, le=512)
    lora_dropout: float = Field(default_factory=lambda: _env_float("LLM_LORA_DROPOUT", 0.05), ge=0, le=0.8)
    lora_target_modules: Optional[List[str]] = Field(
        default_factory=lambda: _env_csv_list(
            "LLM_LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
    )
    resume_from_checkpoint: Optional[str] = None
    async_mode: bool = Field(default_factory=lambda: _env_bool("LLM_ASYNC_MODE", True))


class RiskReportInferenceRequest(BaseModel):
    transaction: Dict[str, Any]
    classification_result: Dict[str, Any]
    customer_context: Optional[Dict[str, Any]] = None
    report_instruction: Optional[str] = None
    base_model_path: str = Field(default_factory=lambda: _env_str("LLM_BASE_MODEL_PATH", "artifacts/llm/base/mistral-7b-instruct-v0.3"))
    adapter_path: str = Field(default_factory=lambda: _env_str("LLM_FINAL_ADAPTER_DIR", "artifacts/llm/finetuned/risk-report-mistral-7b/dpo/final_adapter"))
    use_4bit: bool = Field(default_factory=lambda: _env_bool("LLM_USE_4BIT", True))
    max_new_tokens: int = Field(default_factory=lambda: _env_int("LLM_MAX_NEW_TOKENS", 700), ge=64, le=4096)
    temperature: float = Field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2), ge=0, le=2)
    top_p: float = Field(default_factory=lambda: _env_float("LLM_TOP_P", 0.9), ge=0.01, le=1)
