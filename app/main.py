from __future__ import annotations

import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api_models import (
    DatasetGenerateRequest,
    DownloadLLMBaseModelRequest,
    EnrichXGBoostWithGNNRequest,
    FineTuneLLMRequest,
    GraphDatasetGenerateRequest,
    HybridPipelineRequest,
    InferGNNRequest,
    LLMDatasetInspectRequest,
    RiskReportInferenceRequest,
    ScoreRequest,
    TrainGNNRequest,
    TrainXGBoostRequest,
)
from fraud_ml.gnn.graph_dataset_builder import GraphDatasetBuildConfig, GraphDatasetBuilder
from fraud_ml.gnn.graph_feature_injection import GNNScoreInjectionConfig, GNNScoreInjector
from fraud_ml.gnn.graph_inference import GNNInferConfig, GNNInferenceService
from fraud_ml.gnn.graph_trainer import GNNTrainConfig, GNNTrainingService
from fraud_ml.llm.finetune import LLMFineTuneConfig, LLMFineTuningService
from fraud_ml.llm.inference import RiskReportInferenceConfig, RiskReportInferenceService
from fraud_ml.llm.model_download import LLMDownloadConfig, LLMModelDownloader
from fraud_ml.llm.dataset_utils import inspect_dataset_folder
from fraud_ml.xgboost.dataset_builder import DatasetBuildConfig, XGBoostDatasetBuilder
from fraud_ml.xgboost.inference import XGBoostInferenceService
from fraud_ml.xgboost.trainer import XGBoostTrainingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fraud_api")

APP_ROOT = Path(os.getenv("FRAUD_ML_HOME", ".")).resolve()
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="FraudSentinel Hybrid Fraud ML API",
    version="2.0.0",
    description="Generate fraud datasets, train GNN/GraphSAGE and XGBoost models, fine-tune a Mistral risk-report LLM, and score/report transactions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(APP_ROOT / path)


def _new_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": None,
        "error": None,
    }
    return job_id


def _mark_job(job_id: str, status: str, result: Any = None, error: str | None = None) -> None:
    JOBS[job_id].update({
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "error": error,
    })


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app_root": str(APP_ROOT)}


@app.post("/api/datasets/generate")
def generate_dataset(request: DatasetGenerateRequest) -> Dict[str, Any]:
    """Generate raw, engineered, and split XGBoost dataset CSV files."""
    config = DatasetBuildConfig(
        n_rows=request.n_rows,
        fraud_rate=request.fraud_rate,
        start_date=request.start_date,
        end_date=request.end_date,
        seed=request.seed,
        output_dir=_resolve_path(request.output_dir),
        dataset_name=request.dataset_name,
        generate_engineered_csv=request.generate_engineered_csv,
        generate_split_csvs=request.generate_split_csvs,
        temporal_split=request.temporal_split,
        graph_feature_mode=request.graph_feature_mode,
        graph_score_path=_resolve_path(request.graph_score_path) if request.graph_score_path else None,
    )
    try:
        result = XGBoostDatasetBuilder(config).build()
        return {"status": "completed", "result": result}
    except Exception as exc:
        logger.exception("Dataset generation failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/datasets/enrich-with-gnn")
def enrich_dataset_with_gnn(request: EnrichXGBoostWithGNNRequest) -> Dict[str, Any]:
    """Create a new raw XGBoost dataset by injecting trained GNN account scores."""
    config = GNNScoreInjectionConfig(
        source_transaction_csv=_resolve_path(request.source_transaction_csv),
        gnn_score_path=_resolve_path(request.gnn_score_path),
        output_dir=_resolve_path(request.output_dir),
        output_dataset_name=request.output_dataset_name,
    )
    try:
        return {"status": "completed", "result": GNNScoreInjector(config).inject()}
    except Exception as exc:
        logger.exception("GNN score injection failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/xgboost/train")
def train_xgboost(request: TrainXGBoostRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Train the XGBoost model from a raw training CSV."""
    raw_path = request.raw_data_path or "data/xgboost/xgboost_training_data.csv"
    raw_path = _resolve_path(raw_path)
    artifact_dir = _resolve_path(request.artifact_dir)

    if not Path(raw_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Raw dataset not found: {raw_path}. Call /api/datasets/generate first or pass raw_data_path.",
        )

    def _run() -> Dict[str, Any]:
        return XGBoostTrainingService(artifact_dir=artifact_dir).train(
            raw_data_path=raw_path,
            model_version=request.model_version,
            run_hpo=request.run_hpo,
            n_hpo_trials=request.n_hpo_trials,
            temporal_split=request.temporal_split,
        )

    if request.async_mode:
        return _run_async("xgboost_training", _run, background_tasks)

    try:
        result = _run()
        return {"status": "completed", "result": result}
    except Exception as exc:
        logger.exception("XGBoost training failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/xgboost/score")
def score_xgboost(request: ScoreRequest) -> Dict[str, Any]:
    """Score one or more raw transaction records using a trained XGBoost model."""
    artifact_dir = _resolve_path(request.artifact_dir)
    try:
        service = XGBoostInferenceService(artifact_dir=artifact_dir, model_version=request.model_version)
        return {"status": "completed", "scores": service.score_records(request.records)}
    except Exception as exc:
        logger.exception("XGBoost scoring failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/gnn/datasets/generate")
def generate_gnn_dataset(request: GraphDatasetGenerateRequest) -> Dict[str, Any]:
    """Build graph node/edge CSVs from the XGBoost raw transaction CSV."""
    config = GraphDatasetBuildConfig(
        source_transaction_csv=_resolve_path(request.source_transaction_csv),
        output_dir=_resolve_path(request.output_dir),
        dataset_name=request.dataset_name,
        seed=request.seed,
        max_shared_entity_edges=request.max_shared_entity_edges,
    )
    try:
        return {"status": "completed", "result": GraphDatasetBuilder(config).build()}
    except Exception as exc:
        logger.exception("GNN dataset generation failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/gnn/train")
def train_gnn(request: TrainGNNRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Train the account-level GraphSAGE model from generated graph CSVs."""
    config = GNNTrainConfig(
        graph_data_dir=_resolve_path(request.graph_data_dir),
        dataset_name=request.dataset_name,
        artifact_dir=_resolve_path(request.artifact_dir),
        model_version=request.model_version,
        device=request.device,
        hidden_dim=request.hidden_dim,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        dropout=request.dropout,
        patience=request.patience,
        seed=request.seed,
        include_shared_entity_edges=request.include_shared_entity_edges,
    )

    def _run() -> Dict[str, Any]:
        return GNNTrainingService(config).train()

    if request.async_mode:
        return _run_async("gnn_training", _run, background_tasks)
    try:
        return {"status": "completed", "result": _run()}
    except Exception as exc:
        logger.exception("GNN training failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/gnn/infer")
def infer_gnn(request: InferGNNRequest) -> Dict[str, Any]:
    """Run batch GNN inference and export account-level fraud scores to CSV."""
    config = GNNInferConfig(
        graph_data_dir=_resolve_path(request.graph_data_dir),
        dataset_name=request.dataset_name,
        artifact_dir=_resolve_path(request.artifact_dir),
        model_version=request.model_version,
        output_path=_resolve_path(request.output_path),
        device=request.device,
    )
    try:
        return {"status": "completed", "result": GNNInferenceService(config).infer()}
    except Exception as exc:
        logger.exception("GNN inference failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/pipeline/train-hybrid")
def train_hybrid_pipeline(request: HybridPipelineRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Run the full local hybrid pipeline: base data → graph → GNN → GNN scores → XGBoost."""

    def _run() -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        base = XGBoostDatasetBuilder(DatasetBuildConfig(
            n_rows=request.n_rows,
            fraud_rate=request.fraud_rate,
            start_date=request.start_date,
            end_date=request.end_date,
            seed=request.seed,
            output_dir=_resolve_path("data/xgboost"),
            dataset_name=request.base_dataset_name,
            generate_engineered_csv=False,
            generate_split_csvs=False,
            graph_feature_mode="synthetic",
        )).build()
        results["base_dataset"] = base

        graph = GraphDatasetBuilder(GraphDatasetBuildConfig(
            source_transaction_csv=base["latest_raw_csv_path"],
            output_dir=_resolve_path("data/gnn"),
            dataset_name=request.graph_dataset_name,
            seed=request.seed,
        )).build()
        results["graph_dataset"] = graph

        if request.train_gnn:
            gnn_train = GNNTrainingService(GNNTrainConfig(
                graph_data_dir=_resolve_path("data/gnn"),
                dataset_name=request.graph_dataset_name,
                artifact_dir=_resolve_path("artifacts/gnn"),
                model_version=request.gnn_version,
                seed=request.seed,
            )).train()
            results["gnn_training"] = gnn_train

        score_path = _resolve_path(f"data/gnn/gnn_account_scores_{request.gnn_version}.csv")
        gnn_scores = GNNInferenceService(GNNInferConfig(
            graph_data_dir=_resolve_path("data/gnn"),
            dataset_name=request.graph_dataset_name,
            artifact_dir=_resolve_path("artifacts/gnn"),
            model_version=request.gnn_version,
            output_path=score_path,
        )).infer()
        results["gnn_scores"] = gnn_scores

        enriched = GNNScoreInjector(GNNScoreInjectionConfig(
            source_transaction_csv=base["latest_raw_csv_path"],
            gnn_score_path=score_path,
            output_dir=_resolve_path("data/xgboost"),
            output_dataset_name=request.final_dataset_name,
        )).inject()
        results["xgboost_gnn_enriched_dataset"] = enriched

        if request.train_xgboost:
            xgb_train = XGBoostTrainingService(artifact_dir=_resolve_path("artifacts/xgboost")).train(
                raw_data_path=enriched["latest_csv_path"],
                model_version=request.xgboost_version,
                run_hpo=request.xgboost_run_hpo,
                temporal_split=True,
            )
            results["xgboost_training"] = xgb_train
        return results

    if request.async_mode:
        return _run_async("hybrid_pipeline", _run, background_tasks)
    try:
        return {"status": "completed", "result": _run()}
    except Exception as exc:
        logger.exception("Hybrid pipeline failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/llm/download-base")
def download_llm_base_model(request: DownloadLLMBaseModelRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Download the source/base Mistral model snapshot into artifacts/llm/base."""
    config = LLMDownloadConfig(
        model_id=request.model_id,
        output_dir=_resolve_path(request.output_dir),
        revision=request.revision,
        allow_patterns=request.allow_patterns,
        ignore_patterns=request.ignore_patterns,
    )

    def _run() -> Dict[str, Any]:
        return LLMModelDownloader(config).download()

    if request.async_mode:
        return _run_async("llm_base_download", _run, background_tasks)
    try:
        return {"status": "completed", "result": _run()}
    except Exception as exc:
        logger.exception("LLM base model download failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/llm/datasets/inspect")
def inspect_llm_dataset(request: LLMDatasetInspectRequest) -> Dict[str, Any]:
    """Validate SFT/DPO dataset schema and show the automatic train/validation/test split."""
    try:
        result = inspect_dataset_folder(
            _resolve_path(request.dataset_dir),
            dataset_type=request.dataset_type,
            validation_ratio=request.validation_ratio,
            test_ratio=request.test_ratio,
            seed=request.seed,
        )
        return {"status": "completed", "result": result}
    except Exception as exc:
        logger.exception("LLM dataset inspection failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/llm/finetune")
def fine_tune_llm(request: FineTuneLLMRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Fine-tune the downloaded Mistral model using SFT data and optional DPO preference data."""
    config = LLMFineTuneConfig(
        base_model_path=_resolve_path(request.base_model_path),
        output_dir=_resolve_path(request.output_dir),
        sft_dataset_dir=_resolve_path(request.sft_dataset_dir),
        dpo_dataset_dir=_resolve_path(request.dpo_dataset_dir),
        run_sft=request.run_sft,
        run_dpo=request.run_dpo,
        use_4bit=request.use_4bit,
        max_seq_length=request.max_seq_length,
        sft_epochs=request.sft_epochs,
        dpo_epochs=request.dpo_epochs,
        per_device_train_batch_size=request.per_device_train_batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        learning_rate=request.learning_rate,
        dpo_learning_rate=request.dpo_learning_rate,
        logging_steps=request.logging_steps,
        save_steps=request.save_steps,
        save_total_limit=request.save_total_limit,
        seed=request.seed,
        validation_ratio=request.validation_ratio,
        test_ratio=request.test_ratio,
        lora_r=request.lora_r,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
        lora_target_modules=request.lora_target_modules,
        resume_from_checkpoint=_resolve_path(request.resume_from_checkpoint) if request.resume_from_checkpoint else None,
    )

    def _run() -> Dict[str, Any]:
        return LLMFineTuningService(config).train()

    if request.async_mode:
        return _run_async("llm_fine_tuning", _run, background_tasks)
    try:
        return {"status": "completed", "result": _run()}
    except Exception as exc:
        logger.exception("LLM fine-tuning failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.post("/api/llm/infer-risk-report")
def infer_risk_report(request: RiskReportInferenceRequest) -> Dict[str, Any]:
    """Generate a risk assessment report from GNN+XGBoost classification output."""
    config = RiskReportInferenceConfig(
        base_model_path=_resolve_path(request.base_model_path),
        adapter_path=_resolve_path(request.adapter_path),
        use_4bit=request.use_4bit,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    try:
        return RiskReportInferenceService(config).generate_report(
            transaction=request.transaction,
            classification_result=request.classification_result,
            customer_context=request.customer_context,
            report_instruction=request.report_instruction,
        )
    except Exception as exc:
        logger.exception("LLM risk report inference failed")
        raise HTTPException(status_code=500, detail={"message": str(exc), "traceback": traceback.format_exc()})


@app.get("/api/llm/artifacts")
def list_llm_artifacts() -> Dict[str, Any]:
    paths = {
        "base_models": APP_ROOT / "artifacts" / "llm" / "base",
        "fine_tuned_models": APP_ROOT / "artifacts" / "llm" / "finetuned",
        "sft_datasets": APP_ROOT / "data" / "llm" / "sft",
        "dpo_datasets": APP_ROOT / "data" / "llm" / "dpo",
    }
    return {name: [str(p) for p in sorted(path.rglob("*")) if p.is_file()] if path.exists() else [] for name, path in paths.items()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/api/artifacts")
def list_artifacts() -> Dict[str, Any]:
    paths = {
        "xgboost_data_files": APP_ROOT / "data" / "xgboost",
        "gnn_data_files": APP_ROOT / "data" / "gnn",
        "llm_sft_datasets": APP_ROOT / "data" / "llm" / "sft",
        "llm_dpo_datasets": APP_ROOT / "data" / "llm" / "dpo",
        "xgboost_model_artifacts": APP_ROOT / "artifacts" / "xgboost",
        "gnn_model_artifacts": APP_ROOT / "artifacts" / "gnn",
        "llm_base_artifacts": APP_ROOT / "artifacts" / "llm" / "base",
        "llm_finetuned_artifacts": APP_ROOT / "artifacts" / "llm" / "finetuned",
    }
    return {name: [str(p) for p in sorted(path.glob("*"))] if path.exists() else [] for name, path in paths.items()}


def _run_async(job_type: str, fn, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    job_id = _new_job(job_type)

    def task():
        try:
            _mark_job(job_id, "running")
            result = fn()
            _mark_job(job_id, "completed", result=result)
        except Exception as exc:  # pragma: no cover - background safety
            logger.exception("Background job failed")
            _mark_job(job_id, "failed", error=f"{exc}\n{traceback.format_exc()}")

    background_tasks.add_task(task)
    return {"status": "accepted", "job_id": job_id, "poll_url": f"/api/jobs/{job_id}"}
