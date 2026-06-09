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
    EnrichXGBoostWithGNNRequest,
    GraphDatasetGenerateRequest,
    HybridPipelineRequest,
    InferGNNRequest,
    ScoreRequest,
    TrainGNNRequest,
    TrainXGBoostRequest,
)
from fraud_ml.gnn.graph_dataset_builder import GraphDatasetBuildConfig, GraphDatasetBuilder
from fraud_ml.gnn.graph_feature_injection import GNNScoreInjectionConfig, GNNScoreInjector
from fraud_ml.gnn.graph_inference import GNNInferConfig, GNNInferenceService
from fraud_ml.gnn.graph_trainer import GNNTrainConfig, GNNTrainingService
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
    description="Generate fraud datasets, train GNN/GraphSAGE account models, train XGBoost transaction models, and score transactions.",
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
        "xgboost_model_artifacts": APP_ROOT / "artifacts" / "xgboost",
        "gnn_model_artifacts": APP_ROOT / "artifacts" / "gnn",
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
