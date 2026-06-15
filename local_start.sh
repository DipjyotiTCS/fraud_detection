#!/usr/bin/env bash
set -euo pipefail

# Start the FastAPI backend locally on Linux.
# Usage:
#   chmod +x local_start.sh
#   ./local_start.sh
# Optional overrides:
#   HOST=0.0.0.0 PORT=8000 RELOAD=true ./local_start.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# Load .env first so VENV_DIR, HOST, PORT, and all LLM defaults come from the generated file.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env || true
  set +a
fi

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
RELOAD="${RELOAD:-true}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}. Run ./local_setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Safety defaults if local_start.sh is run without .env.
export HF_TOKEN="${HF_TOKEN:-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}"
export FRAUD_ML_HOME="${FRAUD_ML_HOME:-${PROJECT_ROOT}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LLM_DEFAULT_MODEL_ID="${LLM_DEFAULT_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
export LLM_DEFAULT_MODEL_REVISION="${LLM_DEFAULT_MODEL_REVISION:-main}"
export LLM_BASE_MODEL_DIR="${LLM_BASE_MODEL_DIR:-${PROJECT_ROOT}/artifacts/llm/base}"
export LLM_BASE_MODEL_PATH="${LLM_BASE_MODEL_PATH:-${PROJECT_ROOT}/artifacts/llm/base/mistral-7b-instruct-v0.3}"
export LLM_FINETUNED_MODEL_DIR="${LLM_FINETUNED_MODEL_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned}"
export LLM_FINETUNED_OUTPUT_DIR="${LLM_FINETUNED_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b}"
export LLM_SFT_ADAPTER_DIR="${LLM_SFT_ADAPTER_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b/sft/final_adapter}"
export LLM_DPO_ADAPTER_DIR="${LLM_DPO_ADAPTER_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b/dpo/final_adapter}"
export LLM_FINAL_ADAPTER_DIR="${LLM_FINAL_ADAPTER_DIR:-${LLM_DPO_ADAPTER_DIR}}"
export LLM_SFT_DATASET_DIR="${LLM_SFT_DATASET_DIR:-${PROJECT_ROOT}/data/llm/sft}"
export LLM_DPO_DATASET_DIR="${LLM_DPO_DATASET_DIR:-${PROJECT_ROOT}/data/llm/dpo}"

mkdir -p logs

echo "Starting FraudSentinel API"
echo "Project root: ${FRAUD_ML_HOME}"
echo "Swagger UI:   http://${HOST}:${PORT}/docs"
echo "Health API:   http://${HOST}:${PORT}/api/health"

if [[ "${RELOAD}" == "true" ]]; then
  exec uvicorn app.main:app --reload --host "${HOST}" --port "${PORT}"
else
  exec uvicorn app.main:app --host "${HOST}" --port "${PORT}"
fi
