#!/usr/bin/env bash
set -euo pipefail

# Linux/GPU local setup for FraudSentinel + Mistral fine-tuning.
# Usage:
#   chmod +x local_setup.sh
#   ./local_setup.sh
# Optional overrides before running setup:
#   HF_TOKEN=hf_your_token HOST=0.0.0.0 PORT=8000 ./local_setup.sh
#
# This script creates/updates .env with concrete default values for every
# property used by local setup, local startup, LLM download, fine-tuning,
# and inference. The API can then run without manually editing .env.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Setup/runtime defaults -------------------------------------------------
DEFAULT_HF_TOKEN="${DEFAULT_HF_TOKEN:-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}"
DEFAULT_VENV_DIR="${DEFAULT_VENV_DIR:-${PROJECT_ROOT}/.venv}"
DEFAULT_PYTHON_BIN="${DEFAULT_PYTHON_BIN:-python3}"
DEFAULT_INSTALL_GPU_TORCH="${DEFAULT_INSTALL_GPU_TORCH:-true}"
DEFAULT_PYTORCH_CUDA_INDEX_URL="${DEFAULT_PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DEFAULT_HOST="${DEFAULT_HOST:-0.0.0.0}"
DEFAULT_PORT="${DEFAULT_PORT:-8000}"
DEFAULT_RELOAD="${DEFAULT_RELOAD:-true}"

# ---- LLM/model defaults -----------------------------------------------------
DEFAULT_LLM_MODEL_ID="${DEFAULT_LLM_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
DEFAULT_LLM_MODEL_REVISION="${DEFAULT_LLM_MODEL_REVISION:-main}"
DEFAULT_LLM_BASE_MODEL_DIR="${DEFAULT_LLM_BASE_MODEL_DIR:-${PROJECT_ROOT}/artifacts/llm/base}"
DEFAULT_LLM_BASE_MODEL_PATH="${DEFAULT_LLM_BASE_MODEL_PATH:-${PROJECT_ROOT}/artifacts/llm/base/mistral-7b-instruct-v0.3}"
DEFAULT_LLM_FINETUNED_MODEL_DIR="${DEFAULT_LLM_FINETUNED_MODEL_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned}"
DEFAULT_LLM_FINETUNED_OUTPUT_DIR="${DEFAULT_LLM_FINETUNED_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b}"
DEFAULT_LLM_SFT_ADAPTER_DIR="${DEFAULT_LLM_SFT_ADAPTER_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b/sft/final_adapter}"
DEFAULT_LLM_DPO_ADAPTER_DIR="${DEFAULT_LLM_DPO_ADAPTER_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b/dpo/final_adapter}"
DEFAULT_LLM_FINAL_ADAPTER_DIR="${DEFAULT_LLM_FINAL_ADAPTER_DIR:-${PROJECT_ROOT}/artifacts/llm/finetuned/risk-report-mistral-7b/dpo/final_adapter}"
DEFAULT_LLM_SFT_DATASET_DIR="${DEFAULT_LLM_SFT_DATASET_DIR:-${PROJECT_ROOT}/data/llm/sft}"
DEFAULT_LLM_DPO_DATASET_DIR="${DEFAULT_LLM_DPO_DATASET_DIR:-${PROJECT_ROOT}/data/llm/dpo}"

# ---- Fine-tuning defaults ---------------------------------------------------
DEFAULT_LLM_RUN_SFT="${DEFAULT_LLM_RUN_SFT:-true}"
DEFAULT_LLM_RUN_DPO="${DEFAULT_LLM_RUN_DPO:-true}"
DEFAULT_LLM_USE_4BIT="${DEFAULT_LLM_USE_4BIT:-true}"
DEFAULT_LLM_MAX_SEQ_LENGTH="${DEFAULT_LLM_MAX_SEQ_LENGTH:-2048}"
DEFAULT_LLM_SFT_EPOCHS="${DEFAULT_LLM_SFT_EPOCHS:-2.0}"
DEFAULT_LLM_DPO_EPOCHS="${DEFAULT_LLM_DPO_EPOCHS:-1.0}"
DEFAULT_LLM_TRAIN_BATCH_SIZE="${DEFAULT_LLM_TRAIN_BATCH_SIZE:-1}"
DEFAULT_LLM_GRADIENT_ACCUMULATION_STEPS="${DEFAULT_LLM_GRADIENT_ACCUMULATION_STEPS:-8}"
DEFAULT_LLM_LEARNING_RATE="${DEFAULT_LLM_LEARNING_RATE:-0.0002}"
DEFAULT_LLM_DPO_LEARNING_RATE="${DEFAULT_LLM_DPO_LEARNING_RATE:-0.000005}"
DEFAULT_LLM_LOGGING_STEPS="${DEFAULT_LLM_LOGGING_STEPS:-5}"
DEFAULT_LLM_SAVE_STEPS="${DEFAULT_LLM_SAVE_STEPS:-100}"
DEFAULT_LLM_SAVE_TOTAL_LIMIT="${DEFAULT_LLM_SAVE_TOTAL_LIMIT:-2}"
DEFAULT_LLM_SEED="${DEFAULT_LLM_SEED:-42}"
DEFAULT_LLM_LORA_R="${DEFAULT_LLM_LORA_R:-16}"
DEFAULT_LLM_LORA_ALPHA="${DEFAULT_LLM_LORA_ALPHA:-32}"
DEFAULT_LLM_LORA_DROPOUT="${DEFAULT_LLM_LORA_DROPOUT:-0.05}"
DEFAULT_LLM_LORA_TARGET_MODULES="${DEFAULT_LLM_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# ---- Inference defaults -----------------------------------------------------
DEFAULT_LLM_MAX_NEW_TOKENS="${DEFAULT_LLM_MAX_NEW_TOKENS:-700}"
DEFAULT_LLM_TEMPERATURE="${DEFAULT_LLM_TEMPERATURE:-0.2}"
DEFAULT_LLM_TOP_P="${DEFAULT_LLM_TOP_P:-0.9}"
DEFAULT_LLM_ASYNC_MODE="${DEFAULT_LLM_ASYNC_MODE:-true}"

cd "${PROJECT_ROOT}"

value_or_default() {
  local current_value="${1:-}"
  local default_value="$2"
  if [[ -n "${current_value}" ]]; then
    printf '%s' "${current_value}"
  else
    printf '%s' "${default_value}"
  fi
}

upsert_env_property() {
  local key="$1"
  local value="$2"
  local env_file="${3:-.env}"

  touch "${env_file}"

  if grep -qE "^(export[[:space:]]+)?${key}=" "${env_file}"; then
    python - "$env_file" "$key" "$value" <<'PY'
import pathlib
import sys

env_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = env_path.read_text().splitlines()
updated = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
        updated.append(f'export {key}="{value}"')
    else:
        updated.append(line)
env_path.write_text("\n".join(updated).rstrip() + "\n")
PY
  else
    printf 'export %s="%s"\n' "${key}" "${value}" >> "${env_file}"
  fi
}

VENV_DIR="$(value_or_default "${VENV_DIR:-}" "${DEFAULT_VENV_DIR}")"
PYTHON_BIN="$(value_or_default "${PYTHON_BIN:-}" "${DEFAULT_PYTHON_BIN}")"
INSTALL_GPU_TORCH="$(value_or_default "${INSTALL_GPU_TORCH:-}" "${DEFAULT_INSTALL_GPU_TORCH}")"
PYTORCH_CUDA_INDEX_URL="$(value_or_default "${PYTORCH_CUDA_INDEX_URL:-}" "${DEFAULT_PYTORCH_CUDA_INDEX_URL}")"

# Ensure .env exists early and contains a warning header.
if [[ ! -f .env ]]; then
  cat > .env <<'EOF_ENV'
# Local Linux/GPU configuration for FraudSentinel LLM fine-tuning.
# Generated by local_setup.sh.
# Do not commit this file to Git because it contains credentials.
EOF_ENV
fi

# Write .env before installs as well, so failed dependency installation still leaves a complete config file.
upsert_env_property "HF_TOKEN" "$(value_or_default "${HF_TOKEN:-}" "${DEFAULT_HF_TOKEN}")" ".env"
upsert_env_property "FRAUD_ML_HOME" "${PROJECT_ROOT}" ".env"
upsert_env_property "VENV_DIR" "${VENV_DIR}" ".env"
upsert_env_property "PYTHON_BIN" "${PYTHON_BIN}" ".env"
upsert_env_property "INSTALL_GPU_TORCH" "${INSTALL_GPU_TORCH}" ".env"
upsert_env_property "PYTORCH_CUDA_INDEX_URL" "${PYTORCH_CUDA_INDEX_URL}" ".env"
upsert_env_property "HOST" "$(value_or_default "${HOST:-}" "${DEFAULT_HOST}")" ".env"
upsert_env_property "PORT" "$(value_or_default "${PORT:-}" "${DEFAULT_PORT}")" ".env"
upsert_env_property "RELOAD" "$(value_or_default "${RELOAD:-}" "${DEFAULT_RELOAD}")" ".env"
upsert_env_property "HF_HUB_ENABLE_HF_TRANSFER" "$(value_or_default "${HF_HUB_ENABLE_HF_TRANSFER:-}" "1")" ".env"
upsert_env_property "TOKENIZERS_PARALLELISM" "$(value_or_default "${TOKENIZERS_PARALLELISM:-}" "false")" ".env"
upsert_env_property "LLM_DEFAULT_MODEL_ID" "$(value_or_default "${LLM_DEFAULT_MODEL_ID:-}" "${DEFAULT_LLM_MODEL_ID}")" ".env"
upsert_env_property "LLM_DEFAULT_MODEL_REVISION" "$(value_or_default "${LLM_DEFAULT_MODEL_REVISION:-}" "${DEFAULT_LLM_MODEL_REVISION}")" ".env"
upsert_env_property "LLM_BASE_MODEL_DIR" "$(value_or_default "${LLM_BASE_MODEL_DIR:-}" "${DEFAULT_LLM_BASE_MODEL_DIR}")" ".env"
upsert_env_property "LLM_BASE_MODEL_PATH" "$(value_or_default "${LLM_BASE_MODEL_PATH:-}" "${DEFAULT_LLM_BASE_MODEL_PATH}")" ".env"
upsert_env_property "LLM_FINETUNED_MODEL_DIR" "$(value_or_default "${LLM_FINETUNED_MODEL_DIR:-}" "${DEFAULT_LLM_FINETUNED_MODEL_DIR}")" ".env"
upsert_env_property "LLM_FINETUNED_OUTPUT_DIR" "$(value_or_default "${LLM_FINETUNED_OUTPUT_DIR:-}" "${DEFAULT_LLM_FINETUNED_OUTPUT_DIR}")" ".env"
upsert_env_property "LLM_SFT_ADAPTER_DIR" "$(value_or_default "${LLM_SFT_ADAPTER_DIR:-}" "${DEFAULT_LLM_SFT_ADAPTER_DIR}")" ".env"
upsert_env_property "LLM_DPO_ADAPTER_DIR" "$(value_or_default "${LLM_DPO_ADAPTER_DIR:-}" "${DEFAULT_LLM_DPO_ADAPTER_DIR}")" ".env"
upsert_env_property "LLM_FINAL_ADAPTER_DIR" "$(value_or_default "${LLM_FINAL_ADAPTER_DIR:-}" "${DEFAULT_LLM_FINAL_ADAPTER_DIR}")" ".env"
upsert_env_property "LLM_SFT_DATASET_DIR" "$(value_or_default "${LLM_SFT_DATASET_DIR:-}" "${DEFAULT_LLM_SFT_DATASET_DIR}")" ".env"
upsert_env_property "LLM_DPO_DATASET_DIR" "$(value_or_default "${LLM_DPO_DATASET_DIR:-}" "${DEFAULT_LLM_DPO_DATASET_DIR}")" ".env"
upsert_env_property "LLM_RUN_SFT" "$(value_or_default "${LLM_RUN_SFT:-}" "${DEFAULT_LLM_RUN_SFT}")" ".env"
upsert_env_property "LLM_RUN_DPO" "$(value_or_default "${LLM_RUN_DPO:-}" "${DEFAULT_LLM_RUN_DPO}")" ".env"
upsert_env_property "LLM_USE_4BIT" "$(value_or_default "${LLM_USE_4BIT:-}" "${DEFAULT_LLM_USE_4BIT}")" ".env"
upsert_env_property "LLM_MAX_SEQ_LENGTH" "$(value_or_default "${LLM_MAX_SEQ_LENGTH:-}" "${DEFAULT_LLM_MAX_SEQ_LENGTH}")" ".env"
upsert_env_property "LLM_SFT_EPOCHS" "$(value_or_default "${LLM_SFT_EPOCHS:-}" "${DEFAULT_LLM_SFT_EPOCHS}")" ".env"
upsert_env_property "LLM_DPO_EPOCHS" "$(value_or_default "${LLM_DPO_EPOCHS:-}" "${DEFAULT_LLM_DPO_EPOCHS}")" ".env"
upsert_env_property "LLM_TRAIN_BATCH_SIZE" "$(value_or_default "${LLM_TRAIN_BATCH_SIZE:-}" "${DEFAULT_LLM_TRAIN_BATCH_SIZE}")" ".env"
upsert_env_property "LLM_GRADIENT_ACCUMULATION_STEPS" "$(value_or_default "${LLM_GRADIENT_ACCUMULATION_STEPS:-}" "${DEFAULT_LLM_GRADIENT_ACCUMULATION_STEPS}")" ".env"
upsert_env_property "LLM_LEARNING_RATE" "$(value_or_default "${LLM_LEARNING_RATE:-}" "${DEFAULT_LLM_LEARNING_RATE}")" ".env"
upsert_env_property "LLM_DPO_LEARNING_RATE" "$(value_or_default "${LLM_DPO_LEARNING_RATE:-}" "${DEFAULT_LLM_DPO_LEARNING_RATE}")" ".env"
upsert_env_property "LLM_LOGGING_STEPS" "$(value_or_default "${LLM_LOGGING_STEPS:-}" "${DEFAULT_LLM_LOGGING_STEPS}")" ".env"
upsert_env_property "LLM_SAVE_STEPS" "$(value_or_default "${LLM_SAVE_STEPS:-}" "${DEFAULT_LLM_SAVE_STEPS}")" ".env"
upsert_env_property "LLM_SAVE_TOTAL_LIMIT" "$(value_or_default "${LLM_SAVE_TOTAL_LIMIT:-}" "${DEFAULT_LLM_SAVE_TOTAL_LIMIT}")" ".env"
upsert_env_property "LLM_SEED" "$(value_or_default "${LLM_SEED:-}" "${DEFAULT_LLM_SEED}")" ".env"
upsert_env_property "LLM_LORA_R" "$(value_or_default "${LLM_LORA_R:-}" "${DEFAULT_LLM_LORA_R}")" ".env"
upsert_env_property "LLM_LORA_ALPHA" "$(value_or_default "${LLM_LORA_ALPHA:-}" "${DEFAULT_LLM_LORA_ALPHA}")" ".env"
upsert_env_property "LLM_LORA_DROPOUT" "$(value_or_default "${LLM_LORA_DROPOUT:-}" "${DEFAULT_LLM_LORA_DROPOUT}")" ".env"
upsert_env_property "LLM_LORA_TARGET_MODULES" "$(value_or_default "${LLM_LORA_TARGET_MODULES:-}" "${DEFAULT_LLM_LORA_TARGET_MODULES}")" ".env"
upsert_env_property "LLM_MAX_NEW_TOKENS" "$(value_or_default "${LLM_MAX_NEW_TOKENS:-}" "${DEFAULT_LLM_MAX_NEW_TOKENS}")" ".env"
upsert_env_property "LLM_TEMPERATURE" "$(value_or_default "${LLM_TEMPERATURE:-}" "${DEFAULT_LLM_TEMPERATURE}")" ".env"
upsert_env_property "LLM_TOP_P" "$(value_or_default "${LLM_TOP_P:-}" "${DEFAULT_LLM_TOP_P}")" ".env"
upsert_env_property "LLM_ASYNC_MODE" "$(value_or_default "${LLM_ASYNC_MODE:-}" "${DEFAULT_LLM_ASYNC_MODE}")" ".env"
chmod 600 .env || true

echo "[1/7] Checking Linux packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git git-lfs curl build-essential
  git lfs install || true
else
  echo "apt-get not found. Install python3-venv, python3-pip, git, git-lfs, curl, and build-essential manually."
fi

echo "[2/7] Creating virtual environment at ${VENV_DIR}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[3/7] Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/7] Installing PyTorch GPU wheel..."
if [[ "${INSTALL_GPU_TORCH}" == "true" ]]; then
  python -m pip install --upgrade torch torchvision torchaudio --index-url "${PYTORCH_CUDA_INDEX_URL}"
else
  echo "Skipping explicit GPU PyTorch install because INSTALL_GPU_TORCH=${INSTALL_GPU_TORCH}."
fi

echo "[5/7] Installing all project dependencies from single requirements.txt..."
python -m pip install -r requirements.txt

echo "[6/7] Creating model, dataset, artifact, and log directories..."
mkdir -p \
  "${DEFAULT_LLM_SFT_DATASET_DIR}" \
  "${DEFAULT_LLM_DPO_DATASET_DIR}" \
  "${DEFAULT_LLM_BASE_MODEL_DIR}" \
  "${DEFAULT_LLM_FINETUNED_MODEL_DIR}" \
  logs

echo "Created/updated .env with concrete defaults for all setup/start/LLM properties."

echo "[7/7] Verifying GPU availability..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found. If this is a GPU instance, install/check NVIDIA drivers."
fi

python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda device 0:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo "Setup completed. Activate with: source ${VENV_DIR}/bin/activate"
echo "Then start API with: ./local_start.sh"
