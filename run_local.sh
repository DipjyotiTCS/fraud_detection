#!/usr/bin/env bash
set -euo pipefail
export FRAUD_ML_HOME="${FRAUD_ML_HOME:-$(pwd)}"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
