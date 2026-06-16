# FraudSentinel React Frontend Orchestrator

This frontend intentionally keeps orchestration in React. It does not add a backend orchestration endpoint.

## What the UI does

1. Generates a transaction record using the same JSON structure used in the notebook `payload["records"][0]`.
2. Calls the existing backend scoring endpoint:

```text
POST /api/xgboost/score
```

3. Prepares the notebook-aligned LLM request:

```json
{
  "transaction": {},
  "xgboost_score_response": {},
  "base_model_path": "mistralai/Mistral-7B-Instruct-v0.3",
  "adapter_path": "/workspace/shared/mistral_dpo_v3",
  "use_4bit": false,
  "torch_dtype": "bfloat16",
  "max_new_tokens": 2048,
  "temperature": 0.0,
  "top_p": 0.9
}
```

4. Calls the existing backend LLM endpoint:

```text
POST /api/llm/infer-fraud-report
```

5. Uses `parsed_report` if the backend returns it. If not, it tries to clean and parse `raw_response` on the frontend.
6. Shows a visual fraud investigation report.
7. Shows every step in the UI and marks it green when completed.

## Frontend location

```text
frontend/
```

## Run the backend

From the project root:

```bash
./local_start.sh
```

Or:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Run the React frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Default API assumptions

The frontend defaults are aligned with the notebook commands:

```text
Backend URL: http://localhost:8000
XGBoost artifact_dir: /workspace/shared/fraud_detection/artifacts/xgboost
XGBoost model_version: v1
Base model: mistralai/Mistral-7B-Instruct-v0.3
LoRA adapter: /workspace/shared/mistral_dpo_v3
Torch dtype: bfloat16
use_4bit: false
max_new_tokens: 2048
temperature: 0.0
top_p: 0.9
```

These settings are editable in the UI before running the report.

## Pipeline shown in the UI

The UI shows these steps:

```text
1. Transaction record ready
2. GNN + XGBoost classification completed
3. LLM inference request prepared
4. Local LLM inference completed
5. LLM JSON cleaned and parsed
6. Visual report rendered
```

Each step turns green when completed. If a backend call fails, the current step turns red and the error is shown at the top.

## Important note

The React app calls `/api/xgboost/score` directly for scoring. It does not call `/api/gnn/infer` during report generation because your trained XGBoost scoring response already includes `gnn_findings` and graph-derived features when the model artifacts were prepared using the notebook flow.


## Temporary public sharing

See `README_SHARE_URL.md` for the recommended Cloudflare Tunnel setup. Because the React app calls FastAPI directly from the browser, expose both the frontend and backend through tunnels. Then paste the backend tunnel URL into the UI's `Backend URL` field.
