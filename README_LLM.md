# Mistral 7B Risk Assessment LLM Extension

This project extension adds API support for:

1. Downloading a Mistral source model into `artifacts/llm/base`.
2. Fine-tuning with SFT data from `data/llm/sft`.
3. Optional DPO preference tuning with data from `data/llm/dpo`.
4. Storing LoRA/QLoRA adapters in `artifacts/llm/finetuned`.
5. Generating a fraud risk assessment report from the GNN+XGBoost classification output.

## Linux setup

```bash
chmod +x local_setup.sh local_start.sh
./local_setup.sh
./local_start.sh
```

Open:

```text
http://localhost:8000/docs
```

## Hugging Face token and environment defaults

`local_setup.sh` creates/updates `.env` automatically with all required defaults, including `HF_TOKEN`. You do not need to manually edit `.env` on the Linux instance.

All Python dependencies are now installed from the single main file:

```bash
pip install -r requirements.txt
```

`requirements-llm.txt` is no longer used.

## API flow

### 1. Download the source model

Default source model is `mistralai/Mistral-7B-Instruct-v0.3` because the risk report task is instruction-following.
If you truly want the raw pretrained base, set `model_id` to `mistralai/Mistral-7B-v0.3` and change `output_dir` accordingly.

```bash
curl -X POST "http://localhost:8000/api/llm/download-base" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    "output_dir": "artifacts/llm/base/mistral-7b-instruct-v0.3",
    "async_mode": true
  }'
```

Poll the returned job:

```bash
curl "http://localhost:8000/api/jobs/<job_id>"
```

### 2. Add SFT data

Put `.jsonl` or `.json` files inside:

```text
data/llm/sft/
```

Supported SFT formats:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

or:

```json
{"instruction":"Generate a fraud risk report", "input":"{...classification output...}", "output":"Risk report..."}
```

### 3. Add DPO data

Put `.jsonl` or `.json` files inside:

```text
data/llm/dpo/
```

Supported DPO format:

```json
{"prompt":"Generate a fraud risk report for this transaction...", "chosen":"Better report...", "rejected":"Worse report..."}
```

Conversational DPO is also supported:

```json
{
  "prompt": [{"role":"user", "content":"Generate a fraud risk report..."}],
  "chosen": [{"role":"assistant", "content":"Better report..."}],
  "rejected": [{"role":"assistant", "content":"Worse report..."}]
}
```

### 4. Fine-tune

```bash
curl -X POST "http://localhost:8000/api/llm/finetune" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_path": "artifacts/llm/base/mistral-7b-instruct-v0.3",
    "output_dir": "artifacts/llm/finetuned/risk-report-mistral-7b",
    "sft_dataset_dir": "data/llm/sft",
    "dpo_dataset_dir": "data/llm/dpo",
    "run_sft": true,
    "run_dpo": true,
    "use_4bit": true,
    "max_seq_length": 2048,
    "sft_epochs": 2,
    "dpo_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "async_mode": true
  }'
```

Poll:

```bash
curl "http://localhost:8000/api/jobs/<job_id>"
```

Final adapter will usually be:

```text
artifacts/llm/finetuned/risk-report-mistral-7b/dpo/final_adapter
```

If DPO is disabled, use:

```text
artifacts/llm/finetuned/risk-report-mistral-7b/sft/final_adapter
```

### 5. Load the notebook-aligned inference model once

The notebook loads:

```python
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_PATH = "/workspace/shared/mistral_dpo_v3"
torch_dtype = torch.bfloat16
```

The API defaults now match that setup. You can load the hosted runtime explicitly:

```bash
curl -X POST "http://localhost:8000/api/llm/load-inference-model" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_path": "mistralai/Mistral-7B-Instruct-v0.3",
    "adapter_path": "/workspace/shared/mistral_dpo_v3",
    "use_4bit": false,
    "torch_dtype": "bfloat16"
  }'
```

Check status:

```bash
curl "http://localhost:8000/api/llm/inference-status"
```

### 6. Generate a notebook-aligned JSON fraud report

The API accepts either `classification_result` or the notebook's exact `xgboost_score_response` field name. The prompt sent to the LLM uses the same DPO-style input as the notebook:

```json
{
  "task": "generate_fraud_investigation_outputs",
  "transaction": {},
  "xgboost_score_response": {},
  "operator_note": "Use only supplied transaction, model outputs, graph findings, feature contributions and risk factors. Do not invent facts, historical cases, customer information, regulatory citations or unsupported typologies."
}
```

Example call after `/api/xgboost/score`:

```bash
curl -X POST "http://localhost:8000/api/llm/infer-fraud-report" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "transaction_id": "test-txn-031",
      "account_id_hash": "test-account-001",
      "beneficiary_id_hash": "test-beneficiary-201",
      "amount_inr": 958000,
      "amount_usd_equiv": 11445,
      "channel": "UPI",
      "merchant_category_code": "6012",
      "is_international": false,
      "currency": "INR"
    },
    "xgboost_score_response": {
      "probability": 1.0,
      "probability_percentage": 100.0,
      "risk_score": 100,
      "is_fraud": true,
      "classification_threshold": 0.5,
      "gnn_findings": {
        "shared_device_accounts": 10,
        "shared_ip_accounts": 20,
        "beneficiary_sar_count": 1,
        "second_hop_sar_count": 3,
        "mule_network_probability": 0.78,
        "synthetic_identity_score": 0.42,
        "rapid_fan_out_flag": true,
        "originator_graph_centrality": 0.55
      }
    },
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "torch_dtype": "bfloat16"
  }'
```

`/api/llm/infer-risk-report` is still available as a backward-compatible alias. React should prefer `/api/llm/infer-fraud-report` and render `parsed_report`.

## Important first-time fine-tuning checklist

- Create a held-out validation/evaluation set before training.
- Remove PII or tokenize/anonymize customer identifiers.
- Make sure reports are grounded in fields available at inference time.
- Include both fraud and non-fraud cases in SFT data.
- Include DPO pairs for report quality: concise vs verbose, grounded vs speculative, compliant vs non-compliant.
- Track model version, dataset version, hyperparameters, and metrics.
- Keep a rollback path to the previous adapter.
- Do not use the LLM output as the fraud decision; keep it as an explanation/reporting layer after GNN+XGBoost classification.
