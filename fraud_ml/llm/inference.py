from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RiskReportInferenceConfig:
    base_model_path: str = "artifacts/llm/base/mistral-7b-instruct-v0.3"
    adapter_path: str = "artifacts/llm/finetuned/risk-report-mistral-7b/dpo/final_adapter"
    use_4bit: bool = True
    max_new_tokens: int = 700
    temperature: float = 0.2
    top_p: float = 0.9


class RiskReportInferenceService:
    """Generate a fraud risk assessment report from model/classifier outputs."""

    def __init__(self, config: RiskReportInferenceConfig):
        self.config = config

    def generate_report(
        self,
        transaction: Dict[str, Any],
        classification_result: Dict[str, Any],
        customer_context: Optional[Dict[str, Any]] = None,
        report_instruction: Optional[str] = None,
    ) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        base_path = str(Path(self.config.base_model_path).expanduser().resolve())
        adapter_path = str(Path(self.config.adapter_path).expanduser().resolve())
        tokenizer_path = adapter_path if Path(adapter_path).exists() else base_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            "trust_remote_code": False,
        }
        if self.config.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
        if Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        system_prompt = (
            "You are a fraud risk assessment analyst. Generate a concise, auditable, "
            "non-speculative risk assessment report using only the provided transaction, "
            "GNN/XGBoost classification output, and customer context. Include: risk summary, "
            "key risk drivers, graph/network indicators, recommended action, and confidence notes."
        )
        user_payload = {
            "transaction": transaction,
            "classification_result": classification_result,
            "customer_context": customer_context or {},
            "report_instruction": report_instruction or "Write a business-readable fraud risk assessment report.",
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2, default=str)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        report = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "status": "completed",
            "adapter_used": adapter_path if Path(adapter_path).exists() else None,
            "base_model_path": base_path,
            "report": report,
        }
