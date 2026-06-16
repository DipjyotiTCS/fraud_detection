from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RiskReportInferenceConfig:
    base_model_path: str = "mistralai/Mistral-7B-Instruct-v0.3"
    adapter_path: str = "/workspace/shared/mistral_dpo_v3"
    use_4bit: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    torch_dtype: str = "bfloat16"
    max_input_tokens: int = 8192


class _HostedLLMRuntime:
    """Process-local cache for the hosted base model + LoRA adapter."""

    tokenizer = None
    model = None
    base_model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    use_4bit: Optional[bool] = None
    torch_dtype: Optional[str] = None
    is_loaded: bool = False


RUNTIME = _HostedLLMRuntime()


def _resolve_torch_dtype(dtype_name: str):
    import torch

    dtype_name = (dtype_name or "auto").lower()

    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32

    return torch.float16


def _runtime_device() -> str:
    import torch

    # In PyTorch, AMD ROCm generally appears through the CUDA API.
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clean_json_text(response_text: str) -> str:
    text = (response_text or "").strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    json_text = text[start:] if end == -1 else text[start:end + 1]

    # Common local-generation issue: response is valid JSON but truncated by one or more braces.
    open_braces = json_text.count("{")
    close_braces = json_text.count("}")
    if open_braces > close_braces:
        json_text += "}" * (open_braces - close_braces)

    # Remove common trailing commas.
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)

    return json_text


def parse_llm_json(response_text: str) -> Dict[str, Any]:
    return json.loads(_clean_json_text(response_text))


def build_fraud_report_messages(
    transaction: Dict[str, Any],
    xgboost_score_response: Dict[str, Any],
    customer_context: Optional[Dict[str, Any]] = None,
    report_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Builds the same DPO-style input shape used in the notebook."""

    llm_request = {
        "task": "generate_fraud_investigation_outputs",
        "transaction": transaction,
        "xgboost_score_response": xgboost_score_response,
        "operator_note": report_instruction or (
            "Use only supplied transaction, model outputs, graph findings, "
            "feature contributions and risk factors. Do not invent facts, "
            "historical cases, customer information, regulatory citations "
            "or unsupported typologies."
        ),
    }

    # Keep the default payload exactly aligned with the notebook. Only add
    # customer_context when a caller explicitly provides it.
    if customer_context:
        llm_request["customer_context"] = customer_context

    return [
        {
            "role": "system",
            "content": (
                "You are FraudSentinel, a fraud investigation report generation assistant. "
                "Use only the supplied transaction and xgboost_score_response. "
                "Generate fraud typology, fraud classification, rationale, "
                "SAR narrative, and next best action. "
                "Do not invent facts, historical cases, regulatory citations, "
                "feature values, model outputs, or graph findings. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(llm_request, ensure_ascii=False, default=str),
        },
    ]


class RiskReportInferenceService:
    """Hosted local LLM runtime for fraud report generation.

    This service keeps the base model and LoRA adapter loaded in memory so the
    React frontend can call a fast inference endpoint instead of triggering a
    full model load on every request.
    """

    def __init__(self, config: RiskReportInferenceConfig):
        self.config = config

    @staticmethod
    def status() -> Dict[str, Any]:
        return {
            "model_loaded": RUNTIME.is_loaded,
            "base_model_path": RUNTIME.base_model_path,
            "adapter_path": RUNTIME.adapter_path,
            "use_4bit": RUNTIME.use_4bit,
            "torch_dtype": RUNTIME.torch_dtype,
            "device": _runtime_device(),
        }

    @staticmethod
    def unload() -> Dict[str, Any]:
        import torch

        RUNTIME.model = None
        RUNTIME.tokenizer = None
        RUNTIME.base_model_path = None
        RUNTIME.adapter_path = None
        RUNTIME.use_4bit = None
        RUNTIME.torch_dtype = None
        RUNTIME.is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "unloaded",
            "model_loaded": False,
            "device": _runtime_device(),
        }

    def load_model(self) -> Dict[str, Any]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_path = str(self.config.base_model_path)
        adapter_path = str(self.config.adapter_path) if self.config.adapter_path else ""

        same_runtime = (
            RUNTIME.is_loaded
            and RUNTIME.model is not None
            and RUNTIME.tokenizer is not None
            and RUNTIME.base_model_path == base_model_path
            and RUNTIME.adapter_path == adapter_path
            and RUNTIME.use_4bit == self.config.use_4bit
            and RUNTIME.torch_dtype == self.config.torch_dtype
        )
        if same_runtime:
            return {"status": "already_loaded", **self.status()}

        self.unload()

        torch_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        device = _runtime_device()

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }

        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch.float32

        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "use_4bit=True requires bitsandbytes support. For AMD ROCm inference, "
                    "set use_4bit=false and use bf16/fp16 instead."
                ) from exc

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        if device != "cuda":
            base_model.to("cpu")

        if adapter_path:
            adapter_is_hf_repo = (
                not Path(adapter_path).is_absolute()
                and not adapter_path.startswith((".", "artifacts/", "data/"))
                and "/" in adapter_path
            )
            if not Path(adapter_path).exists() and not adapter_is_hf_repo:
                raise FileNotFoundError(
                    f"LoRA adapter path was provided but does not exist: {adapter_path}"
                )
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model

        model.eval()

        RUNTIME.tokenizer = tokenizer
        RUNTIME.model = model
        RUNTIME.base_model_path = base_model_path
        RUNTIME.adapter_path = adapter_path or None
        RUNTIME.use_4bit = self.config.use_4bit
        RUNTIME.torch_dtype = self.config.torch_dtype
        RUNTIME.is_loaded = True

        return {"status": "loaded", **self.status()}

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        import torch

        self.load_model()

        tokenizer = RUNTIME.tokenizer
        model = RUNTIME.model
        if tokenizer is None or model is None:
            raise RuntimeError("LLM runtime is not loaded.")

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )

        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate_report(
        self,
        transaction: Dict[str, Any],
        classification_result: Dict[str, Any],
        customer_context: Optional[Dict[str, Any]] = None,
        report_instruction: Optional[str] = None,
    ) -> dict:
        messages = build_fraud_report_messages(
            transaction=transaction,
            xgboost_score_response=classification_result,
            customer_context=customer_context,
            report_instruction=report_instruction,
        )

        raw_response = self._generate(messages)

        parsed_report = None
        parse_error = None
        try:
            parsed_report = parse_llm_json(raw_response)
        except Exception as exc:
            parse_error = str(exc)

        return {
            "status": "completed",
            "model_loaded": RUNTIME.is_loaded,
            "adapter_used": RUNTIME.adapter_path,
            "base_model_path": RUNTIME.base_model_path,
            "raw_response": raw_response,
            "parsed_report": parsed_report,
            "parse_error": parse_error,
            # Backward-compatible alias for the older API contract.
            "report": raw_response,
        }
