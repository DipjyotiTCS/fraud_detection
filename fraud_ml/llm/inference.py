from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _preview_text(text: str, limit: int = 1000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... <truncated {len(text) - limit} chars>"


def _format_bytes(num_bytes: Optional[int]) -> Optional[str]:
    if num_bytes is None:
        return None
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _cuda_memory_snapshot() -> Dict[str, Any]:
    """Return a best-effort GPU memory snapshot.

    On AMD ROCm builds, PyTorch still exposes the device through torch.cuda.
    This function is intentionally defensive so logging never breaks inference.
    """

    try:
        import torch

        snapshot: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            snapshot.update(
                {
                    "current_device": current_device,
                    "device_name": torch.cuda.get_device_name(current_device),
                    "allocated": _format_bytes(torch.cuda.memory_allocated(current_device)),
                    "reserved": _format_bytes(torch.cuda.memory_reserved(current_device)),
                    "max_allocated": _format_bytes(torch.cuda.max_memory_allocated(current_device)),
                    "max_reserved": _format_bytes(torch.cuda.max_memory_reserved(current_device)),
                }
            )
        return snapshot
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"memory_snapshot_error": str(exc)}


def _runtime_snapshot() -> Dict[str, Any]:
    """Return runtime/version details useful for ROCm troubleshooting."""

    snapshot: Dict[str, Any] = {}
    try:
        import torch

        snapshot.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "torch_cuda_build": getattr(torch.version, "cuda", None),
                "torch_hip_build": getattr(torch.version, "hip", None),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
            }
        )
        if torch.cuda.is_available():
            snapshot["cuda_current_device"] = torch.cuda.current_device()
            snapshot["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception as exc:  # pragma: no cover - environment dependent
        snapshot["torch_runtime_snapshot_error"] = str(exc)

    try:
        import transformers

        snapshot["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception as exc:  # pragma: no cover - environment dependent
        snapshot["transformers_version_error"] = str(exc)

    try:
        import peft

        snapshot["peft_version"] = getattr(peft, "__version__", None)
    except Exception as exc:  # pragma: no cover - environment dependent
        snapshot["peft_version_error"] = str(exc)

    return snapshot


def _model_debug_snapshot(model: Any) -> Dict[str, Any]:
    """Return model placement details without assuming a specific model class."""

    snapshot: Dict[str, Any] = {}
    try:
        snapshot["model_class"] = model.__class__.__name__
        snapshot["model_device"] = str(getattr(model, "device", None))
        snapshot["is_peft_model"] = model.__class__.__name__.lower().startswith("peft")
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map is not None:
            # Avoid logging hundreds of entries for very large device maps.
            if isinstance(hf_device_map, dict):
                items = list(hf_device_map.items())
                snapshot["hf_device_map_size"] = len(items)
                snapshot["hf_device_map_preview"] = dict(items[:20])
            else:
                snapshot["hf_device_map"] = str(hf_device_map)
        try:
            first_param = next(model.parameters())
            snapshot["first_parameter_device"] = str(first_param.device)
            snapshot["first_parameter_dtype"] = str(first_param.dtype)
        except Exception as exc:
            snapshot["first_parameter_error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive logging only
        snapshot["model_snapshot_error"] = str(exc)
    return snapshot


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
    logger.info("Resolving torch dtype", extra={"requested_dtype": dtype_name})

    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logger.info("Resolved auto dtype to torch.bfloat16")
            return torch.bfloat16
        if torch.cuda.is_available():
            logger.info("Resolved auto dtype to torch.float16")
            return torch.float16
        logger.warning("Resolved auto dtype to torch.float32 because no torch.cuda/HIP device is visible")
        return torch.float32

    if dtype_name in {"bf16", "bfloat16"}:
        logger.info("Resolved dtype to torch.bfloat16")
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        logger.info("Resolved dtype to torch.float16")
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        logger.info("Resolved dtype to torch.float32")
        return torch.float32

    logger.warning("Unknown torch dtype requested; falling back to torch.float16", extra={"requested_dtype": dtype_name})
    return torch.float16


def _from_pretrained_with_dtype(model_cls, base_model_path: str, torch_dtype, model_kwargs: Dict[str, Any]):
    """Load model with new Transformers `dtype` argument, falling back to `torch_dtype`.

    Recent Transformers versions warn that `torch_dtype` is deprecated. Older
    versions may not accept `dtype`, so this keeps the app compatible with both.
    """

    logger.info(
        "Calling AutoModelForCausalLM.from_pretrained with dtype argument",
        extra={
            "base_model_path": base_model_path,
            "dtype": str(torch_dtype),
            "model_kwargs_keys": sorted(model_kwargs.keys()),
        },
    )
    try:
        return model_cls.from_pretrained(
            base_model_path,
            dtype=torch_dtype,
            **model_kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            logger.exception("Model load failed before fallback to torch_dtype")
            raise
        logger.warning(
            "Transformers version does not accept dtype=; retrying with torch_dtype=",
            extra={"error": str(exc)},
        )
        return model_cls.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )


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
    logger.info("Parsing LLM response as JSON", extra={"raw_response_chars": len(response_text or "")})
    json_text = _clean_json_text(response_text)
    logger.info("Cleaned LLM JSON text", extra={"cleaned_json_chars": len(json_text or "")})
    return json.loads(json_text)


def build_fraud_report_messages(
    transaction: Dict[str, Any],
    xgboost_score_response: Dict[str, Any],
    customer_context: Optional[Dict[str, Any]] = None,
    report_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Builds the same DPO-style input shape used in the notebook."""

    tx_id = transaction.get("transaction_id")
    logger.info(
        "Building fraud report chat messages",
        extra={
            "transaction_id": tx_id,
            "transaction_field_count": len(transaction or {}),
            "score_response_field_count": len(xgboost_score_response or {}),
            "has_customer_context": bool(customer_context),
            "has_custom_instruction": bool(report_instruction),
        },
    )

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

    user_content = json.dumps(llm_request, ensure_ascii=False, default=str)
    logger.info(
        "Built LLM request payload",
        extra={
            "transaction_id": tx_id,
            "llm_request_chars": len(user_content),
            "log_prompt_enabled": _bool_env("FRAUD_LLM_LOG_PROMPT", False),
        },
    )
    if _bool_env("FRAUD_LLM_LOG_PROMPT", False):
        logger.debug("LLM request payload preview: %s", _preview_text(user_content, limit=4000))

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
            "content": user_content,
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
        logger.info(
            "RiskReportInferenceService initialized",
            extra={
                "base_model_path": str(config.base_model_path),
                "adapter_path": str(config.adapter_path),
                "use_4bit": config.use_4bit,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "torch_dtype": config.torch_dtype,
                "max_input_tokens": config.max_input_tokens,
            },
        )

    @staticmethod
    def status() -> Dict[str, Any]:
        status_payload = {
            "model_loaded": RUNTIME.is_loaded,
            "base_model_path": RUNTIME.base_model_path,
            "adapter_path": RUNTIME.adapter_path,
            "use_4bit": RUNTIME.use_4bit,
            "torch_dtype": RUNTIME.torch_dtype,
            "device": _runtime_device(),
            "runtime_snapshot": _runtime_snapshot(),
            "gpu_memory": _cuda_memory_snapshot(),
        }
        logger.info("LLM inference runtime status requested", extra={"status": status_payload})
        return status_payload

    @staticmethod
    def unload() -> Dict[str, Any]:
        import torch

        logger.info(
            "Unloading LLM runtime",
            extra={
                "was_loaded": RUNTIME.is_loaded,
                "base_model_path": RUNTIME.base_model_path,
                "adapter_path": RUNTIME.adapter_path,
                "gpu_memory_before": _cuda_memory_snapshot(),
            },
        )

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

        result = {
            "status": "unloaded",
            "model_loaded": False,
            "device": _runtime_device(),
            "gpu_memory_after": _cuda_memory_snapshot(),
        }
        logger.info("LLM runtime unloaded", extra={"result": result})
        return result

    def load_model(self) -> Dict[str, Any]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_started = time.perf_counter()
        base_model_path = str(self.config.base_model_path)
        adapter_path = str(self.config.adapter_path) if self.config.adapter_path else ""

        logger.info(
            "Starting LLM model load",
            extra={
                "base_model_path": base_model_path,
                "adapter_path": adapter_path or None,
                "use_4bit": self.config.use_4bit,
                "torch_dtype": self.config.torch_dtype,
                "runtime_snapshot": _runtime_snapshot(),
                "gpu_memory_before_load": _cuda_memory_snapshot(),
            },
        )

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
            logger.info(
                "Reusing already-loaded LLM runtime",
                extra={
                    "base_model_path": base_model_path,
                    "adapter_path": adapter_path or None,
                    "gpu_memory": _cuda_memory_snapshot(),
                    "model_snapshot": _model_debug_snapshot(RUNTIME.model),
                },
            )
            return {"status": "already_loaded", **self.status()}

        if RUNTIME.is_loaded:
            logger.info(
                "Existing runtime differs from requested config; unloading before reload",
                extra={
                    "loaded_base_model_path": RUNTIME.base_model_path,
                    "loaded_adapter_path": RUNTIME.adapter_path,
                    "requested_base_model_path": base_model_path,
                    "requested_adapter_path": adapter_path or None,
                },
            )
        self.unload()

        torch_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        logger.info(
            "Resolved model load dtype",
            extra={"requested_torch_dtype": self.config.torch_dtype, "resolved_torch_dtype": str(torch_dtype)},
        )

        try:
            logger.info("Loading tokenizer", extra={"base_model_path": base_model_path})
            tokenizer_started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )
            logger.info(
                "Tokenizer loaded",
                extra={
                    "duration_seconds": round(time.perf_counter() - tokenizer_started, 3),
                    "tokenizer_class": tokenizer.__class__.__name__,
                    "vocab_size": getattr(tokenizer, "vocab_size", None),
                    "model_max_length": getattr(tokenizer, "model_max_length", None),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                },
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(
                    "Tokenizer pad token was missing; set pad_token to eos_token",
                    extra={"pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id},
                )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            if self.config.use_4bit:
                logger.warning(
                    "use_4bit=True requested. This may not work in AMD ROCm environments unless bitsandbytes support is configured."
                )
                try:
                    from transformers import BitsAndBytesConfig
                except Exception as exc:  # pragma: no cover - environment dependent
                    logger.exception("BitsAndBytesConfig import failed")
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
                logger.info("Configured 4-bit quantization", extra={"bnb_compute_dtype": str(torch_dtype)})

            logger.info(
                "Loading base model",
                extra={
                    "base_model_path": base_model_path,
                    "model_kwargs": {k: str(v) for k, v in model_kwargs.items()},
                    "gpu_memory_before_base_model": _cuda_memory_snapshot(),
                },
            )
            base_model_started = time.perf_counter()
            base_model = _from_pretrained_with_dtype(
                AutoModelForCausalLM,
                base_model_path,
                torch_dtype,
                model_kwargs,
            )
            logger.info(
                "Base model loaded",
                extra={
                    "duration_seconds": round(time.perf_counter() - base_model_started, 3),
                    "model_snapshot": _model_debug_snapshot(base_model),
                    "gpu_memory_after_base_model": _cuda_memory_snapshot(),
                },
            )

            if adapter_path:
                adapter_is_hf_repo = (
                    not Path(adapter_path).is_absolute()
                    and not adapter_path.startswith((".", "artifacts/", "data/"))
                    and "/" in adapter_path
                )
                adapter_exists = Path(adapter_path).exists()
                logger.info(
                    "Preparing LoRA adapter load",
                    extra={
                        "adapter_path": adapter_path,
                        "adapter_exists": adapter_exists,
                        "adapter_is_hf_repo": adapter_is_hf_repo,
                    },
                )
                if not adapter_exists and not adapter_is_hf_repo:
                    logger.error("LoRA adapter path does not exist", extra={"adapter_path": adapter_path})
                    raise FileNotFoundError(
                        f"LoRA adapter path was provided but does not exist: {adapter_path}"
                    )

                adapter_started = time.perf_counter()
                model = PeftModel.from_pretrained(base_model, adapter_path)
                logger.info(
                    "LoRA adapter loaded",
                    extra={
                        "duration_seconds": round(time.perf_counter() - adapter_started, 3),
                        "model_snapshot": _model_debug_snapshot(model),
                        "gpu_memory_after_adapter": _cuda_memory_snapshot(),
                    },
                )
            else:
                logger.info("No LoRA adapter path supplied; using base model only")
                model = base_model

            model.eval()
            logger.info("Model set to eval mode", extra={"model_snapshot": _model_debug_snapshot(model)})

            RUNTIME.tokenizer = tokenizer
            RUNTIME.model = model
            RUNTIME.base_model_path = base_model_path
            RUNTIME.adapter_path = adapter_path or None
            RUNTIME.use_4bit = self.config.use_4bit
            RUNTIME.torch_dtype = self.config.torch_dtype
            RUNTIME.is_loaded = True

            result = {"status": "loaded", **self.status()}
            logger.info(
                "LLM model load completed",
                extra={
                    "duration_seconds": round(time.perf_counter() - load_started, 3),
                    "result_status": result.get("status"),
                    "gpu_memory_after_load": _cuda_memory_snapshot(),
                },
            )
            return result
        except Exception:
            logger.exception(
                "LLM model load failed",
                extra={
                    "base_model_path": base_model_path,
                    "adapter_path": adapter_path or None,
                    "duration_seconds": round(time.perf_counter() - load_started, 3),
                    "gpu_memory_on_failure": _cuda_memory_snapshot(),
                },
            )
            raise

    def _generate(self, messages: List[Dict[str, str]], request_id: Optional[str] = None) -> str:
        import torch

        request_id = request_id or uuid.uuid4().hex[:12]
        generation_started = time.perf_counter()
        logger.info(
            "Starting LLM generation",
            extra={
                "request_id": request_id,
                "message_count": len(messages or []),
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "gpu_memory_before_generation": _cuda_memory_snapshot(),
            },
        )

        self.load_model()

        tokenizer = RUNTIME.tokenizer
        model = RUNTIME.model
        if tokenizer is None or model is None:
            logger.error("LLM runtime is not loaded after load_model call", extra={"request_id": request_id})
            raise RuntimeError("LLM runtime is not loaded.")

        try:
            logger.info("Applying chat template", extra={"request_id": request_id})
            prompt_started = time.perf_counter()
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.info(
                "Chat template applied",
                extra={
                    "request_id": request_id,
                    "duration_seconds": round(time.perf_counter() - prompt_started, 3),
                    "prompt_chars": len(prompt or ""),
                    "log_prompt_enabled": _bool_env("FRAUD_LLM_LOG_PROMPT", False),
                },
            )
            if _bool_env("FRAUD_LLM_LOG_PROMPT", False):
                logger.debug("LLM prompt preview request_id=%s: %s", request_id, _preview_text(prompt, limit=4000))

            logger.info("Tokenizing prompt", extra={"request_id": request_id})
            tokenize_started = time.perf_counter()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_tokens,
            )
            input_token_count = int(inputs["input_ids"].shape[-1])
            logger.info(
                "Prompt tokenized",
                extra={
                    "request_id": request_id,
                    "duration_seconds": round(time.perf_counter() - tokenize_started, 3),
                    "input_token_count": input_token_count,
                    "max_input_tokens": self.config.max_input_tokens,
                    "was_truncated_possible": input_token_count >= self.config.max_input_tokens,
                    "input_tensor_shape": list(inputs["input_ids"].shape),
                },
            )
            if input_token_count >= self.config.max_input_tokens:
                logger.warning(
                    "Prompt token count reached max_input_tokens; input may have been truncated",
                    extra={"request_id": request_id, "input_token_count": input_token_count},
                )

            target_device = getattr(model, "device", None)
            logger.info(
                "Moving tokenized inputs to model device",
                extra={
                    "request_id": request_id,
                    "model_device": str(target_device),
                    "model_snapshot": _model_debug_snapshot(model),
                },
            )
            inputs = inputs.to(model.device)

            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if self.config.temperature > 0:
                generation_kwargs["temperature"] = self.config.temperature
                generation_kwargs["top_p"] = self.config.top_p

            logger.info(
                "Calling model.generate",
                extra={
                    "request_id": request_id,
                    "generation_kwargs": generation_kwargs,
                    "gpu_memory_before_model_generate": _cuda_memory_snapshot(),
                },
            )
            model_generate_started = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
            logger.info(
                "model.generate completed",
                extra={
                    "request_id": request_id,
                    "duration_seconds": round(time.perf_counter() - model_generate_started, 3),
                    "output_tensor_shape": list(output_ids.shape),
                    "gpu_memory_after_model_generate": _cuda_memory_snapshot(),
                },
            )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            generated_token_count = int(generated_ids.shape[-1])
            logger.info(
                "Decoding generated tokens",
                extra={
                    "request_id": request_id,
                    "generated_token_count": generated_token_count,
                    "total_output_tokens": int(output_ids.shape[-1]),
                    "input_token_count": input_token_count,
                },
            )
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            logger.info(
                "LLM generation completed",
                extra={
                    "request_id": request_id,
                    "duration_seconds": round(time.perf_counter() - generation_started, 3),
                    "generated_token_count": generated_token_count,
                    "decoded_chars": len(decoded or ""),
                    "gpu_memory_after_generation": _cuda_memory_snapshot(),
                    "log_response_enabled": _bool_env("FRAUD_LLM_LOG_RESPONSE", True),
                },
            )
            if _bool_env("FRAUD_LLM_LOG_RESPONSE", True):
                logger.debug("LLM decoded response preview request_id=%s: %s", request_id, _preview_text(decoded, limit=4000))
            return decoded
        except Exception:
            logger.exception(
                "LLM generation failed",
                extra={
                    "request_id": request_id,
                    "duration_seconds": round(time.perf_counter() - generation_started, 3),
                    "gpu_memory_on_failure": _cuda_memory_snapshot(),
                },
            )
            raise

    def generate_report(
        self,
        transaction: Dict[str, Any],
        classification_result: Dict[str, Any],
        customer_context: Optional[Dict[str, Any]] = None,
        report_instruction: Optional[str] = None,
    ) -> dict:
        request_id = uuid.uuid4().hex[:12]
        report_started = time.perf_counter()
        tx_id = (transaction or {}).get("transaction_id")
        logger.info(
            "Fraud report inference started",
            extra={
                "request_id": request_id,
                "transaction_id": tx_id,
                "risk_score": (classification_result or {}).get("risk_score"),
                "probability": (classification_result or {}).get("probability"),
                "is_fraud": (classification_result or {}).get("is_fraud"),
                "classification_result_keys": sorted((classification_result or {}).keys()),
                "transaction_keys": sorted((transaction or {}).keys()),
            },
        )

        try:
            messages = build_fraud_report_messages(
                transaction=transaction,
                xgboost_score_response=classification_result,
                customer_context=customer_context,
                report_instruction=report_instruction,
            )
            logger.info(
                "Fraud report messages ready",
                extra={
                    "request_id": request_id,
                    "transaction_id": tx_id,
                    "message_roles": [message.get("role") for message in messages],
                    "message_char_lengths": [len(message.get("content") or "") for message in messages],
                },
            )

            raw_response = self._generate(messages, request_id=request_id)

            parsed_report = None
            parse_error = None
            try:
                parsed_report = parse_llm_json(raw_response)
                logger.info(
                    "LLM response JSON parsing succeeded",
                    extra={
                        "request_id": request_id,
                        "transaction_id": tx_id,
                        "parsed_report_keys": sorted(parsed_report.keys()) if isinstance(parsed_report, dict) else None,
                    },
                )
            except Exception as exc:
                parse_error = str(exc)
                logger.exception(
                    "LLM response JSON parsing failed",
                    extra={
                        "request_id": request_id,
                        "transaction_id": tx_id,
                        "raw_response_preview": _preview_text(raw_response, limit=2000),
                    },
                )

            response_payload = {
                "status": "completed",
                "request_id": request_id,
                "model_loaded": RUNTIME.is_loaded,
                "adapter_used": RUNTIME.adapter_path,
                "base_model_path": RUNTIME.base_model_path,
                "raw_response": raw_response,
                "parsed_report": parsed_report,
                "parse_error": parse_error,
                # Backward-compatible alias for the older API contract.
                "report": raw_response,
            }
            logger.info(
                "Fraud report inference completed",
                extra={
                    "request_id": request_id,
                    "transaction_id": tx_id,
                    "duration_seconds": round(time.perf_counter() - report_started, 3),
                    "parse_success": parsed_report is not None,
                    "parse_error": parse_error,
                    "raw_response_chars": len(raw_response or ""),
                },
            )
            return response_payload
        except Exception:
            logger.exception(
                "Fraud report inference failed",
                extra={
                    "request_id": request_id,
                    "transaction_id": tx_id,
                    "duration_seconds": round(time.perf_counter() - report_started, 3),
                    "gpu_memory_on_failure": _cuda_memory_snapshot(),
                },
            )
            raise
