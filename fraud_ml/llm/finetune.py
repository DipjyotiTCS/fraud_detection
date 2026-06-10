from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fraud_ml.llm.dataset_utils import (
    dataset_folder_has_supported_files,
    load_dpo_dataset_splits,
    load_sft_dataset_splits,
)


@dataclass
class LLMFineTuneConfig:
    base_model_path: str = "artifacts/llm/base/mistral-7b-instruct-v0.3"
    output_dir: str = "artifacts/llm/finetuned/risk-report-mistral-7b"
    sft_dataset_dir: str = "data/llm/sft"
    dpo_dataset_dir: str = "data/llm/dpo"
    run_sft: bool = True
    run_dpo: bool = True
    use_4bit: bool = True
    max_seq_length: int = 2048
    sft_epochs: float = 2.0
    dpo_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    dpo_learning_rate: float = 5e-6
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    resume_from_checkpoint: Optional[str] = None


class LLMFineTuningService:
    """Run QLoRA-based SFT and optional DPO for risk-report generation."""

    def __init__(self, config: LLMFineTuneConfig):
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.sft_output_dir = self.output_dir / "sft"
        self.sft_adapter_dir = self.sft_output_dir / "final_adapter"
        self.dpo_output_dir = self.output_dir / "dpo"
        self.dpo_adapter_dir = self.dpo_output_dir / "final_adapter"

    def train(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "base_model_path": str(Path(self.config.base_model_path).expanduser().resolve()),
            "output_dir": str(self.output_dir),
            "sft_adapter_dir": None,
            "dpo_adapter_dir": None,
            "final_adapter_dir": None,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        if self.config.run_sft:
            result["sft"] = self._run_sft()
            result["sft_adapter_dir"] = str(self.sft_adapter_dir)
            result["final_adapter_dir"] = str(self.sft_adapter_dir)

        if self.config.run_dpo:
            if not dataset_folder_has_supported_files(self.config.dpo_dataset_dir):
                result["dpo"] = {
                    "status": "skipped",
                    "reason": f"No DPO .jsonl/.json files found in {self.config.dpo_dataset_dir}.",
                }
            else:
                if self.config.run_sft and not self.sft_adapter_dir.exists():
                    raise FileNotFoundError(f"SFT adapter not found for DPO: {self.sft_adapter_dir}")
                result["dpo"] = self._run_dpo()
                result["dpo_adapter_dir"] = str(self.dpo_adapter_dir)
                result["final_adapter_dir"] = str(self.dpo_adapter_dir)

        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        (self.output_dir / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (self.output_dir / "training_config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )
        return result

    def _bnb_config(self):
        import torch
        from transformers import BitsAndBytesConfig

        if not self.config.use_4bit:
            return None
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def _torch_dtype(self):
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load_tokenizer(self, path: str):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _lora_config(self):
        from peft import LoraConfig

        target_modules = self.config.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    def _run_sft(self) -> dict:
        import torch
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM
        from trl import SFTConfig, SFTTrainer

        base_path = str(Path(self.config.base_model_path).expanduser().resolve())
        tokenizer = self._load_tokenizer(base_path)
        dataset_bundle = load_sft_dataset_splits(
            self.config.sft_dataset_dir,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
        )
        self.sft_output_dir.mkdir(parents=True, exist_ok=True)
        (self.sft_output_dir / "dataset_split_summary.json").write_text(
            json.dumps(dataset_bundle.summary, indent=2), encoding="utf-8"
        )

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": self._torch_dtype(),
            "trust_remote_code": False,
        }
        bnb_config = self._bnb_config()
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
        model.config.use_cache = False
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        args = SFTConfig(
            output_dir=str(self.sft_output_dir),
            max_length=self.config.max_seq_length,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.sft_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to="none",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            seed=self.config.seed,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset_bundle.train_dataset,
            eval_dataset=dataset_bundle.validation_dataset if len(dataset_bundle.validation_dataset) else None,
            processing_class=tokenizer,
            peft_config=self._lora_config(),
        )
        train_output = trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        eval_metrics = self._safe_evaluate(trainer, "eval") if len(dataset_bundle.validation_dataset) else {}
        test_metrics = (
            self._safe_evaluate(trainer, "test", dataset_bundle.test_dataset)
            if len(dataset_bundle.test_dataset)
            else {}
        )
        trainer.save_model(str(self.sft_adapter_dir))
        tokenizer.save_pretrained(str(self.sft_adapter_dir))

        metrics = {
            "train": dict(train_output.metrics or {}),
            "validation": eval_metrics,
            "test": test_metrics,
        }
        (self.sft_output_dir / "sft_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._cleanup(model, trainer)
        return {
            "status": "completed",
            "records": dataset_bundle.summary["split_counts"],
            "dataset_summary": dataset_bundle.summary,
            "adapter_dir": str(self.sft_adapter_dir),
            "metrics": metrics,
        }

    def _run_dpo(self) -> dict:
        import torch
        from peft import AutoPeftModelForCausalLM
        from trl import DPOConfig, DPOTrainer

        tokenizer_source = str(self.sft_adapter_dir if self.sft_adapter_dir.exists() else Path(self.config.base_model_path))
        tokenizer = self._load_tokenizer(tokenizer_source)
        dataset_bundle = load_dpo_dataset_splits(
            self.config.dpo_dataset_dir,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
        )
        self.dpo_output_dir.mkdir(parents=True, exist_ok=True)
        (self.dpo_output_dir / "dataset_split_summary.json").write_text(
            json.dumps(dataset_bundle.summary, indent=2), encoding="utf-8"
        )

        model_kwargs = {
            "is_trainable": True,
            "device_map": "auto",
            "torch_dtype": self._torch_dtype(),
            "trust_remote_code": False,
        }
        bnb_config = self._bnb_config()
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoPeftModelForCausalLM.from_pretrained(str(self.sft_adapter_dir), **model_kwargs)
        model.config.use_cache = False

        args = DPOConfig(
            output_dir=str(self.dpo_output_dir),
            max_length=self.config.max_seq_length,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.dpo_learning_rate,
            num_train_epochs=self.config.dpo_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to="none",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            seed=self.config.seed,
            beta=0.1,
        )

        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset_bundle.train_dataset,
            eval_dataset=dataset_bundle.validation_dataset if len(dataset_bundle.validation_dataset) else None,
            processing_class=tokenizer,
        )
        train_output = trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        eval_metrics = self._safe_evaluate(trainer, "eval") if len(dataset_bundle.validation_dataset) else {}
        test_metrics = (
            self._safe_evaluate(trainer, "test", dataset_bundle.test_dataset)
            if len(dataset_bundle.test_dataset)
            else {}
        )
        trainer.save_model(str(self.dpo_adapter_dir))
        tokenizer.save_pretrained(str(self.dpo_adapter_dir))

        metrics = {
            "train": dict(train_output.metrics or {}),
            "validation": eval_metrics,
            "test": test_metrics,
        }
        (self.dpo_output_dir / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._cleanup(model, trainer)
        return {
            "status": "completed",
            "records": dataset_bundle.summary["split_counts"],
            "dataset_summary": dataset_bundle.summary,
            "adapter_dir": str(self.dpo_adapter_dir),
            "metrics": metrics,
        }

    @staticmethod
    def _safe_evaluate(trainer, metric_key_prefix: str, eval_dataset=None) -> dict:
        try:
            metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
            return dict(metrics or {})
        except Exception as exc:
            return {"status": "evaluation_failed", "error": str(exc)}

    @staticmethod
    def _cleanup(model=None, trainer=None) -> None:
        del trainer
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
