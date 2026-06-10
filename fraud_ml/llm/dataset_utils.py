from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")
            yield value


def _read_json(path: Path) -> Iterable[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at index {index} in {path}")
            yield item
        return
    raise ValueError(f"Unsupported JSON dataset shape in {path}")


def load_records_from_folder(folder: str) -> List[Dict[str, Any]]:
    dataset_dir = Path(folder).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    files = sorted([*dataset_dir.glob("*.jsonl"), *dataset_dir.glob("*.json")])
    if not files:
        raise FileNotFoundError(
            f"No .jsonl or .json dataset files found in {dataset_dir}. "
            "Add SFT/DPO data before starting fine-tuning."
        )

    records: List[Dict[str, Any]] = []
    for path in files:
        if path.suffix == ".jsonl":
            records.extend(_read_jsonl(path))
        elif path.suffix == ".json":
            records.extend(_read_json(path))

    if not records:
        raise ValueError(f"No dataset records were loaded from {dataset_dir}")
    return records


def normalize_sft_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert common enterprise instruction dataset shapes to TRL SFT format."""
    if "messages" in record:
        return {"messages": record["messages"]}
    if "prompt" in record and "completion" in record:
        return {"prompt": record["prompt"], "completion": record["completion"]}
    if "text" in record:
        return {"text": record["text"]}
    if "instruction" in record and "output" in record:
        user_content = record["instruction"]
        if record.get("input"):
            user_content = f"{user_content}\n\nInput:\n{record['input']}"
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": record["output"]},
            ]
        }
    raise ValueError(
        "Unsupported SFT record. Use one of: messages, prompt+completion, text, or instruction+output."
    )


def normalize_dpo_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert common preference dataset shapes to TRL DPO format."""
    if {"prompt", "chosen", "rejected"}.issubset(record.keys()):
        return {
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }
    if {"instruction", "chosen", "rejected"}.issubset(record.keys()):
        user_content = record["instruction"]
        if record.get("input"):
            user_content = f"{user_content}\n\nInput:\n{record['input']}"
        return {
            "prompt": [{"role": "user", "content": user_content}],
            "chosen": [{"role": "assistant", "content": record["chosen"]}],
            "rejected": [{"role": "assistant", "content": record["rejected"]}],
        }
    raise ValueError("Unsupported DPO record. Use prompt+chosen+rejected or instruction+chosen+rejected.")


def load_sft_dataset(folder: str):
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("datasets is not installed. Install requirements.txt first.") from exc

    records = [normalize_sft_record(record) for record in load_records_from_folder(folder)]
    return Dataset.from_list(records)


def load_dpo_dataset(folder: str):
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("datasets is not installed. Install requirements.txt first.") from exc

    records = [normalize_dpo_record(record) for record in load_records_from_folder(folder)]
    return Dataset.from_list(records)
