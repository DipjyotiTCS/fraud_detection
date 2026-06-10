from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

SUPPORTED_DATASET_SUFFIXES = {".jsonl", ".json"}
SplitName = Literal["train", "validation", "test"]


@dataclass
class DatasetSplitBundle:
    train_dataset: Any
    validation_dataset: Any
    test_dataset: Any
    summary: Dict[str, Any]


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


def _dataset_files(folder: str, recursive: bool = True) -> List[Path]:
    dataset_dir = Path(folder).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in dataset_dir.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(dataset_dir).parts)
    ]
    return sorted(files)


def dataset_folder_has_supported_files(folder: str, recursive: bool = True) -> bool:
    try:
        return bool(_dataset_files(folder, recursive=recursive))
    except FileNotFoundError:
        return False


def _read_records_from_file(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(_read_jsonl(path))
    if path.suffix.lower() == ".json":
        return list(_read_json(path))
    return []


def load_records_from_folder(folder: str, recursive: bool = True) -> List[Dict[str, Any]]:
    files = _dataset_files(folder, recursive=recursive)
    if not files:
        raise FileNotFoundError(
            f"No .jsonl or .json dataset files found in {Path(folder).expanduser().resolve()}. "
            "Add SFT/DPO data before starting fine-tuning."
        )

    records: List[Dict[str, Any]] = []
    for path in files:
        records.extend(_read_records_from_file(path))

    if not records:
        raise ValueError(f"No dataset records were loaded from {Path(folder).expanduser().resolve()}")
    return records


def _infer_split_from_file(path: Path, root: Path) -> Optional[SplitName]:
    """Infer split from filename first, then from parent folder names.

    Filename priority avoids treating folders such as `sft_test_dataset/train.jsonl`
    as test data just because the parent folder contains the word `test`.
    """
    stem = path.stem.lower()
    filename_tokens = stem.replace("-", "_").split("_")

    def token_match(tokens: Sequence[str]) -> Optional[SplitName]:
        token_set = set(tokens)
        if token_set & {"train", "training"}:
            return "train"
        if token_set & {"val", "valid", "validation", "eval", "evaluation", "dev"}:
            return "validation"
        if token_set & {"test", "testing", "holdout"}:
            return "test"
        return None

    split = token_match(filename_tokens)
    if split:
        return split

    relative_parts = list(path.relative_to(root).parts[:-1])
    for part in reversed(relative_parts):
        tokens = part.lower().replace("-", "_").split("_")
        split = token_match(tokens)
        if split:
            return split
    return None


def _load_records_by_explicit_split(folder: str, recursive: bool = True) -> Tuple[Dict[SplitName, List[Dict[str, Any]]], List[str]]:
    root = Path(folder).expanduser().resolve()
    files = _dataset_files(folder, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No .jsonl or .json dataset files found in {root}")

    split_records: Dict[SplitName, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    unassigned: List[Dict[str, Any]] = []
    file_summaries: List[str] = []

    for path in files:
        records = _read_records_from_file(path)
        split = _infer_split_from_file(path, root)
        file_summaries.append(f"{path.relative_to(root)}:{split or 'unspecified'}:{len(records)}")
        if split:
            split_records[split].extend(records)
        else:
            unassigned.extend(records)

    if any(split_records.values()):
        split_records["train"].extend(unassigned)
    else:
        split_records["train"] = unassigned
    return split_records, file_summaries


def _stratify_key(record: Dict[str, Any]) -> str:
    task = record.get("task")
    typology = record.get("typology")
    if task and typology:
        return f"{task}::{typology}"
    if task:
        return str(task)
    if typology:
        return str(typology)
    return "__all__"


def _split_records(
    records: List[Dict[str, Any]],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[SplitName, List[Dict[str, Any]]]:
    if not records:
        return {"train": [], "validation": [], "test": []}
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be >= 0 and sum to less than 1.0")

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_stratify_key(record)].append(record)

    output: Dict[SplitName, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for key in sorted(grouped.keys()):
        group = grouped[key]
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * validation_ratio))

        # Keep at least one training row for each non-empty stratum.
        if n_test + n_val >= n:
            overflow = n_test + n_val - (n - 1)
            reduce_test = min(n_test, overflow)
            n_test -= reduce_test
            overflow -= reduce_test
            if overflow > 0:
                n_val = max(0, n_val - overflow)

        output["test"].extend(group[:n_test])
        output["validation"].extend(group[n_test:n_test + n_val])
        output["train"].extend(group[n_test + n_val:])

    for split_name in output:
        rng.shuffle(output[split_name])
    return output


def _ensure_three_splits(
    split_records: Dict[SplitName, List[Dict[str, Any]]],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[SplitName, List[Dict[str, Any]]]:
    train_records = split_records.get("train", [])
    validation_records = split_records.get("validation", [])
    test_records = split_records.get("test", [])

    if not train_records and (validation_records or test_records):
        raise ValueError("No training records were found. Provide train.jsonl or an unsplit SFT/DPO dataset file.")

    # Case 1: only unsplit/training data exists. Split into train/validation/test.
    if train_records and not validation_records and not test_records:
        return _split_records(train_records, validation_ratio=validation_ratio, test_ratio=test_ratio, seed=seed)

    # Case 2: explicit validation exists but test is missing. Keep validation; carve test from train.
    if train_records and validation_records and not test_records and test_ratio > 0:
        train_and_test = _split_records(train_records, validation_ratio=0.0, test_ratio=test_ratio, seed=seed)
        return {
            "train": train_and_test["train"],
            "validation": validation_records,
            "test": train_and_test["test"],
        }

    # Case 3: explicit test exists but validation is missing. Keep test; carve validation from train.
    if train_records and test_records and not validation_records and validation_ratio > 0:
        train_and_val = _split_records(train_records, validation_ratio=validation_ratio, test_ratio=0.0, seed=seed)
        return {
            "train": train_and_val["train"],
            "validation": train_and_val["validation"],
            "test": test_records,
        }

    return {"train": train_records, "validation": validation_records, "test": test_records}


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    roles = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role must be system, user, or assistant")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be a non-empty string")
        roles.append(role)
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("messages must contain at least one user message and one assistant message")


def normalize_sft_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert common enterprise instruction dataset shapes to TRL SFT format.

    Supported SFT shapes:
    - {"text": "<s>[INST] ... [/INST] answer</s>"}
    - {"messages": [{"role":"system|user|assistant", "content":"..."}, ...]}
    - {"prompt": "...", "completion": "..."}
    - {"instruction": "...", "input": "...", "output": "..."}

    If both text and messages are present, text is preferred because many exported
    Mistral datasets already contain the exact serialized chat template.
    """
    if isinstance(record.get("text"), str) and record["text"].strip():
        return {"text": record["text"]}
    if "messages" in record:
        _validate_messages(record["messages"])
        return {"messages": record["messages"]}
    if "prompt" in record and "completion" in record:
        return {"prompt": record["prompt"], "completion": record["completion"]}
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
        "Unsupported SFT record. Use one of: text, messages, prompt+completion, or instruction+output."
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


def _records_summary(records_by_split: Dict[SplitName, List[Dict[str, Any]]], file_summaries: List[str]) -> Dict[str, Any]:
    all_records = [record for records in records_by_split.values() for record in records]
    schema_keys = Counter()
    task_counts = Counter()
    typology_counts = Counter()
    format_counts = Counter()
    for record in all_records:
        schema_keys.update(record.keys())
        if record.get("task"):
            task_counts[str(record["task"])] += 1
        if record.get("typology"):
            typology_counts[str(record["typology"])] += 1
        if isinstance(record.get("text"), str) and record["text"].strip():
            format_counts["text"] += 1
        if record.get("messages"):
            format_counts["messages"] += 1
        if "prompt" in record and "completion" in record:
            format_counts["prompt_completion"] += 1
        if "instruction" in record and "output" in record:
            format_counts["instruction_output"] += 1

    return {
        "total_records": len(all_records),
        "split_counts": {name: len(records) for name, records in records_by_split.items()},
        "source_files": file_summaries,
        "schema_keys": dict(schema_keys),
        "format_counts": dict(format_counts),
        "task_counts": dict(task_counts),
        "typology_counts_top_20": dict(typology_counts.most_common(20)),
    }


def _dataset_from_records(records: List[Dict[str, Any]], normalizer):
    from datasets import Dataset

    normalized = [normalizer(record) for record in records]
    return Dataset.from_list(normalized)


def load_sft_dataset(folder: str):
    # Backward-compatible helper: returns one combined training dataset.
    return _dataset_from_records(load_records_from_folder(folder), normalize_sft_record)


def load_dpo_dataset(folder: str):
    # Backward-compatible helper: returns one combined training dataset.
    return _dataset_from_records(load_records_from_folder(folder), normalize_dpo_record)


def load_sft_dataset_splits(
    folder: str,
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplitBundle:
    try:
        from datasets import Dataset  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("datasets is not installed. Install requirements-llm.txt first.") from exc

    explicit, file_summaries = _load_records_by_explicit_split(folder, recursive=True)
    records_by_split = _ensure_three_splits(
        explicit,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if not records_by_split["train"]:
        raise ValueError("No training records are available after dataset split.")

    summary = _records_summary(records_by_split, file_summaries)
    return DatasetSplitBundle(
        train_dataset=_dataset_from_records(records_by_split["train"], normalize_sft_record),
        validation_dataset=_dataset_from_records(records_by_split["validation"], normalize_sft_record),
        test_dataset=_dataset_from_records(records_by_split["test"], normalize_sft_record),
        summary=summary,
    )


def load_dpo_dataset_splits(
    folder: str,
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplitBundle:
    try:
        from datasets import Dataset  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("datasets is not installed. Install requirements-llm.txt first.") from exc

    explicit, file_summaries = _load_records_by_explicit_split(folder, recursive=True)
    records_by_split = _ensure_three_splits(
        explicit,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if not records_by_split["train"]:
        raise ValueError("No training records are available after dataset split.")

    summary = _records_summary(records_by_split, file_summaries)
    return DatasetSplitBundle(
        train_dataset=_dataset_from_records(records_by_split["train"], normalize_dpo_record),
        validation_dataset=_dataset_from_records(records_by_split["validation"], normalize_dpo_record),
        test_dataset=_dataset_from_records(records_by_split["test"], normalize_dpo_record),
        summary=summary,
    )


def inspect_dataset_folder(
    folder: str,
    *,
    dataset_type: Literal["sft", "dpo"] = "sft",
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    explicit, file_summaries = _load_records_by_explicit_split(folder, recursive=True)
    records_by_split = _ensure_three_splits(
        explicit,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    normalizer = normalize_sft_record if dataset_type == "sft" else normalize_dpo_record
    errors: List[str] = []
    for split_name, records in records_by_split.items():
        for index, record in enumerate(records[:100]):
            try:
                normalizer(record)
            except Exception as exc:  # pragma: no cover - validation only
                errors.append(f"{split_name}[{index}]: {exc}")
                if len(errors) >= 20:
                    break
        if len(errors) >= 20:
            break

    summary = _records_summary(records_by_split, file_summaries)
    summary.update({
        "dataset_type": dataset_type,
        "supported": not errors,
        "validation_errors_sample": errors,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    })
    return summary
