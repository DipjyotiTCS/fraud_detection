from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class LLMDownloadConfig:
    model_id: str
    output_dir: str
    revision: str = "main"
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    token_env_var: str = "HF_TOKEN"


class LLMModelDownloader:
    """Download a complete Hugging Face model snapshot into a project directory."""

    def __init__(self, config: LLMDownloadConfig):
        self.config = config

    def download(self) -> dict:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "huggingface_hub is not installed. Run local_setup.sh or install requirements-llm.txt."
            ) from exc

        output_path = Path(self.config.output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        token = os.getenv(self.config.token_env_var)
        downloaded_path = snapshot_download(
            repo_id=self.config.model_id,
            revision=self.config.revision,
            local_dir=str(output_path),
            token=token,
            allow_patterns=self.config.allow_patterns,
            ignore_patterns=self.config.ignore_patterns,
        )

        metadata = {
            **asdict(self.config),
            "downloaded_path": str(downloaded_path),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "hf_token_present": bool(token),
        }
        metadata_path = output_path / "download_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "local_path": str(output_path),
            "metadata_path": str(metadata_path),
        }
