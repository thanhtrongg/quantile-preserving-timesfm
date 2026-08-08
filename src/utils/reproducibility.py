"""Reproducibility and machine/runtime metadata."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit(repo: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(
    *,
    timesfm_metadata: Mapping[str, Any],
    gift_eval_version: str = "0.0.0a0",
    gift_eval_commit: str = "d8184bb51079bb5021332f8e5d7486c378a52202",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "pytorch_version": None,
        "cuda_version": None,
        "gpu_model": None,
        "timesfm": dict(timesfm_metadata),
        "gift_eval_version": gift_eval_version,
        "gift_eval_commit": gift_eval_commit,
    }
    try:
        import torch

        metadata["pytorch_version"] = torch.__version__
        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_model"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return metadata


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
