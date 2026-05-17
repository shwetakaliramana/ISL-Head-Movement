"""
src/utils/config.py
Loads configs/config.yaml and exposes it as a dot-accessible object.
Usage:
    from src.utils.config import cfg
    print(cfg.camera.fps)          # 30
    print(cfg.model.num_classes)   # 5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class _DotDict(dict):
    """dict subclass that allows attribute-style access: d.key instead of d['key']."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        return _DotDict(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path | None = None) -> _DotDict:
    """
    Load the YAML config file.

    Args:
        path: explicit path to config.yaml.
              Defaults to <project_root>/configs/config.yaml.
    Returns:
        _DotDict with dot-access to all config values.
    """
    if path is None:
        this_file = Path(__file__).resolve()
        project_root = this_file.parents[2]
        candidates = [
            project_root / "src" / "utils" / "config.yaml",
            project_root / "files" / "config.yaml",
            project_root / "configs" / "config.yaml",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return _DotDict(raw)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this from anywhere: `from src.utils.config import cfg`
cfg = load_config()
