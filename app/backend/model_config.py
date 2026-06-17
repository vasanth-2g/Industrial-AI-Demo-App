from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("APP_MODEL_CONFIG", APP_ROOT / "model_config.json"))


def load_model_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def section(name: str) -> dict[str, Any]:
    value = load_model_config().get(name, {})
    return value if isinstance(value, dict) else {}


def resolve_app_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return APP_ROOT / path


def config_value(section_name: str, key: str, default: Any = None) -> Any:
    return section(section_name).get(key, default)
