"""Configuration helpers for real hardware connectivity."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_FILE = "hardware_config.json"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(_DEFAULT_CONFIG_FILE)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("Hardware config file not found: %s", path)
        return {}
    if not text.strip():
        logger.warning("Hardware config file is empty: %s", path)
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in hardware config {path}: {exc}") from exc


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the connector configuration file.

    Parameters
    ----------
    path:
        Optional path to a JSON configuration file. When omitted, the default
        file ``hardware_config.json`` in the connector directory is used.
    """

    cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    return _read_json(cfg_path)


def load_real_hardware_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return the ``real_hardware`` section of the connector configuration."""

    config = load_config(path)
    real_section = config.get("real_hardware", {})
    if real_section is None:
        return {}
    if not isinstance(real_section, Mapping):
        raise ValueError("`real_hardware` section must be an object in the config file")
    return dict(real_section)


def is_real_hardware_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """Check if the real hardware backend is marked as enabled."""

    if config is None:
        config = load_real_hardware_config()
    return bool(config.get("enabled"))


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "is_real_hardware_enabled",
    "load_config",
    "load_real_hardware_config",
]
