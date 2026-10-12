"""
core/config_loader.py

Loads settings.yaml and merges .env secrets.
All modules call get_config() to access the unified config dict.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Locate project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_FILE = _PROJECT_ROOT / "settings.yaml"
_ENV_FILE = _PROJECT_ROOT / ".env"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """
    Returns the merged configuration dictionary.
    Cached after first load — restart the process to reload.
    """
    # 1. Load .env file (silently ignored if missing — use .env.example as guide)
    load_dotenv(_ENV_FILE, override=False)

    # 2. Parse settings.yaml
    if not _SETTINGS_FILE.exists():
        raise FileNotFoundError(
            f"[ConfigLoader] settings.yaml not found at {_SETTINGS_FILE}. "
            "Copy settings.yaml.example or create it from the template."
        )

    with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    # 3. Inject secrets from environment variables into the config tree
    #    so callers never need to touch os.environ directly.
    _inject_env(config)

    return config


def _inject_env(config: dict[str, Any]) -> None:
    """Mutates config in-place to overlay secrets from environment variables."""

    # Reddit credentials
    reddit_cfg = config.setdefault("sentinel", {}).setdefault("reddit", {})
    if os.getenv("REDDIT_CLIENT_ID"):
        reddit_cfg["client_id"] = os.environ["REDDIT_CLIENT_ID"]
    if os.getenv("REDDIT_CLIENT_SECRET"):
        reddit_cfg["client_secret"] = os.environ["REDDIT_CLIENT_SECRET"]
    if os.getenv("REDDIT_USER_AGENT"):
        reddit_cfg["user_agent"] = os.environ["REDDIT_USER_AGENT"]

    # ElevenLabs key
    factory_cfg = config.setdefault("factory", {})
    if os.getenv("ELEVENLABS_API_KEY"):
        factory_cfg["elevenlabs_api_key"] = os.environ["ELEVENLABS_API_KEY"]

    # YouTube key (optional, future use)
    if os.getenv("YOUTUBE_API_KEY"):
        factory_cfg["youtube_api_key"] = os.environ["YOUTUBE_API_KEY"]


def get(key_path: str, default: Any = None) -> Any:
    """
    Dot-notation accessor into the config tree.
    Example: get("sentinel.reddit.min_score", 500)
    """
    keys = key_path.split(".")
    node: Any = get_config()
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node
