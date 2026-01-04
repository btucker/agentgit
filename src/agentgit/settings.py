"""Global settings management for agentgit.

This module handles the global config file at ~/.agentgit/config.yml.
For project-specific configuration, see config.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Global config file path
CONFIG_PATH = Path.home() / ".agentgit" / "config.yml"


def get_config() -> dict[str, Any]:
    """Get the current agentgit configuration.

    Config file format (~/.agentgit/config.yml):
    ```yaml
    plugins:
      packages:
        - agentgit-aider
        - agentgit-cursor
    ```

    Returns:
        The config dict, or empty dict if not exists.
    """
    if not CONFIG_PATH.exists():
        return {}

    try:
        content = CONFIG_PATH.read_text()
        return yaml.safe_load(content) or {}
    except (yaml.YAMLError, OSError):
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save the agentgit configuration.

    Args:
        config: The config dict to save.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
