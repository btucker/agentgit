"""Enhancement configuration and utilities for agentgit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentgit.plugins import get_configured_plugin_manager

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt

# Default enhancer to use (rules = no AI, claude_cli = AI-powered)
DEFAULT_ENHANCER = "rules"
DEFAULT_MODEL = "haiku"


@dataclass
class EnhanceConfig:
    """Configuration for enhanced commit message generation."""

    enhancer: str = DEFAULT_ENHANCER
    model: str = DEFAULT_MODEL
    enabled: bool = True


def get_available_enhancers() -> list[dict[str, str]]:
    """Get list of available enhancer plugins.

    Returns:
        List of dicts with 'name' and 'description' keys.
    """
    pm = get_configured_plugin_manager()
    enhancers = []
    for info in pm.hook.agentgit_get_ai_enhancer_info():
        if info:
            enhancers.append(info)
    return enhancers


def generate_operation_commit_message(
    operation: "FileOperation",
    config: EnhanceConfig | None = None,
) -> str | None:
    """Generate an enhanced commit message for a single file operation.

    Args:
        operation: The file operation to generate a message for.
        config: Optional configuration for enhancement.

    Returns:
        Generated commit message subject line, or None if generation fails.
    """
    if config is None:
        config = EnhanceConfig()

    if not config.enabled:
        return None

    pm = get_configured_plugin_manager()
    return pm.hook.agentgit_enhance_operation_message(
        operation=operation,
        enhancer=config.enhancer,
        model=config.model,
    )


def generate_turn_commit_message(
    turn: "AssistantTurn",
    prompt: "Prompt | None" = None,
    config: EnhanceConfig | None = None,
) -> str | None:
    """Generate an enhanced commit message for an assistant turn.

    Args:
        turn: The assistant turn containing grouped operations.
        prompt: Optional user prompt that triggered this turn.
        config: Optional configuration for enhancement.

    Returns:
        Generated commit message subject line, or None if generation fails.
    """
    if config is None:
        config = EnhanceConfig()

    if not config.enabled:
        return None

    pm = get_configured_plugin_manager()
    return pm.hook.agentgit_enhance_turn_message(
        turn=turn,
        prompt=prompt,
        enhancer=config.enhancer,
        model=config.model,
    )


def generate_merge_commit_message(
    prompt: "Prompt",
    turns: list["AssistantTurn"],
    config: EnhanceConfig | None = None,
) -> str | None:
    """Generate an enhanced merge commit message for a user prompt.

    Args:
        prompt: The user prompt.
        turns: All assistant turns that responded to the prompt.
        config: Optional configuration for enhancement.

    Returns:
        Generated commit message subject line, or None if generation fails.
    """
    if config is None:
        config = EnhanceConfig()

    if not config.enabled:
        return None

    pm = get_configured_plugin_manager()
    return pm.hook.agentgit_enhance_merge_message(
        prompt=prompt,
        turns=turns,
        enhancer=config.enhancer,
        model=config.model,
    )
