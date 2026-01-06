"""Enhancement configuration and utilities for agentgit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentgit.plugins import get_configured_plugin_manager

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt, PromptResponse

# Default enhancer to use (rules = no AI, claude_code = AI-powered)
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
    for info in pm.hook.agentgit_get_enhancer_info():
        if info:
            enhancers.append(info)
    return enhancers


def generate_turn_summary(
    turn: "AssistantTurn",
    prompt: "Prompt | None" = None,
    config: EnhanceConfig | None = None,
) -> str | None:
    """Generate an enhanced summary for an assistant turn entry.

    Args:
        turn: The assistant turn containing grouped operations.
        prompt: Optional user prompt that triggered this turn.
        config: Optional configuration for enhancement.

    Returns:
        Generated entry summary, or None if generation fails.
    """
    import logging
    logger = logging.getLogger(__name__)

    if config is None:
        config = EnhanceConfig()

    if not config.enabled:
        logger.debug("generate_turn_summary: enhancement disabled")
        return None

    pm = get_configured_plugin_manager()
    result = pm.hook.agentgit_enhance_turn_summary(
        turn=turn,
        prompt=prompt,
        enhancer=config.enhancer,
        model=config.model,
    )
    logger.debug("generate_turn_summary: enhancer=%s returned %s", config.enhancer, result)
    return result


def generate_prompt_summary(
    prompt: "Prompt",
    turns: list["AssistantTurn"],
    config: EnhanceConfig | None = None,
) -> str | None:
    """Generate an enhanced summary for a user prompt entry.

    Args:
        prompt: The user prompt.
        turns: All assistant turns that responded to the prompt.
        config: Optional configuration for enhancement.

    Returns:
        Generated entry summary, or None if generation fails.
    """
    if config is None:
        config = EnhanceConfig()

    if not config.enabled:
        return None

    pm = get_configured_plugin_manager()
    return pm.hook.agentgit_enhance_prompt_summary(
        prompt=prompt,
        turns=turns,
        enhancer=config.enhancer,
        model=config.model,
    )


def curate_turn_context(
    turn: "AssistantTurn",
    config: EnhanceConfig | None = None,
) -> str | None:
    """Curate the context/reasoning to include in a turn commit body.

    The enhancer can select and organize the most relevant parts of the
    assistant's thinking to include in the commit message body.

    Args:
        turn: The assistant turn to curate context for.
        config: Optional configuration for enhancement.

    Returns:
        Curated context string to include in commit body, or None to use default.
    """
    if config is None:
        config = EnhanceConfig()

    if not config.enabled:
        return None

    pm = get_configured_plugin_manager()
    return pm.hook.agentgit_curate_turn_context(
        turn=turn,
        enhancer=config.enhancer,
        model=config.model,
    )


def generate_session_branch_name(
    prompt_responses: list["PromptResponse"],
    session_id: str | None = None,
    config: EnhanceConfig | None = None,
    agent_name: str | None = None,
) -> str:
    """Generate a descriptive branch name for a coding session.

    Args:
        prompt_responses: All prompts and operations in the session.
        session_id: Optional session identifier for fallback naming.
        config: Optional configuration for enhancement.
        agent_name: Optional agent/format name (e.g., 'claude-code', 'codex').

    Returns:
        A git-safe branch name like 'sessions/claude-code/260106-add-user-authentication'
    """
    import re
    import logging
    from datetime import datetime
    logger = logging.getLogger(__name__)

    if config is None:
        config = EnhanceConfig()

    # Sanitize agent name for branch path
    if agent_name:
        agent_part = re.sub(r'[^\w\-]', '-', agent_name.lower())
        agent_part = re.sub(r'-+', '-', agent_part).strip('-')
    else:
        agent_part = "unknown"

    # Extract creation date from first prompt
    date_prefix = ""
    if prompt_responses and prompt_responses[0].prompt:
        try:
            # Parse ISO timestamp and format as YYMMDD
            timestamp_str = prompt_responses[0].prompt.timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            date_prefix = dt.strftime('%y%m%d')
        except Exception as e:
            logger.warning("Failed to parse timestamp: %s", e)

    # Try AI-generated name if configured
    if config.enabled and config.enhancer == "llm":
        pm = get_configured_plugin_manager()
        result = pm.hook.agentgit_generate_session_name(
            prompt_responses=prompt_responses,
            enhancer=config.enhancer,
            model=config.model,
        )
        if result:
            # Sanitize for git branch name
            safe_name = re.sub(r'[^\w\-/]', '-', result.lower())
            safe_name = re.sub(r'-+', '-', safe_name).strip('-')
            logger.info("Generated session name: %s", safe_name)
            if date_prefix:
                return f"sessions/{agent_part}/{date_prefix}-{safe_name}"
            return f"sessions/{agent_part}/{safe_name}"

    # Fallback: use first prompt as basis
    if prompt_responses and prompt_responses[0].prompt:
        first_prompt = prompt_responses[0].prompt.text
        # Take first line, sanitize, truncate
        first_line = first_prompt.split('\n')[0].strip()
        safe_name = re.sub(r'[^\w\-]', '-', first_line.lower())
        safe_name = re.sub(r'-+', '-', safe_name).strip('-')[:50]
        if date_prefix:
            return f"sessions/{agent_part}/{date_prefix}-{safe_name}"
        return f"sessions/{agent_part}/{safe_name}"

    # Last resort: use session ID or timestamp
    if session_id:
        if date_prefix:
            return f"sessions/{agent_part}/{date_prefix}-{session_id[:12]}"
        return f"sessions/{agent_part}/{session_id[:12]}"

    import time
    return f"sessions/{agent_part}/{int(time.time())}"


def preprocess_batch_enhancement(
    prompt_responses: list["PromptResponse"],
    config: EnhanceConfig | None = None,
) -> None:
    """Pre-process all prompt responses for batch enhancement.

    For AI-powered enhancers like claude_code, this sends all items to the
    AI in a single call, which is much more efficient than individual calls.
    The results are cached and returned by subsequent individual hook calls.

    For non-AI enhancers like rules, this is a no-op.

    Args:
        prompt_responses: List of PromptResponse objects to process.
        config: Optional configuration for enhancement.
    """
    import logging
    logger = logging.getLogger(__name__)

    if config is None:
        config = EnhanceConfig()

    logger.info("preprocess_batch_enhancement: enabled=%s, enhancer=%s, model=%s, num_responses=%d",
                config.enabled, config.enhancer, config.model, len(prompt_responses))

    if not config.enabled:
        return

    # Only do batch processing for llm enhancer
    if config.enhancer == "llm":
        from agentgit.enhancers.llm import batch_enhance_prompt_responses

        result = batch_enhance_prompt_responses(prompt_responses, config.model)
        logger.info("Batch processing completed, cache size: %d", len(result))
