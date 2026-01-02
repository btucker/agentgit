"""Claude Code AI enhancement plugin for agentgit."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agentgit.core import OperationType
from agentgit.plugins import hookimpl

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt, PromptResponse

logger = logging.getLogger(__name__)

# Default model for commit message generation (fast and cheap)
DEFAULT_MODEL = "haiku"

# Plugin identifier
ENHANCER_NAME = "claude_code"

# Global cache for batch-processed messages
_message_cache: dict[str, str] = {}

# Cached model instance
_model_cache: dict[str, object] = {}


def _get_model(model: str = DEFAULT_MODEL):
    """Get or create a ClaudeCode model instance.

    Args:
        model: The Claude model to use (e.g., "haiku", "sonnet").

    Returns:
        ClaudeCode model instance, or None if llm-claude-cli is not installed.
    """
    if model in _model_cache:
        return _model_cache[model]

    try:
        from llm_claude_cli import ClaudeCode

        instance = ClaudeCode(f"claude-code-{model}", claude_model=model)
        _model_cache[model] = instance
        return instance
    except ImportError:
        logger.warning(
            "llm-claude-cli not installed. Install with: pip install 'agentgit[llm]'"
        )
        return None
    except Exception as e:
        logger.warning("Failed to initialize Claude model: %s", e)
        return None


def _run_claude(prompt: str, model: str = DEFAULT_MODEL) -> str | None:
    """Run a prompt through Claude and return the response text.

    Args:
        prompt: The prompt to send to Claude.
        model: The model to use (e.g., "haiku", "sonnet").

    Returns:
        The response text, or None if the request fails.
    """
    claude_model = _get_model(model)
    if claude_model is None:
        return None

    try:
        response = claude_model.prompt(prompt, stream=False)
        return response.text()
    except Exception as e:
        logger.warning("Claude request failed: %s", e)
        return None


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _build_operation_context(operation: "FileOperation", max_length: int = 2000) -> str:
    """Build context string for a single file operation."""
    parts = []

    # Operation type and file
    op_name = {
        OperationType.WRITE: "Created",
        OperationType.EDIT: "Modified",
        OperationType.DELETE: "Deleted",
    }.get(operation.operation_type, "Changed")
    parts.append(f"File: {operation.file_path} ({op_name})")

    # For edits, show the change
    if operation.operation_type == OperationType.EDIT:
        if operation.old_string and operation.new_string:
            change = f"Changed:\n- {_truncate_text(operation.old_string, 200)}\n+ {_truncate_text(operation.new_string, 200)}"
            parts.append(change)

    # For writes, show a preview of content
    if operation.operation_type == OperationType.WRITE and operation.content:
        preview = _truncate_text(operation.content, 500)
        parts.append(f"Content preview:\n{preview}")

    return "\n".join(parts)


def _build_turn_context(turn: "AssistantTurn", max_length: int = 3000) -> str:
    """Build context string for an assistant turn."""
    parts = []

    # List files changed
    if turn.files_created:
        parts.append(f"Created: {', '.join(turn.files_created)}")
    if turn.files_modified:
        parts.append(f"Modified: {', '.join(turn.files_modified)}")
    if turn.files_deleted:
        parts.append(f"Deleted: {', '.join(turn.files_deleted)}")

    # Include assistant reasoning if available
    if turn.context and turn.context.summary:
        reasoning = _truncate_text(turn.context.summary, 1000)
        parts.append(f"\nAssistant reasoning:\n{reasoning}")

    # Include details of operations
    for op in turn.operations[:5]:  # Limit to first 5 operations
        parts.append(f"\n{_build_operation_context(op, 400)}")

    if len(turn.operations) > 5:
        parts.append(f"\n... and {len(turn.operations) - 5} more operations")

    return _truncate_text("\n".join(parts), max_length)


def _clean_message(message: str) -> str:
    """Clean up an AI-generated commit message."""
    message = message.strip().strip('"').strip("'")
    if len(message) > 72:
        message = message[:69] + "..."
    return message


def _get_prompt_key(prompt: "Prompt") -> str:
    """Generate a cache key for a prompt."""
    return f"prompt:{prompt.prompt_id}"


def _get_turn_key(turn: "AssistantTurn") -> str:
    """Generate a cache key for a turn."""
    # Use timestamp and first operation tool_id for uniqueness
    if turn.operations and turn.operations[0].tool_id:
        return f"turn:{turn.timestamp}:{turn.operations[0].tool_id}"
    return f"turn:{turn.timestamp}"


def batch_enhance_prompt_responses(
    prompt_responses: list["PromptResponse"],
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Batch process all prompt responses to generate commit messages efficiently.

    This function sends all prompts to Claude in a single call, which is
    much more efficient than making individual calls for each commit message.

    Args:
        prompt_responses: List of PromptResponse objects to process.
        model: The model to use (e.g., "haiku", "sonnet").

    Returns:
        Dictionary mapping cache keys to generated commit messages.
    """
    global _message_cache

    if not prompt_responses:
        return {}

    # Build batch prompt with all items
    items = []
    item_keys = []

    for pr in prompt_responses:
        # Add prompt/merge message request
        prompt_key = _get_prompt_key(pr.prompt)
        if prompt_key not in _message_cache:
            context_parts = [f"User request: {_truncate_text(pr.prompt.text, 500)}"]

            # Summarize files changed
            all_files = []
            for turn in pr.turns:
                all_files.extend(turn.files_created)
                all_files.extend(turn.files_modified)

            if all_files:
                context_parts.append(f"Files changed: {', '.join(all_files[:10])}")

            items.append({
                "id": len(items) + 1,
                "type": "merge",
                "context": "\n".join(context_parts),
            })
            item_keys.append(prompt_key)

        # Add turn message requests
        for turn in pr.turns:
            turn_key = _get_turn_key(turn)
            if turn_key not in _message_cache:
                items.append({
                    "id": len(items) + 1,
                    "type": "turn",
                    "context": _build_turn_context(turn, 800),
                })
                item_keys.append(turn_key)

    if not items:
        return _message_cache

    # Build the batch prompt
    batch_prompt = """Generate concise git commit message subject lines (max 72 characters each) for these items.

Rules:
- Start with a verb (Add, Fix, Update, Remove, Refactor, Implement)
- Be specific but concise
- Focus on the purpose/intent
- Use imperative mood ("Add feature" not "Added feature")

Items:
"""

    for item in items:
        batch_prompt += f"\n[{item['id']}] ({item['type']})\n{item['context']}\n"

    batch_prompt += f"""
Respond with a JSON object mapping item IDs to commit messages:
{{"1": "Add user authentication", "2": "Fix login validation", ...}}

ONLY respond with the JSON object, nothing else."""

    # Call Claude once for all items
    response = _run_claude(batch_prompt, model)

    if response:
        try:
            # Try to parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            messages = json.loads(response)

            # Map responses back to cache keys
            for i, key in enumerate(item_keys):
                item_id = str(i + 1)
                if item_id in messages:
                    _message_cache[key] = _clean_message(messages[item_id])

        except json.JSONDecodeError as e:
            logger.debug("Failed to parse batch response as JSON: %s", e)
            # Fall back to individual processing

    return _message_cache


def clear_message_cache() -> None:
    """Clear the message cache."""
    global _message_cache
    _message_cache = {}


class ClaudeCodeEnhancerPlugin:
    """AI enhancement plugin using Claude Code via llm-claude-cli."""

    @hookimpl
    def agentgit_get_ai_enhancer_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": ENHANCER_NAME,
            "description": "Use Claude Code to generate commit messages (requires llm-claude-cli)",
        }

    @hookimpl
    def agentgit_enhance_operation_message(
        self,
        operation: "FileOperation",
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an AI-enhanced commit message for a file operation."""
        if enhancer != ENHANCER_NAME:
            return None

        model = model or DEFAULT_MODEL

        # Build context for the prompt
        context_parts = []
        context_parts.append(_build_operation_context(operation))

        if operation.prompt:
            prompt_text = _truncate_text(operation.prompt.text, 500)
            context_parts.append(f"\nUser request:\n{prompt_text}")

        if operation.assistant_context and operation.assistant_context.summary:
            reasoning = _truncate_text(operation.assistant_context.summary, 500)
            context_parts.append(f"\nAssistant reasoning:\n{reasoning}")

        context = "\n".join(context_parts)

        prompt = f"""Generate a concise git commit message subject line (max 72 characters) for this file operation.

The message should:
- Start with a verb (Add, Fix, Update, Remove, Refactor, etc.)
- Be specific about what changed and why
- Focus on the purpose/intent, not just the mechanical change
- Be in imperative mood (e.g., "Add feature" not "Added feature")

Context:
{context}

Respond with ONLY the commit message subject line, nothing else."""

        message = _run_claude(prompt, model)
        if message:
            return _clean_message(message)
        return None

    @hookimpl
    def agentgit_enhance_turn_message(
        self,
        turn: "AssistantTurn",
        prompt: "Prompt | None",
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an AI-enhanced commit message for an assistant turn."""
        if enhancer != ENHANCER_NAME:
            return None

        # Check cache first (populated by batch processing)
        turn_key = _get_turn_key(turn)
        if turn_key in _message_cache:
            return _message_cache[turn_key]

        model = model or DEFAULT_MODEL

        # Build context
        context_parts = []
        context_parts.append(_build_turn_context(turn))

        if prompt:
            prompt_text = _truncate_text(prompt.text, 500)
            context_parts.append(f"\nUser request:\n{prompt_text}")

        context = "\n".join(context_parts)

        ai_prompt = f"""Generate a concise git commit message subject line (max 72 characters) for this set of file changes.

The message should:
- Start with a verb (Add, Fix, Update, Remove, Refactor, Implement, etc.)
- Summarize the overall purpose of all the changes together
- Focus on the intent/goal, not list individual files
- Be in imperative mood (e.g., "Add feature" not "Added feature")

Context:
{context}

Respond with ONLY the commit message subject line, nothing else."""

        message = _run_claude(ai_prompt, model)
        if message:
            result = _clean_message(message)
            _message_cache[turn_key] = result
            return result
        return None

    @hookimpl
    def agentgit_enhance_merge_message(
        self,
        prompt: "Prompt",
        turns: list["AssistantTurn"],
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an AI-enhanced merge commit message for a user prompt."""
        if enhancer != ENHANCER_NAME:
            return None

        # Check cache first (populated by batch processing)
        prompt_key = _get_prompt_key(prompt)
        if prompt_key in _message_cache:
            return _message_cache[prompt_key]

        model = model or DEFAULT_MODEL

        # Build context
        context_parts = []
        context_parts.append(f"User request:\n{_truncate_text(prompt.text, 1000)}")

        # Summarize what was done
        all_created = []
        all_modified = []
        all_deleted = []
        for turn in turns:
            all_created.extend(turn.files_created)
            all_modified.extend(turn.files_modified)
            all_deleted.extend(turn.files_deleted)

        if all_created:
            context_parts.append(f"Files created: {', '.join(all_created[:10])}")
        if all_modified:
            context_parts.append(f"Files modified: {', '.join(all_modified[:10])}")
        if all_deleted:
            context_parts.append(f"Files deleted: {', '.join(all_deleted[:10])}")

        # Include some assistant reasoning
        for turn in turns[:3]:
            if turn.context and turn.context.summary:
                reasoning = _truncate_text(turn.context.summary, 300)
                context_parts.append(f"\nAssistant work:\n{reasoning}")

        context = "\n".join(context_parts)

        ai_prompt = f"""Generate a concise git commit message subject line (max 72 characters) that summarizes the work done in response to this user request.

The message should:
- Start with a verb (Add, Fix, Update, Remove, Refactor, Implement, etc.)
- Capture the overall goal/outcome of the work
- Be specific but concise
- Be in imperative mood (e.g., "Add feature" not "Added feature")

Context:
{context}

Respond with ONLY the commit message subject line, nothing else."""

        message = _run_claude(ai_prompt, model)
        if message:
            result = _clean_message(message)
            _message_cache[prompt_key] = result
            return result
        return None
