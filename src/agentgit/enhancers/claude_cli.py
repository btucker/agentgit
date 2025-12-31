"""Claude CLI AI enhancement plugin for agentgit."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING

from agentgit.core import OperationType
from agentgit.plugins import hookimpl

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt

logger = logging.getLogger(__name__)

# Default model for commit message generation (fast and cheap)
DEFAULT_MODEL = "haiku"

# Plugin identifier
ENHANCER_NAME = "claude_cli"


def _run_claude_cli(prompt: str, model: str = DEFAULT_MODEL) -> str | None:
    """Run Claude CLI with a prompt and return the response text.

    Args:
        prompt: The prompt to send to Claude.
        model: The model to use (e.g., "haiku", "sonnet").

    Returns:
        The response text, or None if the command fails.
    """
    try:
        result = subprocess.run(
            [
                "claude",
                "-p", prompt,
                "--model", model,
                "--output-format", "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.debug("Claude CLI failed: %s", result.stderr)
            return None

        # Parse JSON output
        response = json.loads(result.stdout)

        # Extract the text from the response
        # The JSON format includes a "result" field with the response
        if "result" in response:
            return response["result"].strip()

        # Fallback: try to find text in content blocks
        if "content" in response:
            for block in response["content"]:
                if block.get("type") == "text":
                    return block.get("text", "").strip()

        logger.debug("Could not extract text from Claude response: %s", response)
        return None

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out")
        return None
    except FileNotFoundError:
        logger.debug("Claude CLI not found in PATH")
        return None
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse Claude CLI output: %s", e)
        return None
    except Exception as e:
        logger.warning("Claude CLI error: %s", e)
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


class ClaudeCLIEnhancerPlugin:
    """AI enhancement plugin using Claude Code CLI."""

    @hookimpl
    def agentgit_get_ai_enhancer_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": ENHANCER_NAME,
            "description": "Use Claude Code CLI to generate commit messages",
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

        message = _run_claude_cli(prompt, model)
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

        message = _run_claude_cli(ai_prompt, model)
        if message:
            return _clean_message(message)
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

        message = _run_claude_cli(ai_prompt, model)
        if message:
            return _clean_message(message)
        return None
