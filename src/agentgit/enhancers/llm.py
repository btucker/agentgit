"""LLM-based AI enhancement plugin for agentgit.

This enhancer uses the `llm` library to:
1. Add context to referential prompts (e.g., "yes" → "yes - Add JWT auth")
2. Curate which assistant thinking/reasoning to include in commits
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agentgit.core import OperationType
from agentgit.plugins import hookimpl

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt, PromptResponse

logger = logging.getLogger(__name__)

# Default model for commit message generation (uses llm-claude-cli)
DEFAULT_MODEL = "claude-cli-haiku"

# Maximum items per batch to avoid context limits
MAX_BATCH_SIZE = 25

# Plugin identifier
ENHANCER_NAME = "llm"

# Global cache for batch-processed messages
_message_cache: dict[str, str] = {}

# Cached model instance
_model_cache: dict[str, object] = {}


def _prompt_needs_context(text: str) -> bool:
    """Determine if a prompt is too short/referential to stand alone.

    Returns True if the prompt likely needs assistant context to make sense.
    """
    # Import the shared logic from rules enhancer
    from agentgit.enhancers.rules import _prompt_needs_context as rules_needs_context

    return rules_needs_context(text)


def _get_model(model: str = DEFAULT_MODEL):
    """Get or create an LLM model instance.

    Args:
        model: The model ID to use (e.g., "claude-3-5-haiku-latest", "gpt-4o-mini").

    Returns:
        LLM model instance, or None if llm is not installed.
    """
    if model in _model_cache:
        return _model_cache[model]

    try:
        import llm

        instance = llm.get_model(model)
        _model_cache[model] = instance
        return instance
    except ImportError:
        logger.warning(
            "llm not installed. Install with: pip install 'agentgit[llm]'"
        )
        return None
    except Exception as e:
        logger.warning("Failed to initialize model %s: %s", model, e)
        return None


def _run_llm(prompt: str, model: str = DEFAULT_MODEL, schema: dict | None = None) -> str | None:
    """Run a prompt through the LLM and return the response text.

    Args:
        prompt: The prompt to send.
        model: The model ID to use.
        schema: Optional JSON schema to enforce structured output.

    Returns:
        The response text, or None if the request fails.
    """
    llm_model = _get_model(model)
    if llm_model is None:
        return None

    try:
        response = llm_model.prompt(prompt, schema=schema)
        return response.text()
    except Exception as e:
        logger.warning("LLM request failed: %s", e)
        return None


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _truncate_subject(text: str) -> str:
    """Truncate text to 72 chars for git subject line."""
    return _truncate_text(text, 72)


def _gather_turn_files(
    turns: list["AssistantTurn"],
) -> tuple[list[str], list[str], list[str]]:
    """Gather all files from a list of turns.

    Returns:
        Tuple of (created, modified, deleted) file lists.
    """
    created, modified, deleted = [], [], []
    for turn in turns:
        created.extend(turn.files_created)
        modified.extend(turn.files_modified)
        deleted.extend(turn.files_deleted)
    return created, modified, deleted


def _format_referential_subject(prompt_first_line: str, summary: str | None) -> str:
    """Format a referential prompt with context summary.

    For prompts like "yes" or "do it", combines with summary:
    "yes - Add JWT authentication"

    If summary is empty or prompt is too long, just returns truncated prompt.
    """
    if not summary:
        return _truncate_subject(prompt_first_line)

    max_summary_len = 72 - len(prompt_first_line) - 3  # 3 for " - "
    if max_summary_len > 10:
        if len(summary) > max_summary_len:
            summary = summary[: max_summary_len - 3] + "..."
        return f"{prompt_first_line} - {summary}"

    return _truncate_subject(prompt_first_line)


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
    return _truncate_subject(message)


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

    For merge commits (user prompts):
    - Self-contained prompts are used as-is
    - Referential prompts get context appended: "yes - Add JWT auth"

    For turn commits:
    - Generate a summary of what the assistant did

    Args:
        prompt_responses: List of PromptResponse objects to process.
        model: The model ID to use.

    Returns:
        Dictionary mapping cache keys to generated commit messages.
    """
    global _message_cache

    logger.info("batch_enhance_prompt_responses called with %d prompt responses, model=%s", len(prompt_responses), model)

    if not prompt_responses:
        return {}

    # First pass: handle self-contained prompts directly (no LLM needed)
    # and collect items that need LLM processing
    items = []
    item_keys = []
    item_metadata = []  # Store extra info for post-processing

    for pr in prompt_responses:
        prompt_key = _get_prompt_key(pr.prompt)
        if prompt_key not in _message_cache:
            first_line = pr.prompt.text.split("\n")[0].strip()

            # Self-contained prompts don't need LLM
            if not _prompt_needs_context(pr.prompt.text):
                _message_cache[prompt_key] = _truncate_subject(first_line)
            else:
                # Referential prompt - need LLM to add context
                context_parts = []
                created, modified, _ = _gather_turn_files(pr.turns)
                all_files = created + modified
                if all_files:
                    context_parts.append(f"Files: {', '.join(all_files[:10])}")

                for turn in pr.turns[:2]:
                    if turn.context and turn.context.summary:
                        context_parts.append(
                            _truncate_text(turn.context.summary, 300)
                        )

                items.append({
                    "id": len(items) + 1,
                    "type": "merge",
                    "prompt": first_line,
                    "context": "\n".join(context_parts),
                })
                item_keys.append(prompt_key)
                item_metadata.append({"first_line": first_line})

        # Add turn message requests
        for turn in pr.turns:
            turn_key = _get_turn_key(turn)
            logger.debug("Processing turn with key: %s (timestamp=%s, tool_id=%s)",
                        turn_key, turn.timestamp,
                        turn.operations[0].tool_id if turn.operations else "N/A")
            if turn_key not in _message_cache:
                items.append({
                    "id": len(items) + 1,
                    "type": "turn",
                    "context": _build_turn_context(turn, 800),
                })
                item_keys.append(turn_key)
                item_metadata.append({})

    if not items:
        logger.info("No items need LLM processing (all prompts self-contained)")
        return _message_cache

    logger.info("Processing %d items in batches (batch size: %d)", len(items), MAX_BATCH_SIZE)

    # Process items in chunks to avoid context limits
    for chunk_start in range(0, len(items), MAX_BATCH_SIZE):
        chunk_end = min(chunk_start + MAX_BATCH_SIZE, len(items))
        chunk_items = items[chunk_start:chunk_end]
        chunk_keys = item_keys[chunk_start:chunk_end]
        chunk_metadata = item_metadata[chunk_start:chunk_end]

        # Build the batch prompt for this chunk
        batch_prompt = """Generate commit message content for these items.

For "merge" items: The user prompt is referential (like "yes" or "do it").
Summarize what they were agreeing to in ~30-50 chars.

For "turn" items: Summarize what the assistant did in ~50-70 chars.

Examples:
- merge prompt "yes" with JWT context → "Add JWT authentication"
- merge prompt "go ahead" with refactor context → "Refactor database layer"
- turn with auth files → "Implement login and session handling"

Items:
"""

        for idx, item in enumerate(chunk_items, 1):
            if item["type"] == "merge":
                batch_prompt += f'\n[{idx}] (merge) User said: "{item["prompt"]}"\nContext: {item["context"]}\n'
            else:
                batch_prompt += f"\n[{idx}] (turn)\n{item['context']}\n"

        batch_prompt += """
Respond with a JSON object mapping item IDs to the summary text:
{"1": "Add JWT authentication", "2": "Implement login flow", ...}

ONLY respond with the JSON object, nothing else."""

        # Define JSON schema for structured output
        # Build properties for each item in the chunk
        properties = {}
        required = []
        for idx in range(1, len(chunk_items) + 1):
            properties[str(idx)] = {
                "type": "string",
                "description": f"Commit message summary for item {idx}"
            }
            required.append(str(idx))

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

        # Call LLM for this chunk with schema
        response = _run_llm(batch_prompt, model, schema=schema)

        if response:
            logger.debug("LLM response for chunk %d-%d: %s", chunk_start, chunk_end, response[:200])
            try:
                # Try to parse JSON response
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])

                messages = json.loads(response)
                logger.info("Successfully parsed %d messages from LLM response", len(messages))

                # Map responses back to cache keys
                for idx, (key, item, meta) in enumerate(
                    zip(chunk_keys, chunk_items, chunk_metadata), 1
                ):
                    item_id = str(idx)
                    if item_id in messages:
                        summary = messages[item_id].strip().strip('"').strip("'")

                        if item["type"] == "merge":
                            first_line = meta["first_line"]
                            _message_cache[key] = _format_referential_subject(
                                first_line, summary
                            )
                        else:
                            # Turn - just use the summary
                            _message_cache[key] = _clean_message(summary)

            except json.JSONDecodeError as e:
                logger.warning("Failed to parse batch response as JSON: %s", e)
                logger.debug("Response was: %s", response[:500])
        else:
            logger.warning("LLM returned no response for chunk %d-%d", chunk_start, chunk_end)

    logger.info("Batch enhancement complete. Total cache entries: %d", len(_message_cache))
    return _message_cache


def clear_message_cache() -> None:
    """Clear the message cache."""
    global _message_cache
    _message_cache = {}


class LLMEnhancerPlugin:
    """AI enhancement plugin using the llm library."""

    @hookimpl
    def agentgit_get_enhancer_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": ENHANCER_NAME,
            "description": "Use LLM to generate commit messages (requires llm package)",
        }

    @hookimpl
    def agentgit_enhance_turn_summary(
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
        logger.debug("agentgit_enhance_turn_summary: turn_key=%s, in_cache=%s, cache_size=%d",
                    turn_key, turn_key in _message_cache, len(_message_cache))
        if turn_key in _message_cache:
            logger.debug("Returning cached message: %s", _message_cache[turn_key])
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

Examples of good commit messages:
- "Implement OAuth2 login flow"
- "Fix form validation across signup pages"
- "Refactor database queries to use connection pooling"
- "Configure ESLint rules for TypeScript"

Context:
{context}

Respond with ONLY the commit message subject line, nothing else."""

        message = _run_llm(ai_prompt, model)
        if message:
            result = _clean_message(message)
            _message_cache[turn_key] = result
            return result
        return None

    @hookimpl
    def agentgit_enhance_prompt_summary(
        self,
        prompt: "Prompt",
        turns: list["AssistantTurn"],
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an enhanced merge commit message for a user prompt.

        Preserves the exact user prompt text. For referential prompts like
        "yes" or "do it", appends context about what was agreed to.
        """
        if enhancer != ENHANCER_NAME:
            return None

        # Check cache first (populated by batch processing)
        prompt_key = _get_prompt_key(prompt)
        if prompt_key in _message_cache:
            return _message_cache[prompt_key]

        # Get first line of prompt for subject
        first_line = prompt.text.split("\n")[0].strip()

        # If prompt is self-contained, just use it as-is (truncated if needed)
        if not _prompt_needs_context(prompt.text):
            result = _truncate_subject(first_line)
            _message_cache[prompt_key] = result
            return result

        # Prompt is referential - need to add context from assistant
        model = model or DEFAULT_MODEL

        # Gather context about what was done
        context_parts = []
        all_created, all_modified, all_deleted = _gather_turn_files(turns)

        if all_created:
            context_parts.append(f"Files created: {', '.join(all_created[:10])}")
        if all_modified:
            context_parts.append(f"Files modified: {', '.join(all_modified[:10])}")
        if all_deleted:
            context_parts.append(f"Files deleted: {', '.join(all_deleted[:10])}")

        # Include assistant reasoning
        for turn in turns[:3]:
            if turn.context and turn.context.summary:
                reasoning = _truncate_text(turn.context.summary, 500)
                context_parts.append(f"\nAssistant reasoning:\n{reasoning}")

        context = "\n".join(context_parts)

        # Ask LLM to summarize what the user was agreeing to
        ai_prompt = f"""The user said: "{first_line}"

This was a response to assistant work. Summarize what the user was agreeing to in a few words (max 50 characters).

{context}

Examples:
- User: "yes" → "Add JWT authentication"
- User: "go ahead" → "Refactor database queries"
- User: "the first one" → "Use Redis for caching"

Respond with ONLY the short summary, nothing else."""

        summary = _run_llm(ai_prompt, model)
        if summary:
            summary = summary.strip().strip('"').strip("'")
        result = _format_referential_subject(first_line, summary)

        _message_cache[prompt_key] = result
        return result

    @hookimpl
    def agentgit_curate_turn_context(
        self,
        turn: "AssistantTurn",
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Curate the context/reasoning to include in a turn commit body.

        Selects and organizes the most relevant parts of the assistant's
        thinking to explain why these changes were made.
        """
        if enhancer != ENHANCER_NAME:
            return None

        # If no context available, nothing to curate
        if not turn.context or not turn.context.summary:
            return None

        model = model or DEFAULT_MODEL

        # Build the raw context
        raw_context = turn.context.summary

        # If context is short enough, use as-is
        if len(raw_context) <= 500:
            return raw_context

        # Ask LLM to curate/summarize the key reasoning
        ai_prompt = f"""Summarize the key reasoning from this assistant's thinking.
Focus on:
- Why these changes were made
- Key decisions or trade-offs
- Important context for understanding the code

Keep it concise (2-4 sentences, max 300 chars).

Assistant thinking:
{_truncate_text(raw_context, 2000)}

Respond with ONLY the summary, nothing else."""

        result = _run_llm(ai_prompt, model)
        if result:
            return result.strip()

        # Fall back to truncated original
        return _truncate_text(raw_context, 500)
