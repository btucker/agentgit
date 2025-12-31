"""Rules-based enhancer plugin for agentgit.

This enhancer uses heuristics to improve commit messages without requiring
any external AI API. It's the default enhancer when --enhance is used.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentgit.core import OperationType
from agentgit.plugins import hookimpl

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt

# Plugin identifier
ENHANCER_NAME = "rules"


def _prompt_needs_context(text: str) -> bool:
    """Determine if a prompt is too short/referential to stand alone.

    Returns True if the prompt likely needs assistant context to make sense.
    """
    text = text.strip()

    # If prompt starts with an action verb, it's likely self-contained
    if _extract_action_from_prompt(text):
        return False

    # Very short prompts almost always need context
    if len(text) < 50:
        return True

    # Common affirmative/directive responses that need context
    contextual_starters = [
        "yes",
        "no",
        "ok",
        "okay",
        "sure",
        "do it",
        "go ahead",
        "let's do",
        "let's go",
        "sounds good",
        "that works",
        "perfect",
        "great",
        "please do",
        "go for it",
        "make it so",
        "proceed",
        "continue",
        "approved",
        "confirmed",
        "agreed",
        "correct",
        "right",
        "exactly",
        "yep",
        "yup",
        "nope",
        "skip",
        "ignore",
        "both",
        "neither",
        "all of",
        "none of",
        "the first",
        "the second",
        "the last",
        "option",
    ]

    text_lower = text.lower()
    for starter in contextual_starters:
        if text_lower.startswith(starter):
            return True

    # Check for numbered references like "1, 2, 3" or "2 and 4"
    if re.search(r"\b\d+\s*(,|and|&|\+)\s*\d+", text_lower):
        return True

    # Check for referential starts
    referential_starts = ["that", "this", "it ", "its ", "those", "these", "the "]
    for ref in referential_starts:
        if text_lower.startswith(ref):
            return True

    return False


def _extract_action_from_prompt(text: str) -> str | None:
    """Try to extract an action verb from the prompt text."""
    text = text.strip().lower()

    # Common action patterns
    action_patterns = [
        (r"^(?:please\s+)?add\s+(.+)", "Add"),
        (r"^(?:please\s+)?create\s+(.+)", "Create"),
        (r"^(?:please\s+)?implement\s+(.+)", "Implement"),
        (r"^(?:please\s+)?fix\s+(.+)", "Fix"),
        (r"^(?:please\s+)?update\s+(.+)", "Update"),
        (r"^(?:please\s+)?remove\s+(.+)", "Remove"),
        (r"^(?:please\s+)?delete\s+(.+)", "Delete"),
        (r"^(?:please\s+)?refactor\s+(.+)", "Refactor"),
        (r"^(?:please\s+)?rename\s+(.+)", "Rename"),
        (r"^(?:please\s+)?move\s+(.+)", "Move"),
        (r"^(?:please\s+)?change\s+(.+)", "Update"),
        (r"^(?:please\s+)?modify\s+(.+)", "Modify"),
        (r"^(?:please\s+)?write\s+(.+)", "Add"),
        (r"^(?:please\s+)?make\s+(.+)", "Update"),
        (r"^(?:please\s+)?set\s+(.+)", "Set"),
        (r"^(?:please\s+)?configure\s+(.+)", "Configure"),
        (r"^(?:please\s+)?enable\s+(.+)", "Enable"),
        (r"^(?:please\s+)?disable\s+(.+)", "Disable"),
        (r"^(?:please\s+)?install\s+(.+)", "Install"),
        (r"^(?:please\s+)?upgrade\s+(.+)", "Upgrade"),
    ]

    for pattern, verb in action_patterns:
        match = re.match(pattern, text)
        if match:
            return verb

    return None


def _summarize_files(files: list[str], max_files: int = 3) -> str:
    """Create a summary of files for commit message."""
    if not files:
        return ""

    # Get just filenames
    names = [f.split("/")[-1] for f in files]

    if len(names) == 1:
        return names[0]
    elif len(names) <= max_files:
        return ", ".join(names[:-1]) + " and " + names[-1]
    else:
        return f"{names[0]} and {len(names) - 1} other files"


def _generate_from_context(
    operation: "FileOperation | None" = None,
    turn: "AssistantTurn | None" = None,
    prompt: "Prompt | None" = None,
) -> str | None:
    """Generate a commit message using context from operation/turn/prompt."""
    # Try to get verb from prompt
    verb = None
    if prompt and not _prompt_needs_context(prompt.text):
        verb = _extract_action_from_prompt(prompt.text)

    # Fall back to operation type
    if not verb:
        if operation:
            verb = {
                OperationType.WRITE: "Add",
                OperationType.EDIT: "Update",
                OperationType.DELETE: "Remove",
            }.get(operation.operation_type, "Update")
        elif turn:
            # Determine primary action from turn
            if turn.files_created and not turn.files_modified:
                verb = "Add"
            elif turn.files_deleted and not turn.files_created:
                verb = "Remove"
            else:
                verb = "Update"

    if not verb:
        return None

    # Build subject based on what we have
    if operation:
        filename = operation.file_path.split("/")[-1]
        return f"{verb} {filename}"

    if turn:
        all_files = turn.files_created + turn.files_modified + turn.files_deleted
        if all_files:
            file_summary = _summarize_files(all_files)
            return f"{verb} {file_summary}"

    # If we have a good prompt, use first line
    if prompt and not _prompt_needs_context(prompt.text):
        first_line = prompt.text.split("\n")[0].strip()
        if len(first_line) <= 72:
            return first_line
        return first_line[:69] + "..."

    return None


class RulesEnhancerPlugin:
    """Rules-based enhancer using heuristics (no AI required)."""

    @hookimpl
    def agentgit_get_ai_enhancer_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": ENHANCER_NAME,
            "description": "Rules-based enhancement using heuristics (no AI)",
        }

    @hookimpl
    def agentgit_enhance_operation_message(
        self,
        operation: "FileOperation",
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate a commit message for a file operation using rules."""
        if enhancer != ENHANCER_NAME:
            return None

        return _generate_from_context(
            operation=operation,
            prompt=operation.prompt,
        )

    @hookimpl
    def agentgit_enhance_turn_message(
        self,
        turn: "AssistantTurn",
        prompt: "Prompt | None",
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate a commit message for an assistant turn using rules."""
        if enhancer != ENHANCER_NAME:
            return None

        return _generate_from_context(
            turn=turn,
            prompt=prompt,
        )

    @hookimpl
    def agentgit_enhance_merge_message(
        self,
        prompt: "Prompt",
        turns: list["AssistantTurn"],
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate a merge commit message using rules."""
        if enhancer != ENHANCER_NAME:
            return None

        # For merge commits, try to use the prompt if it's descriptive enough
        if not _prompt_needs_context(prompt.text):
            first_line = prompt.text.split("\n")[0].strip()
            if len(first_line) <= 72:
                return first_line
            return first_line[:69] + "..."

        # Otherwise, summarize what was done
        all_created = []
        all_modified = []
        all_deleted = []
        for turn in turns:
            all_created.extend(turn.files_created)
            all_modified.extend(turn.files_modified)
            all_deleted.extend(turn.files_deleted)

        # Determine primary action
        if all_created and not all_modified and not all_deleted:
            verb = "Add"
            files = all_created
        elif all_deleted and not all_created and not all_modified:
            verb = "Remove"
            files = all_deleted
        else:
            verb = "Update"
            files = all_created + all_modified + all_deleted

        if files:
            file_summary = _summarize_files(files)
            return f"{verb} {file_summary}"

        return None
