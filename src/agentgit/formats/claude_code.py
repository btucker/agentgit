"""Claude Code JSONL format parser plugin for agentgit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentgit.core import (
    AssistantContext,
    AssistantTurn,
    FileOperation,
    OperationType,
    Prompt,
    PromptResponse,
    Transcript,
    TranscriptEntry,
)
from agentgit.plugins import hookimpl
from agentgit.utils import extract_deleted_paths

# Format identifier for Claude Code JSONL transcripts
FORMAT_CLAUDE_CODE_JSONL = "claude_code_jsonl"


class ClaudeCodePlugin:
    """Plugin for parsing Claude Code JSONL transcripts."""

    @hookimpl
    def agentgit_get_plugin_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": "claude_code",
            "description": "Claude Code JSONL transcripts",
        }

    @hookimpl
    def agentgit_detect_format(self, path: Path) -> str | None:
        """Detect Claude Code JSONL format."""
        if path.suffix != ".jsonl":
            return None

        try:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    try:
                        obj = json.loads(line.strip())
                        if obj.get("type") in ("user", "assistant", "summary"):
                            return FORMAT_CLAUDE_CODE_JSONL
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_parse_transcript(self, path: Path, format: str) -> Transcript | None:
        """Parse Claude Code JSONL transcript."""
        if format != FORMAT_CLAUDE_CODE_JSONL:
            return None

        entries = []
        prompts = []
        session_id = None
        session_cwd = None

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type")
                if entry_type not in ("user", "assistant"):
                    continue

                timestamp = obj.get("timestamp", "")
                message = obj.get("message", {})

                if not session_id:
                    session_id = obj.get("sessionId")
                if not session_cwd:
                    session_cwd = obj.get("cwd")

                entry = TranscriptEntry(
                    entry_type=entry_type,
                    timestamp=timestamp,
                    message=message,
                    raw_entry=obj,
                    is_continuation=obj.get("isCompactSummary", False),
                    is_meta=obj.get("isMeta", False),
                )
                entries.append(entry)

                if entry_type == "user" and not entry.is_meta:
                    text = self._extract_text_from_content(message.get("content", ""))
                    if text:
                        prompts.append(
                            Prompt(
                                text=text,
                                timestamp=timestamp,
                                raw_entry=obj,
                            )
                        )

        return Transcript(
            entries=entries,
            prompts=prompts,
            source_path=str(path),
            source_format=FORMAT_CLAUDE_CODE_JSONL,
            session_id=session_id,
            session_cwd=session_cwd,
        )

    @hookimpl
    def agentgit_extract_operations(self, transcript: Transcript) -> list[FileOperation]:
        """Extract file operations from Claude Code transcript."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return []

        operations = []

        # First pass: collect originalFile content from tool results
        tool_id_to_original: dict[str, str] = {}
        for entry in transcript.entries:
            tool_use_result = entry.raw_entry.get("toolUseResult", {})
            if tool_use_result and "originalFile" in tool_use_result:
                content = entry.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "")
                            if tool_id:
                                tool_id_to_original[tool_id] = tool_use_result.get(
                                    "originalFile"
                                )

        # Second pass: extract operations
        for entry in transcript.entries:
            if entry.entry_type != "assistant":
                continue

            content = entry.message.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_id = block.get("id", "")
                tool_input = block.get("input", {})

                if tool_name == "Write":
                    file_path = tool_input.get("file_path", "")
                    if file_path:  # Skip if file_path is empty or missing
                        operations.append(
                            FileOperation(
                                file_path=file_path,
                                operation_type=OperationType.WRITE,
                                timestamp=entry.timestamp,
                                tool_id=tool_id,
                                content=tool_input.get("content", ""),
                                raw_tool_use=block,
                            )
                        )

                elif tool_name == "Edit":
                    file_path = tool_input.get("file_path", "")
                    if file_path:  # Skip if file_path is empty or missing
                        operations.append(
                            FileOperation(
                                file_path=file_path,
                                operation_type=OperationType.EDIT,
                                timestamp=entry.timestamp,
                                tool_id=tool_id,
                                old_string=tool_input.get("old_string", ""),
                                new_string=tool_input.get("new_string", ""),
                                replace_all=tool_input.get("replace_all", False),
                                original_content=tool_id_to_original.get(tool_id),
                                raw_tool_use=block,
                            )
                        )

                elif tool_name == "Bash":
                    command = tool_input.get("command", "")
                    deleted_paths = extract_deleted_paths(command)
                    is_recursive = "-r" in command

                    for path in deleted_paths:
                        operations.append(
                            FileOperation(
                                file_path=path,
                                operation_type=OperationType.DELETE,
                                timestamp=entry.timestamp,
                                tool_id=tool_id,
                                recursive=is_recursive,
                                raw_tool_use=block,
                            )
                        )

        operations.sort(key=lambda op: op.timestamp)
        return operations

    @hookimpl
    def agentgit_enrich_operation(
        self,
        operation: FileOperation,
        transcript: Transcript,
    ) -> FileOperation:
        """Enrich operation with prompt and assistant context."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return operation

        # Find the prompt that triggered this operation
        current_prompt = None
        for prompt in transcript.prompts:
            if prompt.timestamp <= operation.timestamp:
                current_prompt = prompt
            else:
                break

        if current_prompt:
            operation.prompt = current_prompt

        # Find assistant context (thinking/text) immediately before this tool use
        context = AssistantContext()
        for entry in transcript.entries:
            if entry.timestamp > operation.timestamp:
                break

            if entry.entry_type == "assistant":
                content = entry.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking":
                                context.thinking = block.get("thinking", "")
                                context.timestamp = entry.timestamp
                            elif block.get("type") == "text":
                                context.text = block.get("text", "")
                                context.timestamp = entry.timestamp
                            elif block.get("type") == "tool_use":
                                if block.get("id") == operation.tool_id:
                                    if context.thinking or context.text:
                                        operation.assistant_context = context
                                    return operation
                                context = AssistantContext()

        return operation

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract plain text from message content."""
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts).strip()
        return ""

    def _add_context_if_needed(
        self, prompt: Prompt, assistant_texts: list[str]
    ) -> Prompt:
        """Add assistant context to a prompt if it appears to need it.

        Short prompts like "yes", "do it", or "let's do 2, 3, 4" need the
        preceding assistant text to make sense.
        """
        if not self._prompt_needs_context(prompt.text):
            return prompt

        # Combine assistant texts, most recent last
        context = "\n\n".join(assistant_texts)

        # Truncate if too long (keep last 1500 chars which likely has the question)
        if len(context) > 1500:
            context = "..." + context[-1500:]

        # Format with clear delimiter
        combined_text = f"[Assistant context:\n{context}]\n\n{prompt.text}"

        return Prompt(
            text=combined_text,
            timestamp=prompt.timestamp,
            raw_entry=prompt.raw_entry,
        )

    def _prompt_needs_context(self, text: str) -> bool:
        """Determine if a prompt is too short/referential to stand alone.

        Returns True if the prompt likely needs assistant context to make sense.
        """
        text = text.strip()

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
        # These typically reference a list from the assistant
        if re.search(r"\b\d+\s*(,|and|&|\+)\s*\d+", text_lower):
            return True

        # Check for "that", "this", "it", "those" at start - referential
        referential_starts = ["that", "this", "it ", "its ", "those", "these", "the "]
        for ref in referential_starts:
            if text_lower.startswith(ref):
                return True

        return False

    @hookimpl
    def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
        """Get the project name from a Claude Code transcript.

        Reads the transcript to find the first cwd, then finds the git root
        (tracing upward if needed) and returns its directory name.
        """
        from agentgit import find_git_root

        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Check if this transcript is in ~/.claude/projects/
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        # Extract cwd from the transcript
        cwd = self._extract_cwd_from_transcript(transcript_path)
        if not cwd:
            return None

        # Find git root (may be cwd itself or a parent directory)
        project_root = find_git_root(cwd)
        return project_root.name if project_root else Path(cwd).name

    def _extract_cwd_from_transcript(self, transcript_path: Path) -> str | None:
        """Extract the first cwd from a Claude Code transcript."""
        try:
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cwd = obj.get("cwd")
                        if cwd:
                            return cwd
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_get_display_name(self, transcript_path: Path) -> str | None:
        """Get display name for a Claude Code transcript.

        Returns the filename with UUIDs truncated for readability.
        """
        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Only handle Claude Code transcripts
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        filename = transcript_path.name

        # Truncate UUIDs (8-4-4-4-12 pattern) to first 8 chars
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        filename = re.sub(uuid_pattern, lambda m: m.group()[:8] + "…", filename)

        return filename

    @hookimpl
    def agentgit_build_prompt_responses(
        self, transcript: Transcript
    ) -> list[PromptResponse]:
        """Build prompt-response structure grouping operations by assistant message."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return []

        # Build a mapping of tool_id -> operation for quick lookup
        tool_id_to_op: dict[str, FileOperation] = {}
        for op in transcript.operations:
            if op.tool_id:
                tool_id_to_op[op.tool_id] = op

        prompt_responses: list[PromptResponse] = []
        current_prompt: Prompt | None = None
        current_turns: list[AssistantTurn] = []
        # Track assistant text between user prompts for context
        pending_assistant_text: list[str] = []

        for entry in transcript.entries:
            # New user prompt starts a new PromptResponse
            if entry.entry_type == "user" and not entry.is_meta:
                # Find or create the Prompt object for this entry
                text = self._extract_text_from_content(entry.message.get("content", ""))
                new_prompt: Prompt | None = None
                for p in transcript.prompts:
                    if p.timestamp == entry.timestamp:
                        new_prompt = p
                        break
                if new_prompt is None and text:
                    new_prompt = Prompt(
                        text=text, timestamp=entry.timestamp, raw_entry=entry.raw_entry
                    )

                # Add assistant context if the prompt needs it
                if new_prompt and pending_assistant_text:
                    new_prompt = self._add_context_if_needed(
                        new_prompt, pending_assistant_text
                    )

                if current_prompt is not None:
                    if current_turns:
                        # Previous prompt had operations - save it and start fresh
                        prompt_responses.append(
                            PromptResponse(prompt=current_prompt, turns=current_turns)
                        )
                        current_prompt = new_prompt
                        current_turns = []
                        pending_assistant_text = []
                    elif new_prompt:
                        # No operations between prompts - concatenate them
                        combined_text = current_prompt.text + "\n\n" + new_prompt.text
                        current_prompt = Prompt(
                            text=combined_text,
                            timestamp=current_prompt.timestamp,  # Keep original timestamp
                            raw_entry=current_prompt.raw_entry,
                        )
                else:
                    current_prompt = new_prompt
                    current_turns = []
                    pending_assistant_text = []

            # Assistant message - track text and create turns
            elif entry.entry_type == "assistant" and current_prompt is not None:
                # Collect assistant text for potential context
                assistant_text = self._extract_text_from_content(
                    entry.message.get("content", [])
                )
                if assistant_text:
                    pending_assistant_text.append(assistant_text)

                turn = self._build_assistant_turn(entry, tool_id_to_op)
                if turn.operations:  # Only add turns that have file operations
                    current_turns.append(turn)
                    # Clear pending text after operations - context is less relevant
                    pending_assistant_text = []

        # Don't forget the last prompt response
        if current_prompt is not None and current_turns:
            prompt_responses.append(
                PromptResponse(prompt=current_prompt, turns=current_turns)
            )

        return prompt_responses

    def _build_assistant_turn(
        self, entry: TranscriptEntry, tool_id_to_op: dict[str, FileOperation]
    ) -> AssistantTurn:
        """Build an AssistantTurn from an assistant entry."""
        operations: list[FileOperation] = []
        context = AssistantContext()

        content = entry.message.get("content", [])
        if not isinstance(content, list):
            return AssistantTurn(timestamp=entry.timestamp, raw_entry=entry.raw_entry)

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "thinking":
                context.thinking = block.get("thinking", "")
                context.timestamp = entry.timestamp
            elif block_type == "text":
                context.text = block.get("text", "")
                context.timestamp = entry.timestamp
            elif block_type == "tool_use":
                tool_id = block.get("id", "")
                if tool_id in tool_id_to_op:
                    operations.append(tool_id_to_op[tool_id])

        return AssistantTurn(
            operations=operations,
            context=context if (context.thinking or context.text) else None,
            timestamp=entry.timestamp,
            raw_entry=entry.raw_entry,
        )

    @hookimpl
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Discover Claude Code transcripts for a project.

        Looks in ~/.claude/projects/ for directories matching the project path.
        Claude Code stores projects with paths like:
        ~/.claude/projects/-Users-username-path-to-project/

        Args:
            project_path: Path to the project. If None, returns all transcripts
                from all projects.
        """
        claude_projects_dir = Path.home() / ".claude" / "projects"
        if not claude_projects_dir.exists():
            return []

        transcripts = []

        if project_path is None:
            # Return all transcripts from all projects
            for project_dir in claude_projects_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                for jsonl_file in project_dir.glob("*.jsonl"):
                    # Skip agent sub-transcripts (they start with "agent-")
                    if jsonl_file.name.startswith("agent-"):
                        continue
                    transcripts.append(jsonl_file)
        else:
            # Convert project path to Claude's format: /path/to/project -> -path-to-project
            project_path = project_path.resolve()
            encoded_path = str(project_path).replace("/", "-")

            # Look for matching project directory
            project_dir = claude_projects_dir / encoded_path
            if not project_dir.exists():
                return []

            # Find all JSONL files in the project directory
            for jsonl_file in project_dir.glob("*.jsonl"):
                # Skip agent sub-transcripts (they start with "agent-")
                if jsonl_file.name.startswith("agent-"):
                    continue
                transcripts.append(jsonl_file)

        # Sort by modification time, most recent first
        transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return transcripts
