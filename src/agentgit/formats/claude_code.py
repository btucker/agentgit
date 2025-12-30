"""Claude Code JSONL format parser plugin for agentgit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentgit.core import (
    AssistantContext,
    FileOperation,
    OperationType,
    Prompt,
    Transcript,
    TranscriptEntry,
)
from agentgit.plugins import hookimpl

# Format identifier for Claude Code JSONL transcripts
FORMAT_CLAUDE_CODE_JSONL = "claude_code_jsonl"

RM_COMMAND_PATTERN = re.compile(r"^\s*rm\s+(?:-[rfivI]+\s+)*(.+)$")


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
                    deleted_paths = self._extract_deleted_paths(command)
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

    def _extract_deleted_paths(self, command: str) -> list[str]:
        """Extract file paths deleted by an rm command."""
        paths = []
        match = RM_COMMAND_PATTERN.match(command)
        if not match:
            return paths

        args_str = match.group(1).strip()
        current_path = ""
        in_quotes: str | None = None

        for char in args_str:
            if in_quotes:
                if char == in_quotes:
                    if current_path:
                        paths.append(current_path)
                        current_path = ""
                    in_quotes = None
                else:
                    current_path += char
            elif char in ('"', "'"):
                in_quotes = char
            elif char == " ":
                if current_path:
                    paths.append(current_path)
                    current_path = ""
            else:
                current_path += char

        if current_path:
            paths.append(current_path)

        return paths

    @hookimpl
    def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
        """Get the project name from a Claude Code transcript location.

        Returns the encoded project directory name from
        ~/.claude/projects/-Users-name-project/session.jsonl
        which is "-Users-name-project".
        """
        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Check if this transcript is in ~/.claude/projects/
        try:
            relative = transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        # Return the encoded project directory name
        return relative.parts[0]

    @hookimpl
    def agentgit_discover_transcripts(self, project_path: Path) -> list[Path]:
        """Discover Claude Code transcripts for a project.

        Looks in ~/.claude/projects/ for directories matching the project path.
        Claude Code stores projects with paths like:
        ~/.claude/projects/-Users-username-path-to-project/
        """
        claude_projects_dir = Path.home() / ".claude" / "projects"
        if not claude_projects_dir.exists():
            return []

        # Convert project path to Claude's format: /path/to/project -> -path-to-project
        project_path = project_path.resolve()
        encoded_path = str(project_path).replace("/", "-")

        # Look for matching project directory
        project_dir = claude_projects_dir / encoded_path
        if not project_dir.exists():
            return []

        # Find all JSONL files in the project directory
        transcripts = []
        for jsonl_file in project_dir.glob("*.jsonl"):
            # Skip agent sub-transcripts (they start with "agent-")
            if jsonl_file.name.startswith("agent-"):
                continue
            transcripts.append(jsonl_file)

        # Sort by modification time, most recent first
        transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return transcripts
