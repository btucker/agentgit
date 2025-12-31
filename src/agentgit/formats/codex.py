"""OpenAI Codex JSONL format parser plugin for agentgit."""

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
from agentgit.utils import extract_deleted_paths

# Format identifier for Codex JSONL transcripts
FORMAT_CODEX_JSONL = "codex_jsonl"

# Pattern to extract cwd from environment_context
CWD_PATTERN = re.compile(r"<cwd>([^<]+)</cwd>")

# Patterns for apply_patch parsing
PATCH_BEGIN = "*** Begin Patch"
PATCH_END = "*** End Patch"
ADD_FILE_PATTERN = re.compile(r"^\*\*\* Add File:\s*(.+)$")
UPDATE_FILE_PATTERN = re.compile(r"^\*\*\* Update File:\s*(.+)$")
DELETE_FILE_PATTERN = re.compile(r"^\*\*\* Delete File:\s*(.+)$")


class CodexPlugin:
    """Plugin for parsing OpenAI Codex JSONL transcripts."""

    @hookimpl
    def agentgit_get_plugin_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": "codex",
            "description": "OpenAI Codex CLI JSONL transcripts",
        }

    @hookimpl
    def agentgit_detect_format(self, path: Path) -> str | None:
        """Detect Codex JSONL format.

        Codex transcripts are identified by:
        - .jsonl extension
        - Contains events like thread.started, turn.started, or
          message objects with role field
        """
        if path.suffix != ".jsonl":
            return None

        try:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    try:
                        obj = json.loads(line.strip())

                        # Skip state records
                        if obj.get("record_type") == "state":
                            continue

                        # Check for Codex-specific markers
                        event_type = obj.get("type", "")

                        # Codex thread/turn events
                        if event_type in (
                            "thread.started",
                            "turn.started",
                            "turn.completed",
                            "item.started",
                            "item.completed",
                            "session_meta",  # Session metadata at start of file
                        ):
                            return FORMAT_CODEX_JSONL

                        # Codex message format: { type: "message", role: "user"|"assistant" }
                        if event_type == "message" and "role" in obj:
                            return FORMAT_CODEX_JSONL

                        # Codex function calls
                        if event_type == "function_call" and "call_id" in obj:
                            return FORMAT_CODEX_JSONL

                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_parse_transcript(self, path: Path, format: str) -> Transcript | None:
        """Parse Codex JSONL transcript."""
        if format != FORMAT_CODEX_JSONL:
            return None

        entries: list[TranscriptEntry] = []
        prompts: list[Prompt] = []
        session_id: str | None = None
        session_cwd: str | None = None

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip state records
                if obj.get("record_type") == "state":
                    continue

                event_type = obj.get("type", "")

                # Extract session_id from thread.started
                if event_type == "thread.started":
                    session_id = obj.get("thread_id")
                    entry = TranscriptEntry(
                        entry_type="thread",
                        timestamp="",
                        message=obj,
                        raw_entry=obj,
                    )
                    entries.append(entry)
                    continue

                # Handle user messages
                if event_type == "message" and obj.get("role") == "user":
                    content = obj.get("content", [])
                    text, cwd = self._extract_user_text_and_cwd(content)

                    if cwd and not session_cwd:
                        session_cwd = cwd

                    entry = TranscriptEntry(
                        entry_type="user",
                        timestamp=obj.get("timestamp", ""),
                        message={"content": content},
                        raw_entry=obj,
                    )
                    entries.append(entry)

                    if text:
                        prompts.append(
                            Prompt(
                                text=text,
                                timestamp=obj.get("timestamp", ""),
                                raw_entry=obj,
                            )
                        )
                    continue

                # Handle assistant messages
                if event_type == "message" and obj.get("role") == "assistant":
                    entry = TranscriptEntry(
                        entry_type="assistant",
                        timestamp=obj.get("timestamp", ""),
                        message=obj,
                        raw_entry=obj,
                    )
                    entries.append(entry)
                    continue

                # Handle function calls
                if event_type == "function_call":
                    entry = TranscriptEntry(
                        entry_type="function_call",
                        timestamp=obj.get("timestamp", ""),
                        message=obj,
                        raw_entry=obj,
                    )
                    entries.append(entry)
                    continue

                # Handle function call outputs
                if event_type == "function_call_output":
                    entry = TranscriptEntry(
                        entry_type="function_call_output",
                        timestamp=obj.get("timestamp", ""),
                        message=obj,
                        raw_entry=obj,
                    )
                    entries.append(entry)
                    continue

                # Handle item events (reasoning, agent_message, command_execution, etc.)
                if event_type in ("item.started", "item.completed", "item.updated"):
                    item = obj.get("item", {})
                    entry = TranscriptEntry(
                        entry_type=f"item_{item.get('type', 'unknown')}",
                        timestamp=obj.get("timestamp", ""),
                        message=item,
                        raw_entry=obj,
                    )
                    entries.append(entry)
                    continue

                # Handle turn events
                if event_type in ("turn.started", "turn.completed", "turn.failed"):
                    entry = TranscriptEntry(
                        entry_type=event_type,
                        timestamp=obj.get("timestamp", ""),
                        message=obj,
                        raw_entry=obj,
                    )
                    entries.append(entry)
                    continue

        return Transcript(
            entries=entries,
            prompts=prompts,
            source_path=str(path),
            source_format=FORMAT_CODEX_JSONL,
            session_id=session_id,
            session_cwd=session_cwd,
        )

    @hookimpl
    def agentgit_extract_operations(
        self, transcript: Transcript
    ) -> list[FileOperation]:
        """Extract file operations from Codex transcript."""
        if transcript.source_format != FORMAT_CODEX_JSONL:
            return []

        operations: list[FileOperation] = []

        for entry in transcript.entries:
            # Extract operations from function_call entries (apply_patch)
            if entry.entry_type == "function_call":
                ops = self._extract_from_function_call(entry)
                operations.extend(ops)

            # Extract operations from command_execution items (rm commands)
            elif entry.entry_type == "item_command_execution":
                ops = self._extract_from_command_execution(entry)
                operations.extend(ops)

        return operations

    def _extract_from_function_call(
        self, entry: TranscriptEntry
    ) -> list[FileOperation]:
        """Extract file operations from a function_call entry."""
        operations: list[FileOperation] = []
        raw = entry.raw_entry

        name = raw.get("name", "")
        call_id = raw.get("call_id", "")
        timestamp = entry.timestamp

        # Only handle shell function calls with apply_patch
        if name != "shell":
            return operations

        try:
            arguments = json.loads(raw.get("arguments", "{}"))
        except json.JSONDecodeError:
            return operations

        cmd = arguments.get("cmd", [])
        if not cmd or cmd[0] != "apply_patch":
            return operations

        # The patch content is the second element
        if len(cmd) < 2:
            return operations

        patch_content = cmd[1]
        return self._parse_patch(patch_content, call_id, timestamp)

    def _parse_patch(
        self, patch_content: str, call_id: str, timestamp: str
    ) -> list[FileOperation]:
        """Parse apply_patch content into file operations.

        Patch format:
        *** Begin Patch
        *** Add File: path/to/file.py
        content here
        *** Update File: another/file.py
        @@ diff here @@
        *** Delete File: old/file.py
        *** End Patch
        """
        operations: list[FileOperation] = []
        lines = patch_content.split("\n")

        current_op: dict[str, Any] | None = None
        content_lines: list[str] = []

        for line in lines:
            # Skip begin/end markers
            if line.strip() == PATCH_BEGIN:
                continue
            if line.strip() == PATCH_END:
                # Finalize any pending operation
                if current_op:
                    self._finalize_operation(current_op, content_lines, operations)
                    current_op = None  # Prevent double finalization
                break

            # Check for Add File
            add_match = ADD_FILE_PATTERN.match(line)
            if add_match:
                # Finalize previous operation
                if current_op:
                    self._finalize_operation(current_op, content_lines, operations)

                current_op = {
                    "type": OperationType.WRITE,
                    "file_path": add_match.group(1).strip(),
                    "call_id": call_id,
                    "timestamp": timestamp,
                }
                content_lines = []
                continue

            # Check for Update File
            update_match = UPDATE_FILE_PATTERN.match(line)
            if update_match:
                if current_op:
                    self._finalize_operation(current_op, content_lines, operations)

                current_op = {
                    "type": OperationType.EDIT,
                    "file_path": update_match.group(1).strip(),
                    "call_id": call_id,
                    "timestamp": timestamp,
                }
                content_lines = []
                continue

            # Check for Delete File
            delete_match = DELETE_FILE_PATTERN.match(line)
            if delete_match:
                if current_op:
                    self._finalize_operation(current_op, content_lines, operations)

                current_op = {
                    "type": OperationType.DELETE,
                    "file_path": delete_match.group(1).strip(),
                    "call_id": call_id,
                    "timestamp": timestamp,
                }
                content_lines = []
                continue

            # Accumulate content lines
            if current_op:
                content_lines.append(line)

        # Handle case where patch doesn't end with *** End Patch
        if current_op:
            self._finalize_operation(current_op, content_lines, operations)

        return operations

    def _finalize_operation(
        self,
        op_data: dict[str, Any],
        content_lines: list[str],
        operations: list[FileOperation],
    ) -> None:
        """Create a FileOperation from accumulated data."""
        op_type = op_data["type"]
        file_path = op_data["file_path"]
        call_id = op_data["call_id"]
        timestamp = op_data["timestamp"]

        if op_type == OperationType.WRITE:
            content = "\n".join(content_lines)
            operations.append(
                FileOperation(
                    file_path=file_path,
                    operation_type=OperationType.WRITE,
                    timestamp=timestamp,
                    tool_id=call_id,
                    content=content,
                    raw_tool_use=op_data,
                )
            )
        elif op_type == OperationType.EDIT:
            # For edits, store the diff content
            diff_content = "\n".join(content_lines)
            operations.append(
                FileOperation(
                    file_path=file_path,
                    operation_type=OperationType.EDIT,
                    timestamp=timestamp,
                    tool_id=call_id,
                    content=diff_content,  # Store diff in content field
                    raw_tool_use=op_data,
                )
            )
        elif op_type == OperationType.DELETE:
            operations.append(
                FileOperation(
                    file_path=file_path,
                    operation_type=OperationType.DELETE,
                    timestamp=timestamp,
                    tool_id=call_id,
                    raw_tool_use=op_data,
                )
            )

    def _extract_from_command_execution(
        self, entry: TranscriptEntry
    ) -> list[FileOperation]:
        """Extract delete operations from command_execution items (rm commands)."""
        operations: list[FileOperation] = []
        item = entry.message

        command = item.get("command", "")
        item_id = item.get("id", "")
        timestamp = entry.timestamp

        # Check for rm commands
        deleted_paths = extract_deleted_paths(command)
        is_recursive = "-r" in command

        for path in deleted_paths:
            operations.append(
                FileOperation(
                    file_path=path,
                    operation_type=OperationType.DELETE,
                    timestamp=timestamp,
                    tool_id=item_id,
                    recursive=is_recursive,
                    raw_tool_use=item,
                )
            )

        return operations

    @hookimpl
    def agentgit_enrich_operation(
        self,
        operation: FileOperation,
        transcript: Transcript,
    ) -> FileOperation:
        """Enrich operation with prompt and assistant context."""
        if transcript.source_format != FORMAT_CODEX_JSONL:
            return operation

        # Find the prompt that triggered this operation
        current_prompt = None
        for prompt in transcript.prompts:
            if not operation.timestamp or prompt.timestamp <= operation.timestamp:
                current_prompt = prompt
            else:
                break

        if current_prompt:
            operation.prompt = current_prompt

        # Find reasoning context before this operation
        context = AssistantContext()
        for entry in transcript.entries:
            if operation.timestamp and entry.timestamp > operation.timestamp:
                break

            # Look for reasoning items
            if entry.entry_type == "item_reasoning":
                reasoning_text = entry.message.get("text", "")
                if reasoning_text:
                    context.thinking = reasoning_text
                    context.timestamp = entry.timestamp

            # Look for agent_message items
            elif entry.entry_type == "item_agent_message":
                msg_text = entry.message.get("text", "")
                if msg_text:
                    context.text = msg_text
                    context.timestamp = entry.timestamp

            # Reset context when we hit the function call for this operation
            elif entry.entry_type == "function_call":
                if entry.raw_entry.get("call_id") == operation.tool_id:
                    if context.thinking or context.text:
                        operation.assistant_context = context
                    return operation

            # Reset context when we hit the command_execution for this operation
            elif entry.entry_type == "item_command_execution":
                if entry.message.get("id") == operation.tool_id:
                    if context.thinking or context.text:
                        operation.assistant_context = context
                    return operation

        # If no explicit match, still set context if available
        if context.thinking or context.text:
            operation.assistant_context = context

        return operation

    def _extract_user_text_and_cwd(
        self, content: list | str
    ) -> tuple[str, str | None]:
        """Extract user text and cwd from message content.

        Codex user messages can contain environment_context blocks with cwd info.
        These should be excluded from the prompt text.
        """
        if isinstance(content, str):
            # Check for environment_context in string content
            cwd_match = CWD_PATTERN.search(content)
            cwd = cwd_match.group(1) if cwd_match else None

            # Remove environment_context from text
            text = content
            if "<environment_context>" in text:
                # Remove the entire environment_context block
                text = re.sub(
                    r"<environment_context>.*?</environment_context>\s*",
                    "",
                    text,
                    flags=re.DOTALL,
                )
            return text.strip(), cwd

        texts = []
        cwd = None

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "input_text":
                    block_text = block.get("text", "")

                    # Check for environment_context
                    if "<environment_context>" in block_text:
                        cwd_match = CWD_PATTERN.search(block_text)
                        if cwd_match:
                            cwd = cwd_match.group(1)
                        # Skip this block - it's environment context
                        continue

                    # Skip instruction blocks
                    skip_prefixes = (
                        "<user_instructions>",
                        "<system_instructions>",
                        "<developer_instructions>",
                        "<assistant_instructions>",
                        "<agent_instructions>",
                    )
                    if any(block_text.startswith(prefix) for prefix in skip_prefixes):
                        continue

                    texts.append(block_text)

        return " ".join(texts).strip(), cwd

    @hookimpl
    def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
        """Get the project name from a Codex transcript.

        Reads the transcript to find the cwd from session_meta, then finds
        the git root (tracing upward if needed) and returns its directory name.
        """
        from agentgit import find_git_root

        transcript_abs = transcript_path.resolve()
        codex_sessions_dir = Path.home() / ".codex" / "sessions"

        # Check if this transcript is in ~/.codex/sessions/
        try:
            transcript_abs.relative_to(codex_sessions_dir)
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
        """Extract the cwd from a Codex transcript.

        Codex stores cwd in session_meta record:
        {"type":"session_meta","payload":{"cwd":"/path/to/project",...}}
        """
        try:
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for cwd in session_meta payload
                    if obj.get("type") == "session_meta":
                        payload = obj.get("payload", {})
                        cwd = payload.get("cwd")
                        if cwd:
                            return cwd
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_get_display_name(self, transcript_path: Path) -> str | None:
        """Get display name for a Codex transcript.

        Returns a date-based name like "2025-12-27 session".
        """
        transcript_abs = transcript_path.resolve()
        codex_sessions_dir = Path.home() / ".codex" / "sessions"

        # Only handle Codex transcripts
        try:
            transcript_abs.relative_to(codex_sessions_dir)
        except ValueError:
            return None

        filename = transcript_path.stem  # rollout-2025-12-27T18-07-08-uuid

        # Extract date from rollout filename
        if filename.startswith("rollout-"):
            # rollout-2025-12-27T18-07-08-... -> 2025-12-27
            match = re.match(r"rollout-(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})", filename)
            if match:
                date = match.group(1)
                hour = match.group(2)
                minute = match.group(3)
                return f"{date} {hour}:{minute}"

        return filename

    @hookimpl
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Discover Codex transcripts.

        Codex stores all sessions in ~/.codex/sessions/YYYY/MM/DD/
        Unlike Claude Code, there's no per-project organization in the
        filesystem, but each transcript contains a cwd in session_meta.
        We filter by comparing the transcript's cwd git root to project_path.

        Args:
            project_path: Path to the project. If None, returns all transcripts.
        """
        from agentgit import find_git_root

        codex_sessions_dir = Path.home() / ".codex" / "sessions"
        if not codex_sessions_dir.exists():
            return []

        # Find all rollout-*.jsonl files recursively
        all_transcripts = []
        for jsonl_file in codex_sessions_dir.glob("**/*.jsonl"):
            if jsonl_file.name.startswith("rollout-"):
                all_transcripts.append(jsonl_file)

        if project_path is None:
            # Return all transcripts
            transcripts = all_transcripts
        else:
            # Filter by project - compare git roots
            project_path = project_path.resolve()
            project_git_root = find_git_root(project_path)
            target_root = project_git_root or project_path

            transcripts = []
            for jsonl_file in all_transcripts:
                cwd = self._extract_cwd_from_transcript(jsonl_file)
                if cwd:
                    transcript_git_root = find_git_root(cwd)
                    transcript_root = transcript_git_root or Path(cwd)
                    if transcript_root.resolve() == target_root:
                        transcripts.append(jsonl_file)

        # Sort by modification time, most recent first
        transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return transcripts
