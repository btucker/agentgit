"""Claude Code JSONL format parser plugin for agentgit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

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

# Format identifier for Claude Code JSONL transcripts
FORMAT_CLAUDE_CODE_JSONL = "claude_code_jsonl"

# Tools to skip when rendering commit messages
# These are either implicit in the git diff (file operations) or internal mechanics
SKIP_TOOLS = {
    "Read", "Write", "Edit", "Glob", "Grep", "LSP",
    "NotebookEdit", "NotebookRead",
    "TodoWrite", "AskUserQuestion",
}


def get_last_timestamp_from_jsonl(file_path: Path) -> float | None:
    """Extract the last timestamp from a JSONL file.

    Reads the last 8KB of the file to find the most recent entry's timestamp.
    This is more efficient than reading the entire file and more accurate than
    using file modification time.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Unix timestamp (seconds since epoch) of the last entry, or None if not found.
    """
    try:
        with open(file_path, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            if file_size == 0:
                return None

            # Read last 8KB (or entire file if smaller)
            chunk_size = min(8192, file_size)
            f.seek(file_size - chunk_size)
            chunk = f.read(chunk_size).decode('utf-8', errors='ignore')

            # Split into lines and process in reverse
            # Skip first line as it's likely truncated (unless we read the whole file)
            lines = chunk.split('\n')
            if chunk_size < file_size and len(lines) > 1:
                lines = lines[1:]  # Discard potentially truncated first line

            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    timestamp_str = obj.get("timestamp")
                    if timestamp_str:
                        # Parse ISO timestamp and convert to unix timestamp
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        return dt.timestamp()
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        pass

    return None


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

                        # Skip non-interactive sessions (from claude-cli --print mode)
                        # These are created when tools like llm make API calls
                        if i == 0 and obj.get("type") == "queue-operation" and obj.get("operation") == "dequeue":
                            return None

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
                    # Skip system-generated prompts (commands, continuations, etc.)
                    if text and not self._is_system_prompt(text):
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
        """Enrich operation with prompt and assistant context.

        Captures two levels of context:
        1. Immediate context: thinking/text in the same message as the tool_use
        2. Previous message context: the assistant message before the one with tool_use
           (often the explanatory message that describes the approach)
        """
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

        # Find assistant context - track both current and previous message
        context = AssistantContext()
        previous_assistant_text: str | None = None

        for entry in transcript.entries:
            if entry.timestamp > operation.timestamp:
                break

            if entry.entry_type == "assistant":
                content = entry.message.get("content", [])
                if isinstance(content, list):
                    # First pass: check if this entry contains our target tool_use
                    # and collect the text from this entry
                    entry_text: str | None = None
                    entry_thinking: str | None = None
                    has_target_tool = False

                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking":
                                entry_thinking = block.get("thinking", "")
                            elif block.get("type") == "text":
                                entry_text = block.get("text", "")
                            elif block.get("type") == "tool_use":
                                if block.get("id") == operation.tool_id:
                                    has_target_tool = True

                    if has_target_tool:
                        # Found the entry with our tool_use
                        # Use thinking/text from THIS entry for immediate context
                        context.thinking = entry_thinking
                        context.text = entry_text
                        context.timestamp = entry.timestamp
                        # Use text from PREVIOUS assistant entry for contextual summary
                        context.previous_message_text = previous_assistant_text
                        operation.assistant_context = context
                        return operation
                    else:
                        # This is a previous assistant entry, save its text
                        if entry_text:
                            previous_assistant_text = entry_text

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

    def _is_system_prompt(self, text: str) -> bool:
        """Check if a prompt is a system-generated message that shouldn't become a commit.

        This filters out:
        - Session continuation messages (injected when resuming from context overflow)
        - Slash command executions (like /clear, /help)
        - Command output markers

        Returns True if the prompt should be skipped.
        """
        # Continuation prompts (session resumption)
        continuation_markers = [
            "This session is being continued from a previous conversation",
            "conversation that ran out of context",
            "The conversation is summarized below",
        ]

        for marker in continuation_markers:
            if marker in text:
                return True

        # Slash command executions (wrapped in <command-name> tags)
        if text.strip().startswith("<command-name>"):
            return True

        # Command output markers (local command stdout - these are CLI feedback, not user prompts)
        if text.strip().startswith("<local-command-stdout>"):
            return True

        # Task notifications (background task completions)
        if text.strip().startswith("<task-notification>"):
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

        Returns the filename as-is without truncation.
        """
        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Only handle Claude Code transcripts
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        return transcript_path.name

    @hookimpl
    def agentgit_get_last_timestamp(self, transcript_path: Path) -> float | None:
        """Get the last timestamp from a Claude Code transcript.

        Reads the end of the JSONL file to find the last entry's timestamp
        instead of using file modification time.
        """
        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Only handle Claude Code transcripts
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        return get_last_timestamp_from_jsonl(transcript_path)

    @hookimpl
    def agentgit_get_author_info(self, transcript: Transcript) -> dict[str, str] | None:
        """Get git author information for Claude Code transcripts.

        Returns Claude as the author. In the future, this could extract
        specific model information from the transcript if available.
        """
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return None

        # Default author for Claude Code transcripts
        # TODO: Extract actual model info from transcript when available
        return {
            "name": "Claude",
            "email": "claude@anthropic.com",
        }

    @hookimpl
    def agentgit_format_tool(self, tool_name: str, tool_input: dict) -> str | None:
        """Format a tool call as markdown for commit messages.

        Renders Claude Code tool calls appropriately:
        - Bash: as ```bash code blocks
        - Task: as ### Agent: heading
        - WebFetch/WebSearch: as markdown links
        - File ops and internal tools: skipped (implicit in diff or not useful)
        """
        if tool_name in SKIP_TOOLS:
            return None

        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            return f"```bash\n{cmd}\n```"

        if tool_name == "Task":
            desc = tool_input.get("description", "")
            return f"### Agent: {desc}"

        if tool_name == "WebFetch":
            url = tool_input.get("url", "")
            return f"[{url}]({url})"

        if tool_name == "WebSearch":
            query = tool_input.get("query", "")
            return f"[Search: {query}](https://google.com/search?q={quote_plus(query)})"

        return None  # Skip unknown tools


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
                    # Skip empty files (0 bytes)
                    if jsonl_file.stat().st_size == 0:
                        continue
                    # Skip non-interactive API call sessions (queue-operation dequeue)
                    if self._is_api_call_session(jsonl_file):
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
                # Skip empty files (0 bytes)
                if jsonl_file.stat().st_size == 0:
                    continue
                # Skip non-interactive API call sessions (queue-operation dequeue)
                if self._is_api_call_session(jsonl_file):
                    continue
                transcripts.append(jsonl_file)

        # Sort by modification time, most recent first
        transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return transcripts

    def _is_api_call_session(self, path: Path) -> bool:
        """Check if a transcript file is a non-interactive API call session.

        These are created when tools like llm use claude-cli with --print mode.
        They start with a queue-operation dequeue entry.

        Args:
            path: Path to the transcript file.

        Returns:
            True if this is an API call session, False otherwise.
        """
        try:
            with open(path, encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                obj = json.loads(first_line)
                return (
                    obj.get("type") == "queue-operation"
                    and obj.get("operation") == "dequeue"
                )
        except Exception:
            return False
