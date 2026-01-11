"""Git repository builder for agentgit."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from agentgit.core import (
    FileOperation,
    OperationType,
    Prompt,
    Transcript,
    TranscriptEntry,
)
from agentgit.plugins import get_configured_plugin_manager

if TYPE_CHECKING:
    from agentgit.enhance import EnhanceConfig

logger = logging.getLogger(__name__)


@dataclass
class CommitMetadata:
    """Metadata extracted from a commit message."""

    tool_id: str | None = None
    timestamp: str | None = None
    operation: str | None = None
    file_path: str | None = None
    prompt_id: str | None = None
    source_commit: str | None = None
    entry_id: str | None = None


def format_git_date(iso_timestamp: str) -> str:
    """Format ISO 8601 timestamp for git environment variables.

    Git expects dates in format: "YYYY-MM-DD HH:MM:SS +0000"

    Args:
        iso_timestamp: ISO 8601 timestamp (e.g., "2025-01-15T14:30:45Z")

    Returns:
        Git-formatted date string.
    """
    from datetime import datetime

    # Parse ISO 8601 timestamp
    # Handle both Z suffix and timezone offsets
    if iso_timestamp.endswith('Z'):
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime.fromisoformat(iso_timestamp)

    # Format for git: "YYYY-MM-DD HH:MM:SS +HHMM"
    # Use strftime with timezone
    return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def parse_commit_trailers(message: str) -> CommitMetadata:
    """Parse git trailers from a commit message.

    Args:
        message: The full commit message.

    Returns:
        CommitMetadata with extracted values.
    """
    metadata = CommitMetadata()

    # Parse trailers (Key: Value format)
    trailer_patterns = {
        "tool_id": r"^Tool-Id:\s*(.+)$",
        "timestamp": r"^Timestamp:\s*(.+)$",
        "operation": r"^Operation:\s*(.+)$",
        "file_path": r"^File:\s*(.+)$",
        "prompt_id": r"^Prompt-Id:\s*(.+)$",
        "source_commit": r"^Source-Commit:\s*(.+)$",
        "entry_id": r"^Entry-Id:\s*(.+)$",
    }

    for field, pattern in trailer_patterns.items():
        match = re.search(pattern, message, re.MULTILINE)
        if match:
            setattr(metadata, field, match.group(1).strip())

    return metadata


def get_processed_operations(repo: Repo) -> set[str]:
    """Get the set of already-processed operation identifiers from a repo.

    Uses Tool-Id from commit trailers to identify processed operations.
    Falls back to Timestamp if Tool-Id is not available.

    Args:
        repo: The git repository to scan.

    Returns:
        Set of operation identifiers (tool_id or timestamp).
    """
    processed = set()

    try:
        # Use --all to scan commits from all branches, not just current branch.
        # This is critical because session branches contain the actual operation
        # commits, while main may only have merges or be empty.
        for commit in repo.iter_commits("--all"):
            metadata = parse_commit_trailers(commit.message)
            if metadata.tool_id:
                processed.add(metadata.tool_id)
            elif metadata.timestamp:
                # Fallback to timestamp if no tool_id
                processed.add(f"ts:{metadata.timestamp}")
    except ValueError:
        # Empty repository has no commits
        pass
    except GitCommandError as e:
        logger.warning("Failed to read commits from repository: %s", e)

    return processed


def format_commit_message(operation: FileOperation) -> str:
    """Format a rich commit message for a file operation.

    Structure:
    - Subject line: contextual summary (from previous assistant message) or operation
    - Blank line
    - Context: previous assistant message (the explanation before tool call)
    - Blank line
    - Thinking: the reasoning that preceded the tool call
    - Blank line
    - User Prompt: truncated to 200 chars
    - Blank line
    - Git trailers for machine parsing

    Args:
        operation: The file operation to format.
    """
    # Subject: prefer contextual summary from previous assistant message
    subject = None
    if operation.assistant_context:
        subject = operation.assistant_context.contextual_summary

    # Fallback to operation-based subject
    if not subject:
        op_verb = {
            OperationType.WRITE: "Create",
            OperationType.EDIT: "Edit",
            OperationType.DELETE: "Delete",
        }.get(operation.operation_type, "Modify")
        subject = f"{op_verb} {operation.filename}"

    body_parts = []

    # Context: previous assistant message (the explanation)
    if operation.assistant_context and operation.assistant_context.previous_message_text:
        body_parts.append(f"Context:\n{operation.assistant_context.previous_message_text}")

    # Thinking: the reasoning that preceded the tool call
    if operation.assistant_context and operation.assistant_context.thinking:
        body_parts.append(f"Thinking:\n{operation.assistant_context.thinking}")

    # User prompt (truncated to 200 chars)
    if operation.prompt:
        prompt_text = operation.prompt.text
        if len(prompt_text) > 200:
            prompt_text = prompt_text[:197] + "..."
        body_parts.append(f"User Prompt: {prompt_text}")

    body = "\n\n".join(body_parts) if body_parts else ""

    # Trailers
    trailers = []
    if operation.prompt:
        trailers.append(f"Prompt-Id: {operation.prompt.prompt_id}")
    trailers.append(f"Operation: {operation.operation_type.value}")
    trailers.append(f"File: {operation.file_path}")
    trailers.append(f"Timestamp: {operation.timestamp}")
    if operation.tool_id:
        trailers.append(f"Tool-Id: {operation.tool_id}")

    trailers_str = "\n".join(trailers)

    if body:
        return f"{subject}\n\n{body}\n\n{trailers_str}"
    else:
        return f"{subject}\n\n{trailers_str}"


def normalize_file_paths(operations: list[FileOperation]) -> tuple[str, dict[str, str]]:
    """Normalize file paths by finding common prefix.

    Returns:
        Tuple of (common_prefix, path_mapping) where path_mapping
        maps original paths to relative paths.
    """
    paths = [op.file_path for op in operations if op.file_path]
    if not paths:
        return "", {}

    # Deduplicate paths while preserving order
    unique_paths = list(dict.fromkeys(paths))

    # Normalize all paths to absolute to handle mix of absolute/relative
    abs_paths = [str(Path(p).resolve()) for p in unique_paths]

    if len(unique_paths) == 1:
        # Single unique file - use its parent as prefix
        common_prefix = str(Path(abs_paths[0]).parent)
    else:
        common_prefix = os.path.commonpath(abs_paths)
        # If common_prefix is a file path (not a directory), use its parent
        # This happens when one path is a prefix of another
        if not os.path.isdir(common_prefix) and common_prefix in abs_paths:
            common_prefix = str(Path(common_prefix).parent)

    # If common_prefix is root or too short, use a more sensible prefix
    # This prevents relative paths like "Users/btucker/..." which are invalid
    if common_prefix == "/" or len(common_prefix) < 3:
        # Find a better common prefix by looking for common directory components
        # Use the longest common path that's at least 2 levels deep
        path_parts = [Path(p).parts for p in abs_paths]
        min_len = min(len(parts) for parts in path_parts)
        common_parts = []
        for i in range(min_len):
            if all(parts[i] == path_parts[0][i] for parts in path_parts):
                common_parts.append(path_parts[0][i])
            else:
                break
        if len(common_parts) >= 2:
            common_prefix = str(Path(*common_parts))
        else:
            # Fall back to using the first path's parent directory
            common_prefix = str(Path(abs_paths[0]).parent)

    path_mapping = {}
    for orig_path, abs_path in zip(unique_paths, abs_paths):
        rel_path = os.path.relpath(abs_path, common_prefix)
        # Sanity check: if rel_path still looks like an absolute path (starts with Users/ etc.)
        # use the full path structure from a reasonable point
        if rel_path.startswith("Users/") or rel_path.startswith("home/") or rel_path.startswith(".."):
            # Use path relative to the file's own project-like root
            # Look for common project markers like src/, tests/, etc.
            path_obj = Path(abs_path)
            for i, part in enumerate(path_obj.parts):
                if part in ("src", "tests", "lib", "pkg", "app"):
                    rel_path = str(Path(*path_obj.parts[i:]))
                    break
            else:
                # Last resort: use just parent/filename
                rel_path = str(Path(path_obj.parent.name) / path_obj.name)
        path_mapping[orig_path] = rel_path

    return common_prefix, path_mapping


class GitRepoBuilder:
    """Builds a git repository from file operations."""

    def __init__(
        self,
        output_dir: Path | None = None,
        enhance_config: Optional["EnhanceConfig"] = None,
        code_repo: Path | None = None,
        session_branch_name: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize the builder.

        Args:
            output_dir: Directory for the git repo. If None, creates a temp dir.
            enhance_config: Optional AI configuration for generating commit messages.
            code_repo: Path to the code repository. If provided, sets up git alternates
                to share objects with the code repo. If None, auto-detects from cwd.
            session_branch_name: Optional branch name for session-based workflow. If provided,
                all commits go to this branch (never merged to main).
            session_id: Optional session identifier to include in commit trailers.
        """
        self.output_dir = output_dir
        self.enhance_config = enhance_config
        self.code_repo = code_repo
        self.session_branch_name = session_branch_name
        self.session_id = session_id
        self.repo: Repo | None = None
        self.path_mapping: dict[str, str] = {}
        self.file_states: dict[str, str] = {}
        self._processed_ops: set[str] = set()

    def _get_code_repo_path(self) -> Path | None:
        """Get the code repository path.

        Returns:
            Path to code repo, or None if not found/not a git repo.
        """
        if self.code_repo:
            return self.code_repo

        # Auto-detect from current directory
        from agentgit import find_git_root

        return find_git_root()

    def _get_code_repo_author(self) -> tuple[str, str]:
        """Get the author name and email from the code repository.

        Returns:
            Tuple of (name, email) from code repo git config, or defaults.
        """
        code_repo_path = self._get_code_repo_path()
        if not code_repo_path:
            return ("User", "user@local")

        try:
            code_repo = Repo(code_repo_path)
            reader = code_repo.config_reader()
            name = reader.get_value("user", "name", default="User")
            email = reader.get_value("user", "email", default="user@local")
            return (name, email)
        except Exception:
            return ("User", "user@local")

    def _setup_alternates(self, code_repo: Path) -> None:
        """Set up git alternates to share objects with code repo.

        Args:
            code_repo: Path to the code repository.
        """
        if not self.repo:
            return

        # Path to code repo's objects directory
        code_objects = code_repo / ".git" / "objects"
        if not code_objects.exists():
            logger.warning(
                "Code repo objects directory not found: %s", code_objects
            )
            return

        # Create alternates file
        alternates_file = self.output_dir / ".git" / "objects" / "info" / "alternates"
        alternates_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the path to code repo's objects
        alternates_file.write_text(str(code_objects) + "\n")
        logger.info("Set up git alternates to share objects with: %s", code_repo)

    def _is_operation_processed(self, operation: FileOperation) -> bool:
        """Check if an operation has already been processed."""
        if operation.tool_id and operation.tool_id in self._processed_ops:
            return True
        if operation.timestamp and f"ts:{operation.timestamp}" in self._processed_ops:
            return True
        return False

    def _get_operation_id(self, operation: FileOperation) -> str:
        """Get the identifier for an operation."""
        if operation.tool_id:
            return operation.tool_id
        return f"ts:{operation.timestamp}"

    def _setup_repo(self, incremental: bool) -> bool:
        """Initialize or load the git repository.

        Args:
            incremental: If True and repo exists, load processed operations.

        Returns:
            True if existing repo was found, False if new repo was created.
        """
        if self.output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="agentgit_"))
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.repo = Repo(self.output_dir)
            if incremental:
                self._processed_ops = get_processed_operations(self.repo)
                self._load_file_states_from_repo()
            return True
        except InvalidGitRepositoryError:
            self.repo = Repo.init(self.output_dir)

            # Set up git alternates to share objects with code repo
            code_repo_path = self._get_code_repo_path()
            if code_repo_path:
                self._setup_alternates(code_repo_path)

            # Disable commit signing for agentgit-created repos
            with self.repo.config_writer() as config:
                config.set_value("commit", "gpgsign", "false")
                # Store code repo path for reference
                if code_repo_path:
                    config.set_value("agentgit", "coderepo", str(code_repo_path))

            return False

    def build_from_prompts(
        self,
        transcript: Transcript,
        author_name: str = "Agent",
        author_email: str = "agent@local",
        incremental: bool = True,
    ) -> tuple[Repo, Path, dict[str, str]]:
        """Build git repo with one commit per user prompt.

        This is the simplified build method that creates exactly one commit
        per user prompt, with all tool calls and reasoning rendered as markdown
        in the commit body.

        Args:
            transcript: The parsed transcript.
            author_name: Name for agent commits.
            author_email: Email for agent commits.
            incremental: If True, skip already-processed prompts.

        Returns:
            Tuple of (repo, repo_path, path_mapping).
        """
        self._setup_repo(incremental)

        # Get author info from plugin if available, but only use it
        # when caller is using default values (don't override explicit params)
        pm = get_configured_plugin_manager()
        plugin_author_info = pm.hook.agentgit_get_author_info(transcript=transcript)

        if plugin_author_info:
            if author_name == "Agent":
                author_name = plugin_author_info.get("name", author_name)
            if author_email == "agent@local":
                author_email = plugin_author_info.get("email", author_email)

        with self.repo.config_writer() as config:
            config.set_value("user", "name", author_name)
            config.set_value("user", "email", author_email)

        # Collect all operations for path normalization
        all_operations = [op for op in transcript.operations if op.operation_type != OperationType.DELETE]
        _, self.path_mapping = normalize_file_paths(all_operations)

        if not transcript.prompts:
            return self.repo, self.output_dir, self.path_mapping

        # Ensure we have an initial commit on main
        if not self._has_commits():
            self._create_initial_commit()

        # Create or checkout session branch
        if self.session_branch_name:
            if self.session_branch_name in self.repo.heads:
                self.repo.heads[self.session_branch_name].checkout()
            else:
                main_ref = self._get_main_branch_commit()
                session_branch = self.repo.create_head(self.session_branch_name, main_ref)
                session_branch.checkout()
                logger.info("Created session branch: %s", self.session_branch_name)

        # Group entries by prompt
        prompt_groups = self._group_entries_by_prompt(transcript)

        prev_assistant_text = ""
        for prompt, entries in prompt_groups:
            # Check if this prompt is already processed (by checking its operations)
            ops = self._collect_operations_from_entries(entries, transcript)
            if incremental and ops and all(self._is_operation_processed(op) for op in ops):
                # Update prev_assistant_text even for skipped prompts
                prev_assistant_text = self._get_final_assistant_text(entries)
                continue

            # Get commit subject (smart handling for short prompts)
            subject = self._get_prompt_commit_subject(prompt, prev_assistant_text)

            # Render body as markdown
            body = self._render_prompt_response_markdown(prompt, entries)

            # Apply all file operations from these entries
            files_changed = []
            for op in ops:
                if self._apply_operation_no_commit(op):
                    if op.operation_type != OperationType.DELETE:
                        rel_path = self.path_mapping.get(op.file_path)
                        if rel_path:
                            files_changed.append(rel_path)

            # Add trailers
            trailers = [f"Prompt-Id: {prompt.prompt_id}"]
            trailers.append(f"Timestamp: {prompt.timestamp}")
            for op in ops:
                if op.tool_id:
                    trailers.append(f"Tool-Id: {op.tool_id}")

            # Build full commit message
            message = f"{subject}\n\n{body}\n\n" + "\n".join(trailers)

            # Stage changed files
            for rel_path in files_changed:
                # Skip paths that look like absolute paths without leading /
                # This happens when normalize_file_paths gets / as the common prefix
                if rel_path.startswith("Users/") or rel_path.startswith("home/"):
                    logger.warning("Skipping invalid relative path: %s", rel_path)
                    continue
                try:
                    self.repo.index.add([rel_path])
                except GitCommandError:
                    pass
                except FileNotFoundError:
                    logger.warning("File not found when staging: %s", rel_path)

            # Create commit
            try:
                if files_changed:
                    self._commit_with_date(message, prompt.timestamp)
                else:
                    self._commit_with_date(message, prompt.timestamp, allow_empty=True)
            except GitCommandError as e:
                logger.warning("Failed to commit prompt: %s", e)

            # Mark operations as processed
            for op in ops:
                self._processed_ops.add(self._get_operation_id(op))

            # Track assistant text for next iteration
            prev_assistant_text = self._get_final_assistant_text(entries)

        # Create session ref if session_id is provided
        if self.session_id and self.session_branch_name:
            session_ref_path = self.output_dir / ".git" / "refs" / "sessions" / self.session_id
            session_ref_path.parent.mkdir(parents=True, exist_ok=True)
            session_ref_path.write_text(f"ref: refs/heads/{self.session_branch_name}\n")
            logger.info("Created session ref: %s -> %s", f"refs/sessions/{self.session_id}", self.session_branch_name)

        return self.repo, self.output_dir, self.path_mapping

    def _group_entries_by_prompt(
        self, transcript: Transcript
    ) -> list[tuple[Prompt, list[TranscriptEntry]]]:
        """Group transcript entries by their triggering prompt.

        Returns a list of (prompt, entries) tuples where entries includes
        all assistant responses until the next user prompt.
        """
        groups: list[tuple[Prompt, list[TranscriptEntry]]] = []
        current_prompt: Prompt | None = None
        current_entries: list[TranscriptEntry] = []

        prompt_timestamps = {p.timestamp for p in transcript.prompts}

        for entry in transcript.entries:
            # Check if this is a user prompt (not tool result or meta)
            if entry.entry_type == "user" and entry.timestamp in prompt_timestamps:
                # Save previous group if we have one
                if current_prompt is not None:
                    groups.append((current_prompt, current_entries))

                # Start new group - find the Prompt object for this entry
                for p in transcript.prompts:
                    if p.timestamp == entry.timestamp:
                        current_prompt = p
                        break

                current_entries = [entry]
            elif current_prompt is not None:
                # Add to current group
                current_entries.append(entry)

        # Don't forget the last group
        if current_prompt is not None and current_entries:
            groups.append((current_prompt, current_entries))

        return groups

    def _get_prompt_commit_subject(self, prompt: Prompt, prev_assistant_text: str) -> str:
        """Get commit subject line for a prompt.

        If the prompt is short (<50 chars) and preceded by assistant text,
        uses the final paragraph of the assistant text instead.
        """
        prompt_text = prompt.text.strip()

        # Short prompt preceded by assistant? Use final paragraph of assistant
        if len(prompt_text) < 50 and prev_assistant_text:
            # Get final paragraph
            paragraphs = prev_assistant_text.strip().split('\n\n')
            final_para = paragraphs[-1].strip()
            # Clean up and truncate
            subject = final_para.replace('\n', ' ')[:80]
            if len(final_para) > 80:
                subject = subject.rsplit(' ', 1)[0] + "..."
            return subject

        # Otherwise use first line of prompt
        first_line = prompt_text.split('\n')[0]
        if len(first_line) > 80:
            return first_line[:77] + "..."
        return first_line

    def _render_prompt_response_markdown(
        self, prompt: Prompt, entries: list[TranscriptEntry]
    ) -> str:
        """Render prompt and response entries as markdown for commit body."""
        pm = get_configured_plugin_manager()
        parts = []

        # Prompt section
        parts.append(f"## Prompt\n\n{prompt.text}")

        # Response section
        parts.append("## Response")

        for entry in entries:
            if entry.entry_type != "assistant":
                continue

            content = entry.message.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        parts.append(f"### Thinking\n\n{thinking}")

                elif block_type == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)

                elif block_type == "tool_use":
                    # Use plugin hook to format the tool call
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    rendered = pm.hook.agentgit_format_tool(
                        tool_name=tool_name, tool_input=tool_input
                    )
                    if rendered:
                        parts.append(rendered)

        return "\n\n".join(parts)

    def _collect_operations_from_entries(
        self, entries: list[TranscriptEntry], transcript: Transcript
    ) -> list[FileOperation]:
        """Collect all file operations from a list of entries."""
        # Build tool_id -> operation mapping
        tool_id_to_op: dict[str, FileOperation] = {}
        for op in transcript.operations:
            if op.tool_id:
                tool_id_to_op[op.tool_id] = op

        operations = []
        for entry in entries:
            if entry.entry_type != "assistant":
                continue

            content = entry.message.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue

                tool_id = block.get("id", "")
                if tool_id in tool_id_to_op:
                    operations.append(tool_id_to_op[tool_id])

        return operations

    def _get_final_assistant_text(self, entries: list[TranscriptEntry]) -> str:
        """Get the final assistant text from a list of entries.

        Used to provide context for short user prompts like "yes" or "do it".
        """
        final_text = ""

        for entry in entries:
            if entry.entry_type != "assistant":
                continue

            content = entry.message.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        final_text = text

        return final_text

    def _has_commits(self) -> bool:
        """Check if the repository has any commits."""
        try:
            _ = self.repo.head.commit
            return True
        except ValueError:
            return False

    def _get_main_branch_commit(self):
        """Get the commit from the main branch (not HEAD).

        This is important when creating new session branches - they should
        branch from main, not from whatever branch HEAD currently points to.
        Otherwise, if HEAD is on session A's branch, new session B would
        incorrectly inherit all of session A's commits.

        Returns:
            The commit object from the main branch.
        """
        # Try 'main' first, then 'master' for compatibility
        if "main" in self.repo.heads:
            return self.repo.heads.main.commit
        elif "master" in self.repo.heads:
            return self.repo.heads.master.commit
        else:
            # Fall back to HEAD if no main/master branch exists
            # (this handles the initial setup case)
            return self.repo.head.commit

    def _create_initial_commit(self, timestamp: str = "1970-01-01T00:00:00Z") -> None:
        """Create an initial empty commit.

        Args:
            timestamp: ISO 8601 timestamp for the commit. Defaults to epoch.
        """
        self._commit_with_date("Initial commit", timestamp, allow_empty=True)

    def _apply_operation_no_commit(self, operation: FileOperation) -> bool:
        """Apply a single operation without creating a commit.

        Returns:
            True if file was changed, False otherwise.
        """
        if operation.operation_type == OperationType.DELETE:
            return self._apply_delete_no_commit(operation)
        elif operation.operation_type == OperationType.WRITE:
            return self._apply_write_no_commit(operation)
        elif operation.operation_type == OperationType.EDIT:
            return self._apply_edit_no_commit(operation)
        return False

    def _apply_write_no_commit(self, operation: FileOperation) -> bool:
        """Apply a write operation without committing."""
        rel_path = self.path_mapping.get(operation.file_path)

        # Fix paths that look like absolute paths without leading /
        # This happens when normalize_file_paths gets / as the common prefix
        if rel_path and (rel_path.startswith("Users/") or rel_path.startswith("home/")):
            rel_path = Path(operation.file_path).name
            self.path_mapping[operation.file_path] = rel_path
            logger.warning("Fixed invalid relative path, using basename: %s", rel_path)

        if not rel_path:
            # Path not in mapping - compute a reasonable relative path
            # Use the basename to avoid writing outside the repo
            rel_path = Path(operation.file_path).name
            self.path_mapping[operation.file_path] = rel_path
            logger.warning("Path not in mapping, using basename: %s -> %s", operation.file_path, rel_path)

        full_path = self.output_dir / rel_path

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(operation.content or "")

        self.file_states[operation.file_path] = operation.content or ""
        return True

    def _ensure_initial_state_for_edit(self, operation: FileOperation) -> str | None:
        """Ensure file exists and return its content for editing.

        Args:
            operation: The edit operation.

        Returns:
            File content if available, None if file cannot be prepared.
        """
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        if full_path.exists():
            return full_path.read_text()
        elif operation.file_path in self.file_states:
            return self.file_states[operation.file_path]
        elif operation.original_content:
            # Create file with original content (will be included in next commit)
            content = operation.original_content
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.file_states[operation.file_path] = content
            return content
        else:
            return None

    def _apply_edit_no_commit(self, operation: FileOperation) -> bool:
        """Apply an edit operation without committing."""
        rel_path = self.path_mapping.get(operation.file_path)

        # Fix paths that look like absolute paths without leading /
        if rel_path and (rel_path.startswith("Users/") or rel_path.startswith("home/")):
            rel_path = Path(operation.file_path).name
            self.path_mapping[operation.file_path] = rel_path
            logger.warning("Fixed invalid relative path, using basename: %s", rel_path)

        if not rel_path:
            # Path not in mapping - compute a reasonable relative path
            rel_path = Path(operation.file_path).name
            self.path_mapping[operation.file_path] = rel_path
            logger.warning("Path not in mapping, using basename: %s -> %s", operation.file_path, rel_path)

        full_path = self.output_dir / rel_path

        # Get or create initial content
        content = self._ensure_initial_state_for_edit(operation)
        if content is None:
            return False

        old_str = operation.old_string or ""
        new_str = operation.new_string or ""

        if old_str in content:
            if operation.replace_all:
                content = content.replace(old_str, new_str)
            else:
                content = content.replace(old_str, new_str, 1)

            full_path.write_text(content)
            self.file_states[operation.file_path] = content
            return True

        return False

    def _apply_delete_no_commit(self, operation: FileOperation) -> bool:
        """Apply a delete operation without committing."""
        delete_path = operation.file_path
        files_removed = False

        if operation.recursive:
            delete_prefix = delete_path.rstrip("/") + "/"
            for orig_path, rel_path in self.path_mapping.items():
                if orig_path.startswith(delete_prefix) or orig_path == delete_path:
                    file_abs = self.output_dir / rel_path
                    if file_abs.exists():
                        file_abs.unlink()
                        try:
                            self.repo.index.remove([rel_path])
                        except GitCommandError:
                            pass
                        files_removed = True
        else:
            if delete_path in self.path_mapping:
                rel_path = self.path_mapping[delete_path]
                file_abs = self.output_dir / rel_path
                if file_abs.exists():
                    file_abs.unlink()
                    try:
                        self.repo.index.remove([rel_path])
                    except GitCommandError:
                        pass
                    files_removed = True

        return files_removed

    def _commit_with_date(self, commit_msg: str, timestamp: str, allow_empty: bool = False) -> None:
        """Create a commit with a specific timestamp.

        Args:
            commit_msg: The commit message.
            timestamp: ISO 8601 timestamp to use for commit date.
            allow_empty: Whether to allow empty commits.
        """
        git_date = format_git_date(timestamp)
        old_author_date = os.environ.get('GIT_AUTHOR_DATE')
        old_committer_date = os.environ.get('GIT_COMMITTER_DATE')
        try:
            os.environ['GIT_AUTHOR_DATE'] = git_date
            os.environ['GIT_COMMITTER_DATE'] = git_date
            if allow_empty:
                self.repo.git.commit("--allow-empty", "-m", commit_msg)
            else:
                self.repo.index.commit(commit_msg)
        finally:
            # Restore old env vars
            if old_author_date is not None:
                os.environ['GIT_AUTHOR_DATE'] = old_author_date
            else:
                os.environ.pop('GIT_AUTHOR_DATE', None)
            if old_committer_date is not None:
                os.environ['GIT_COMMITTER_DATE'] = old_committer_date
            else:
                os.environ.pop('GIT_COMMITTER_DATE', None)

    def _load_file_states_from_repo(self) -> None:
        """Load current file states from existing repo."""
        if not self.repo or self.repo.bare:
            return

        try:
            tree = self.repo.head.commit.tree
        except ValueError:
            # Empty repository has no HEAD commit
            return

        for item in tree.traverse():
            if item.type == "blob":
                rel_path = item.path
                full_path = self.output_dir / rel_path
                if full_path.exists():
                    try:
                        self.file_states[str(full_path)] = full_path.read_text()
                    except (OSError, UnicodeDecodeError) as e:
                        logger.debug("Could not read file %s: %s", full_path, e)

