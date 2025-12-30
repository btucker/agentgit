"""Git repository builder for agentgit."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from git import Repo
from git.exc import InvalidGitRepositoryError

from agentgit.core import FileOperation, OperationType, SourceCommit


@dataclass
class CommitMetadata:
    """Metadata extracted from a commit message."""

    tool_id: str | None = None
    timestamp: str | None = None
    operation: str | None = None
    file_path: str | None = None
    prompt_id: str | None = None
    source_commit: str | None = None


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
        for commit in repo.iter_commits():
            metadata = parse_commit_trailers(commit.message)
            if metadata.tool_id:
                processed.add(metadata.tool_id)
            elif metadata.timestamp:
                # Fallback to timestamp if no tool_id
                processed.add(f"ts:{metadata.timestamp}")
    except Exception:
        pass

    return processed


def get_last_processed_timestamp(repo: Repo) -> str | None:
    """Get the timestamp of the last processed operation.

    Args:
        repo: The git repository to scan.

    Returns:
        ISO timestamp string or None if no operations found.
    """
    try:
        for commit in repo.iter_commits():
            metadata = parse_commit_trailers(commit.message)
            if metadata.timestamp:
                return metadata.timestamp
    except Exception:
        pass

    return None


def format_commit_message(operation: FileOperation) -> str:
    """Format a rich commit message for a file operation.

    Structure:
    - Subject line: operation type and file
    - Blank line
    - User prompt (the "why") - full text, no truncation
    - Blank line
    - Assistant context if available
    - Blank line
    - Git trailers for machine parsing
    """
    op_verb = {
        OperationType.WRITE: "Create",
        OperationType.EDIT: "Edit",
        OperationType.DELETE: "Delete",
    }.get(operation.operation_type, "Modify")

    subject = f"{op_verb} {operation.filename}"

    body_parts = []

    # User prompt (the "why") - full text, no truncation
    if operation.prompt:
        body_parts.append(f"Prompt #{operation.prompt.short_id}:\n{operation.prompt.text}")

    # Assistant context (the reasoning)
    if operation.assistant_context and operation.assistant_context.summary:
        context = operation.assistant_context.summary
        if context:
            body_parts.append(f"Context:\n{context}")

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

    if len(unique_paths) == 1:
        # Single unique file - use its parent as prefix
        common_prefix = str(Path(unique_paths[0]).parent)
    else:
        common_prefix = os.path.commonpath(unique_paths)
        # If common_prefix is a file path (not a directory), use its parent
        # This happens when one path is a prefix of another
        if not os.path.isdir(common_prefix) and common_prefix in unique_paths:
            common_prefix = str(Path(common_prefix).parent)

    path_mapping = {}
    for path in unique_paths:
        rel_path = os.path.relpath(path, common_prefix)
        path_mapping[path] = rel_path

    return common_prefix, path_mapping


def get_commits_in_timeframe(
    repo_path: Path,
    start_time: str,
    end_time: str,
) -> list[SourceCommit]:
    """Get commits from a repository within a timeframe.

    Args:
        repo_path: Path to the git repository.
        start_time: ISO format start timestamp.
        end_time: ISO format end timestamp.

    Returns:
        List of SourceCommit objects.
    """
    repo = Repo(repo_path)
    commits = []

    for commit in repo.iter_commits():
        commit_time = commit.committed_datetime.isoformat()

        if commit_time < start_time:
            break
        if commit_time > end_time:
            continue

        files_changed = list(commit.stats.files.keys())

        commits.append(
            SourceCommit(
                sha=commit.hexsha,
                message=commit.message.strip(),
                timestamp=commit_time,
                author=commit.author.name,
                author_email=commit.author.email,
                files_changed=files_changed,
            )
        )

    commits.reverse()
    return commits


def merge_timeline(
    operations: list[FileOperation],
    source_commits: list[SourceCommit],
) -> list[Union[FileOperation, SourceCommit]]:
    """Merge operations and commits by timestamp."""
    combined: list[Union[FileOperation, SourceCommit]] = [*operations, *source_commits]
    return sorted(combined, key=lambda x: x.timestamp)


class GitRepoBuilder:
    """Builds a git repository from file operations."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize the builder.

        Args:
            output_dir: Directory for the git repo. If None, creates a temp dir.
        """
        self.output_dir = output_dir
        self.repo: Repo | None = None
        self.path_mapping: dict[str, str] = {}
        self.file_states: dict[str, str] = {}
        self._processed_ops: set[str] = set()

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

    def build(
        self,
        operations: list[FileOperation],
        author_name: str = "Agent",
        author_email: str = "agent@local",
        source_repo: Path | None = None,
        incremental: bool = True,
    ) -> tuple[Repo, Path, dict[str, str]]:
        """Build a git repository from file operations.

        If the output directory already contains a git repository and
        incremental=True, only new operations will be added.

        Args:
            operations: List of FileOperation objects in chronological order.
            author_name: Name for git commits.
            author_email: Email for git commits.
            source_repo: Optional source repository to interleave commits from.
            incremental: If True, skip already-processed operations.

        Returns:
            Tuple of (repo, repo_path, path_mapping).
        """
        if self.output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="agentgit_"))
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if repo already exists
        existing_repo = False
        try:
            self.repo = Repo(self.output_dir)
            existing_repo = True
            if incremental:
                self._processed_ops = get_processed_operations(self.repo)
                # Load current file states from the repo
                self._load_file_states_from_repo()
        except InvalidGitRepositoryError:
            self.repo = Repo.init(self.output_dir)

        with self.repo.config_writer() as config:
            config.set_value("user", "name", author_name)
            config.set_value("user", "email", author_email)

        non_delete_ops = [op for op in operations if op.operation_type != OperationType.DELETE]
        _, self.path_mapping = normalize_file_paths(non_delete_ops)

        # Filter out already-processed operations if incremental
        if incremental and existing_repo:
            operations = [op for op in operations if not self._is_operation_processed(op)]

        if not operations:
            return self.repo, self.output_dir, self.path_mapping

        # Merge with source repo commits if provided
        if source_repo and operations:
            start_time = min(op.timestamp for op in operations)
            end_time = max(op.timestamp for op in operations)
            source_commits = get_commits_in_timeframe(source_repo, start_time, end_time)
            timeline = merge_timeline(operations, source_commits)
        else:
            timeline = list(operations)

        timeline.sort(key=lambda x: x.timestamp)

        for item in timeline:
            if isinstance(item, FileOperation):
                self._apply_operation(item)
                # Track as processed
                self._processed_ops.add(self._get_operation_id(item))
            elif isinstance(item, SourceCommit):
                self._apply_source_commit(item, source_repo)

        return self.repo, self.output_dir, self.path_mapping

    def _load_file_states_from_repo(self) -> None:
        """Load current file states from existing repo."""
        if not self.repo or self.repo.bare:
            return

        try:
            for item in self.repo.head.commit.tree.traverse():
                if item.type == "blob":
                    rel_path = item.path
                    full_path = self.output_dir / rel_path
                    if full_path.exists():
                        self.file_states[str(full_path)] = full_path.read_text()
        except Exception:
            pass

    def _apply_operation(self, operation: FileOperation) -> None:
        """Apply a single operation and create a commit."""
        if operation.operation_type == OperationType.DELETE:
            self._apply_delete(operation)
        elif operation.operation_type == OperationType.WRITE:
            self._apply_write(operation)
        elif operation.operation_type == OperationType.EDIT:
            self._apply_edit(operation)

    def _apply_write(self, operation: FileOperation) -> None:
        """Apply a write operation."""
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(operation.content or "")

        self.file_states[operation.file_path] = operation.content or ""

        self.repo.index.add([rel_path])
        commit_msg = format_commit_message(operation)
        self.repo.index.commit(commit_msg)

    def _apply_edit(self, operation: FileOperation) -> None:
        """Apply an edit operation."""
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        if full_path.exists():
            content = full_path.read_text()
        elif operation.file_path in self.file_states:
            content = self.file_states[operation.file_path]
        elif operation.original_content:
            content = operation.original_content
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.repo.index.add([rel_path])
            self.repo.index.commit("Initial state (pre-session)")
        else:
            return

        old_str = operation.old_string or ""
        new_str = operation.new_string or ""

        if old_str in content:
            if operation.replace_all:
                content = content.replace(old_str, new_str)
            else:
                content = content.replace(old_str, new_str, 1)

            full_path.write_text(content)
            self.file_states[operation.file_path] = content

            self.repo.index.add([rel_path])
            commit_msg = format_commit_message(operation)
            self.repo.index.commit(commit_msg)

    def _apply_delete(self, operation: FileOperation) -> None:
        """Apply a delete operation."""
        is_recursive = operation.replace_all
        delete_path = operation.file_path

        files_to_remove = []

        if is_recursive:
            delete_prefix = delete_path.rstrip("/") + "/"
            for orig_path, rel_path in self.path_mapping.items():
                if orig_path.startswith(delete_prefix) or orig_path == delete_path:
                    file_abs = self.output_dir / rel_path
                    if file_abs.exists():
                        files_to_remove.append((file_abs, rel_path))
        else:
            if delete_path in self.path_mapping:
                rel_path = self.path_mapping[delete_path]
                file_abs = self.output_dir / rel_path
                if file_abs.exists():
                    files_to_remove.append((file_abs, rel_path))

        if files_to_remove:
            for file_abs, rel_path in files_to_remove:
                file_abs.unlink()
                try:
                    self.repo.index.remove([rel_path])
                except Exception:
                    pass

            commit_msg = format_commit_message(operation)
            try:
                self.repo.index.commit(commit_msg)
            except Exception:
                pass

    def _apply_source_commit(
        self,
        commit: SourceCommit,
        source_repo: Path | None,
    ) -> None:
        """Apply a commit from the source repository."""
        if not source_repo:
            return

        source = Repo(source_repo)
        source_commit = source.commit(commit.sha)

        # Get the diff and apply changes
        if source_commit.parents:
            parent = source_commit.parents[0]
            diffs = parent.diff(source_commit)
        else:
            diffs = source_commit.diff(None)

        for diff in diffs:
            if diff.a_path:
                rel_path = diff.a_path
                full_path = self.output_dir / rel_path

                if diff.deleted_file:
                    if full_path.exists():
                        full_path.unlink()
                        try:
                            self.repo.index.remove([rel_path])
                        except Exception:
                            pass
                elif diff.new_file or diff.a_blob:
                    if diff.b_blob:
                        content = diff.b_blob.data_stream.read().decode("utf-8", errors="replace")
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        full_path.write_text(content)
                        self.repo.index.add([rel_path])

        # Create commit with original metadata
        message = f"{commit.message}\n\nSource-Commit: {commit.sha}\nSource-Author: {commit.author} <{commit.author_email}>"
        try:
            self.repo.index.commit(message)
        except Exception:
            pass
