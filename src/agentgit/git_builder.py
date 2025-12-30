"""Git repository builder for agentgit."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from agentgit.core import (
    AssistantTurn,
    FileOperation,
    OperationType,
    Prompt,
    PromptResponse,
    SourceCommit,
)

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
    except ValueError:
        # Empty repository has no commits
        pass
    except GitCommandError as e:
        logger.warning("Failed to read commits from repository: %s", e)

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
    except ValueError:
        # Empty repository has no commits
        pass
    except GitCommandError as e:
        logger.warning("Failed to read commits from repository: %s", e)

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


def format_turn_commit_message(turn: AssistantTurn) -> str:
    """Format a commit message for an assistant turn (grouped operations).

    Structure:
    - Subject line: summary of what was done
    - Blank line
    - Files modified/created/deleted
    - Blank line
    - Context if available
    - Blank line
    - Trailers
    """
    subject = turn.summary_line

    body_parts = []

    # List of files changed
    file_lists = []
    if turn.files_created:
        file_lists.append(f"Created: {', '.join(turn.files_created)}")
    if turn.files_modified:
        file_lists.append(f"Modified: {', '.join(turn.files_modified)}")
    if turn.files_deleted:
        file_lists.append(f"Deleted: {', '.join(turn.files_deleted)}")
    if file_lists:
        body_parts.append("\n".join(file_lists))

    # Assistant context (the reasoning)
    if turn.context and turn.context.summary:
        context = turn.context.summary
        if context:
            body_parts.append(f"Context:\n{context}")

    body = "\n\n".join(body_parts) if body_parts else ""

    # Trailers
    trailers = []
    if turn.operations and turn.operations[0].prompt:
        trailers.append(f"Prompt-Id: {turn.operations[0].prompt.prompt_id}")
    trailers.append(f"Timestamp: {turn.timestamp}")
    for op in turn.operations:
        if op.tool_id:
            trailers.append(f"Tool-Id: {op.tool_id}")

    trailers_str = "\n".join(trailers)

    if body:
        return f"{subject}\n\n{body}\n\n{trailers_str}"
    else:
        return f"{subject}\n\n{trailers_str}"


def format_prompt_merge_message(prompt: Prompt, turns: list[AssistantTurn]) -> str:
    """Format a merge commit message for a prompt.

    Structure:
    - Subject line: first line of prompt (truncated if needed)
    - Blank line
    - Full prompt text
    - Blank line
    - Trailers
    """
    # Subject: first meaningful line of prompt
    first_line = prompt.text.split("\n")[0].strip()
    if len(first_line) > 72:
        subject = first_line[:69] + "..."
    else:
        subject = first_line

    body_parts = []

    # Full prompt text
    body_parts.append(f"Prompt #{prompt.short_id}:\n{prompt.text}")

    body = "\n\n".join(body_parts)

    # Trailers
    trailers = [f"Prompt-Id: {prompt.prompt_id}"]

    trailers_str = "\n".join(trailers)

    return f"{subject}\n\n{body}\n\n{trailers_str}"


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

    def __init__(
        self,
        output_dir: Path | None = None,
        source_repo: Path | None = None,
        branch: str | None = None,
        orphan: bool = False,
    ):
        """Initialize the builder.

        Args:
            output_dir: Directory for the git repo. If None, creates a temp dir.
            source_repo: Path to source repository for worktree mode.
                When provided with branch, creates a worktree instead of standalone repo.
            branch: Branch name for worktree mode (e.g., "agentgit/history").
                Required when source_repo is provided.
            orphan: If True, create an orphan branch (no common ancestor with main).
                Only used when source_repo and branch are provided.
        """
        self.output_dir = output_dir
        self.source_repo_path = source_repo
        self.branch = branch
        self.orphan = orphan
        self.repo: Repo | None = None
        self.path_mapping: dict[str, str] = {}
        self.file_states: dict[str, str] = {}
        self._processed_ops: set[str] = set()
        self._is_worktree = False

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
        # Use worktree mode if source_repo and branch are specified
        if self.source_repo_path and self.branch:
            return self._setup_worktree(incremental)

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
            return False

    def _setup_worktree(self, incremental: bool) -> bool:
        """Set up a git worktree for the agentgit output.

        Creates an orphan branch in the source repo and adds a worktree
        at output_dir pointing to that branch.

        Args:
            incremental: If True and worktree exists, load processed operations.

        Returns:
            True if existing worktree was found, False if new worktree was created.
        """
        if self.output_dir is None:
            raise ValueError("output_dir is required for worktree mode")

        source_repo = Repo(self.source_repo_path)
        self._is_worktree = True

        # Check if output_dir already exists as a worktree
        if self.output_dir.exists() and (self.output_dir / ".git").exists():
            # Worktree already exists
            self.repo = Repo(self.output_dir)
            if incremental:
                self._processed_ops = get_processed_operations(self.repo)
                self._load_file_states_from_repo()
            return True

        # Check if branch already exists
        branch_exists = self.branch in [b.name for b in source_repo.branches]

        if branch_exists:
            # Branch exists, just add worktree
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            source_repo.git.worktree("add", str(self.output_dir), self.branch)
            self.repo = Repo(self.output_dir)
            if incremental:
                self._processed_ops = get_processed_operations(self.repo)
                self._load_file_states_from_repo()
            return True

        # Create new branch (orphan or regular)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)

        if self.orphan:
            # Create orphan branch using worktree
            # git worktree add --detach creates a detached HEAD worktree
            # We then create the orphan branch from there
            source_repo.git.worktree("add", "--detach", str(self.output_dir))
            self.repo = Repo(self.output_dir)

            # Now create the orphan branch
            # git checkout --orphan creates a branch with no parent
            self.repo.git.checkout("--orphan", self.branch)

            # Clean up any files that might have been checked out
            # (orphan branch starts with staged files from detached HEAD)
            try:
                self.repo.git.rm("-rf", "--cached", ".")
                # Also remove actual files
                for item in self.output_dir.iterdir():
                    if item.name != ".git":
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            import shutil
                            shutil.rmtree(item)
            except GitCommandError:
                pass  # May fail if no files to remove
        else:
            # Create regular branch from current HEAD
            source_repo.git.worktree("add", "-b", self.branch, str(self.output_dir))
            self.repo = Repo(self.output_dir)

        return False

    def _get_merged_timeline(
        self,
        operations: list[FileOperation],
        source_repo: Path | None,
    ) -> list[Union[FileOperation, SourceCommit]]:
        """Merge operations with source repo commits by timestamp.

        Args:
            operations: List of operations to include.
            source_repo: Optional source repository to include commits from.

        Returns:
            Combined list of operations and commits sorted by timestamp.
        """
        if not source_repo or not operations:
            return list(operations)

        start_time = min(op.timestamp for op in operations)
        end_time = max(op.timestamp for op in operations)
        source_commits = get_commits_in_timeframe(source_repo, start_time, end_time)
        timeline = merge_timeline(operations, source_commits)
        timeline.sort(key=lambda x: x.timestamp)
        return timeline

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
        existing_repo = self._setup_repo(incremental)

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

        timeline = self._get_merged_timeline(operations, source_repo)

        for item in timeline:
            if isinstance(item, FileOperation):
                self._apply_operation(item)
                self._processed_ops.add(self._get_operation_id(item))
            elif isinstance(item, SourceCommit):
                self._apply_source_commit(item, source_repo)

        return self.repo, self.output_dir, self.path_mapping

    def build_from_prompt_responses(
        self,
        prompt_responses: list[PromptResponse],
        author_name: str = "Agent",
        author_email: str = "agent@local",
        incremental: bool = True,
    ) -> tuple[Repo, Path, dict[str, str]]:
        """Build a git repository using merge-based structure.

        Each prompt becomes a merge commit on main, with individual assistant
        turns as commits on a feature branch.

        Args:
            prompt_responses: List of PromptResponse objects.
            author_name: Name for git commits.
            author_email: Email for git commits.
            incremental: If True, skip already-processed operations.

        Returns:
            Tuple of (repo, repo_path, path_mapping).
        """
        existing_repo = self._setup_repo(incremental)

        with self.repo.config_writer() as config:
            config.set_value("user", "name", author_name)
            config.set_value("user", "email", author_email)

        # Collect all operations for path normalization
        all_operations = []
        for pr in prompt_responses:
            for turn in pr.turns:
                all_operations.extend(turn.operations)

        non_delete_ops = [op for op in all_operations if op.operation_type != OperationType.DELETE]
        _, self.path_mapping = normalize_file_paths(non_delete_ops)

        if not prompt_responses:
            return self.repo, self.output_dir, self.path_mapping

        # Ensure we have an initial commit
        if not self._has_commits():
            self._create_initial_commit()

        for pr in prompt_responses:
            self._process_prompt_response(pr, incremental)

        return self.repo, self.output_dir, self.path_mapping

    def _has_commits(self) -> bool:
        """Check if the repository has any commits."""
        try:
            _ = self.repo.head.commit
            return True
        except ValueError:
            return False

    def _create_initial_commit(self) -> None:
        """Create an initial empty commit."""
        self.repo.index.commit("Initial commit")

    def _process_prompt_response(
        self, pr: PromptResponse, incremental: bool
    ) -> None:
        """Process a single prompt response, creating feature branch and merge."""
        if not pr.turns:
            return

        # Filter turns to only those with unprocessed operations
        turns_to_process = []
        for turn in pr.turns:
            has_unprocessed = False
            for op in turn.operations:
                if not self._is_operation_processed(op):
                    has_unprocessed = True
                    break
            if has_unprocessed:
                turns_to_process.append(turn)

        if not turns_to_process:
            return

        # Remember the current main branch position
        main_ref = self.repo.head.commit

        # Create a feature branch for this prompt
        branch_name = f"prompt-{pr.prompt.short_id}"
        feature_branch = self.repo.create_head(branch_name, main_ref)
        feature_branch.checkout()

        # Apply each turn as a commit on the feature branch
        for turn in turns_to_process:
            self._apply_turn(turn)
            # Mark all operations in this turn as processed
            for op in turn.operations:
                self._processed_ops.add(self._get_operation_id(op))

        # Switch back to main
        self.repo.heads.main.checkout() if "main" in self.repo.heads else self.repo.heads.master.checkout()

        # Merge the feature branch with a merge commit
        merge_message = format_prompt_merge_message(pr.prompt, pr.turns)
        try:
            self.repo.git.merge(branch_name, "--no-ff", "-m", merge_message)
        except GitCommandError as e:
            logger.warning("Merge failed for prompt %s: %s", pr.prompt.short_id, e)
            # Try to continue despite merge issues
            try:
                self.repo.git.merge("--abort")
            except GitCommandError:
                pass

        # Delete the feature branch
        try:
            self.repo.delete_head(branch_name, force=True)
        except GitCommandError:
            pass

    def _apply_turn(self, turn: AssistantTurn) -> None:
        """Apply all operations in a turn and create a single commit."""
        if not turn.operations:
            return

        files_changed = []
        files_deleted = []

        for operation in turn.operations:
            changed = self._apply_operation_no_commit(operation)
            if changed:
                rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
                if operation.operation_type == OperationType.DELETE:
                    files_deleted.append(rel_path)
                else:
                    files_changed.append(rel_path)

        if files_changed or files_deleted:
            # Add changed files (not deleted ones)
            for rel_path in files_changed:
                try:
                    self.repo.index.add([rel_path])
                except GitCommandError:
                    pass

            commit_msg = format_turn_commit_message(turn)
            try:
                self.repo.index.commit(commit_msg)
            except GitCommandError as e:
                logger.warning("Failed to commit turn: %s", e)

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
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(operation.content or "")

        self.file_states[operation.file_path] = operation.content or ""
        return True

    def _apply_edit_no_commit(self, operation: FileOperation) -> bool:
        """Apply an edit operation without committing."""
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
        delete_path = operation.file_path

        files_to_remove = []

        if operation.recursive:
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
                except GitCommandError:
                    # File may not be tracked in git
                    logger.debug("Could not remove %s from index (may not be tracked)", rel_path)

            commit_msg = format_commit_message(operation)
            try:
                self.repo.index.commit(commit_msg)
            except GitCommandError as e:
                # May fail if nothing to commit
                logger.debug("Could not commit delete operation: %s", e)

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
                        except GitCommandError:
                            # File may not be tracked in git
                            logger.debug("Could not remove %s from index", rel_path)
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
        except GitCommandError as e:
            # May fail if nothing to commit
            logger.debug("Could not commit source commit %s: %s", commit.sha[:8], e)
