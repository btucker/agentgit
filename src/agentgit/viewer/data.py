"""ViewerData extraction from agentgit repositories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from git import Repo
from git.exc import GitCommandError

from agentgit.git_builder import parse_commit_trailers

logger = logging.getLogger(__name__)

# Colors for prompts (cycle through these)
PROMPT_COLORS = [
    "#e3f2fd",  # Light blue
    "#fff3e0",  # Light orange
    "#e8f5e9",  # Light green
    "#fce4ec",  # Light pink
    "#f3e5f5",  # Light purple
    "#e0f7fa",  # Light cyan
    "#fffde7",  # Light yellow
    "#efebe9",  # Light brown
    "#eceff1",  # Light gray-blue
    "#fbe9e7",  # Light deep orange
]


@dataclass
class BlameRange:
    """A range of lines attributed to a single commit/prompt."""

    start_line: int
    end_line: int
    prompt_id: str
    prompt_num: int  # 1-indexed prompt number
    tool_id: str
    color_index: int
    commit_sha: str
    assistant_context: str | None = None

    @property
    def color(self) -> str:
        """Get the color for this blame range."""
        return PROMPT_COLORS[self.color_index % len(PROMPT_COLORS)]


@dataclass
class OperationSummary:
    """Summary of a file operation for the viewer."""

    operation_type: str  # "write", "edit", "delete"
    timestamp: str
    tool_id: str
    prompt_id: str
    prompt_num: int
    commit_sha: str
    old_content: str | None = None
    new_content: str | None = None


@dataclass
class FileViewerData:
    """Data for displaying a single file in the viewer."""

    path: str
    final_content: str
    status: str  # "added", "modified", "deleted"
    operations: list[OperationSummary] = field(default_factory=list)
    blame_ranges: list[BlameRange] = field(default_factory=list)


@dataclass
class TurnSummary:
    """Summary of an assistant turn."""

    timestamp: str
    summary: str
    files_modified: list[str]
    files_created: list[str]
    files_deleted: list[str]
    tool_ids: list[str]
    context: str | None = None


@dataclass
class PromptViewerData:
    """Data for displaying a prompt in the viewer."""

    prompt_id: str
    prompt_num: int  # 1-indexed
    text: str
    timestamp: str
    files_modified: list[str] = field(default_factory=list)
    turns: list[TurnSummary] = field(default_factory=list)
    color_index: int = 0

    @property
    def color(self) -> str:
        """Get the color for this prompt."""
        return PROMPT_COLORS[self.color_index % len(PROMPT_COLORS)]

    @property
    def short_id(self) -> str:
        """First 8 characters of prompt_id."""
        return self.prompt_id[:8] if self.prompt_id else ""


@dataclass
class CommitViewerData:
    """Data for displaying a commit in the timeline."""

    sha: str
    short_sha: str
    subject: str
    timestamp: str
    prompt_id: str | None
    prompt_num: int | None
    tool_id: str | None
    files_changed: list[str]
    is_merge: bool = False


@dataclass
class ViewerData:
    """Complete data for the web viewer."""

    files: dict[str, FileViewerData] = field(default_factory=dict)
    prompts: list[PromptViewerData] = field(default_factory=list)
    commits: list[CommitViewerData] = field(default_factory=list)
    project_name: str = ""
    version: int = 0  # Incremented on each rebuild for cache busting

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def build_viewer_data(repo_path: Path) -> ViewerData:
    """Build ViewerData from an agentgit repository.

    Args:
        repo_path: Path to the git repository.

    Returns:
        ViewerData with all information needed for the viewer.
    """
    repo = Repo(repo_path)
    data = ViewerData(version=int(Path(repo_path).stat().st_mtime))

    # Extract project name from repo path
    data.project_name = repo_path.name

    # Build prompt map from commits
    prompt_map: dict[str, PromptViewerData] = {}
    prompt_order: list[str] = []  # Track order prompts appear
    commit_list: list[CommitViewerData] = []

    # Iterate commits in reverse chronological order
    try:
        commits = list(repo.iter_commits())
    except ValueError:
        # Empty repository
        return data

    # Process in chronological order (oldest first)
    for commit in reversed(commits):
        metadata = parse_commit_trailers(commit.message)

        # Extract prompt text from commit body
        prompt_text = _extract_prompt_text(commit.message)

        # Track prompt
        if metadata.prompt_id and metadata.prompt_id not in prompt_map:
            prompt_num = len(prompt_order) + 1
            prompt_order.append(metadata.prompt_id)
            prompt_map[metadata.prompt_id] = PromptViewerData(
                prompt_id=metadata.prompt_id,
                prompt_num=prompt_num,
                text=prompt_text or "",
                timestamp=metadata.timestamp or "",
                color_index=(prompt_num - 1) % len(PROMPT_COLORS),
            )

        # Get files changed in this commit
        files_changed = _get_commit_files(commit)

        # Update prompt's files_modified
        if metadata.prompt_id and metadata.prompt_id in prompt_map:
            prompt_data = prompt_map[metadata.prompt_id]
            for f in files_changed:
                if f not in prompt_data.files_modified:
                    prompt_data.files_modified.append(f)

        # Build commit data
        prompt_num = None
        if metadata.prompt_id and metadata.prompt_id in prompt_map:
            prompt_num = prompt_map[metadata.prompt_id].prompt_num

        commit_data = CommitViewerData(
            sha=commit.hexsha,
            short_sha=commit.hexsha[:7],
            subject=commit.message.split("\n")[0],
            timestamp=metadata.timestamp or commit.committed_datetime.isoformat(),
            prompt_id=metadata.prompt_id,
            prompt_num=prompt_num,
            tool_id=metadata.tool_id,
            files_changed=files_changed,
            is_merge=len(commit.parents) > 1,
        )
        commit_list.append(commit_data)

    # Build file data from the final state
    data.files = _build_file_data(repo, prompt_map)

    # Finalize prompts and commits
    data.prompts = [prompt_map[pid] for pid in prompt_order]
    data.commits = commit_list

    return data


def _extract_prompt_text(message: str) -> str | None:
    """Extract prompt text from a commit message.

    Looks for 'Prompt #XXXXXXXX:' followed by the prompt text.
    """
    import re

    match = re.search(r"Prompt #[a-f0-9]+:\n(.+?)(?:\n\n|$)", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _get_commit_files(commit) -> list[str]:
    """Get list of files changed in a commit."""
    files = []
    if commit.parents:
        diff = commit.parents[0].diff(commit)
        for d in diff:
            if d.a_path:
                files.append(d.a_path)
            if d.b_path and d.b_path != d.a_path:
                files.append(d.b_path)
    else:
        # Initial commit - list all files
        for item in commit.tree.traverse():
            if item.type == "blob":
                files.append(item.path)
    return files


def _build_file_data(
    repo: Repo, prompt_map: dict[str, PromptViewerData]
) -> dict[str, FileViewerData]:
    """Build file data including blame information."""
    files: dict[str, FileViewerData] = {}

    # Get all files in the current tree
    try:
        tree = repo.head.commit.tree
    except ValueError:
        return files

    for item in tree.traverse():
        if item.type != "blob":
            continue

        path = item.path
        try:
            content = item.data_stream.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue

        # Determine file status by checking history
        status = _get_file_status(repo, path)

        # Build blame ranges
        blame_ranges = _compute_blame_ranges(repo, path, prompt_map)

        files[path] = FileViewerData(
            path=path,
            final_content=content,
            status=status,
            blame_ranges=blame_ranges,
        )

    return files


def _get_file_status(repo: Repo, path: str) -> str:
    """Determine if a file was added, modified, or exists from start."""
    try:
        commits = list(repo.iter_commits(paths=path))
        if len(commits) == 1:
            return "added"
        return "modified"
    except GitCommandError:
        return "added"


def _compute_blame_ranges(
    repo: Repo, path: str, prompt_map: dict[str, PromptViewerData]
) -> list[BlameRange]:
    """Compute blame ranges for a file."""
    ranges: list[BlameRange] = []

    try:
        blame = repo.blame("HEAD", path)
    except GitCommandError as e:
        logger.warning("Failed to blame %s: %s", path, e)
        return ranges

    current_line = 1
    for commit, lines in blame:
        metadata = parse_commit_trailers(commit.message)
        num_lines = len(lines)

        prompt_num = 0
        color_index = 0
        if metadata.prompt_id and metadata.prompt_id in prompt_map:
            prompt_data = prompt_map[metadata.prompt_id]
            prompt_num = prompt_data.prompt_num
            color_index = prompt_data.color_index

        # Extract assistant context
        context = _extract_context(commit.message)

        ranges.append(
            BlameRange(
                start_line=current_line,
                end_line=current_line + num_lines - 1,
                prompt_id=metadata.prompt_id or "",
                prompt_num=prompt_num,
                tool_id=metadata.tool_id or "",
                color_index=color_index,
                commit_sha=commit.hexsha,
                assistant_context=context,
            )
        )

        current_line += num_lines

    # Merge consecutive ranges with same prompt
    return _merge_blame_ranges(ranges)


def _extract_context(message: str) -> str | None:
    """Extract assistant context from a commit message."""
    import re

    match = re.search(r"Context:\n(.+?)(?:\n\n|$)", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _merge_blame_ranges(ranges: list[BlameRange]) -> list[BlameRange]:
    """Merge consecutive blame ranges with the same prompt."""
    if not ranges:
        return []

    merged: list[BlameRange] = []
    current = ranges[0]

    for r in ranges[1:]:
        if (
            r.prompt_id == current.prompt_id
            and r.start_line == current.end_line + 1
        ):
            # Merge ranges
            current = BlameRange(
                start_line=current.start_line,
                end_line=r.end_line,
                prompt_id=current.prompt_id,
                prompt_num=current.prompt_num,
                tool_id=current.tool_id,  # Keep first tool_id
                color_index=current.color_index,
                commit_sha=current.commit_sha,
                assistant_context=current.assistant_context,
            )
        else:
            merged.append(current)
            current = r

    merged.append(current)
    return merged
