"""agentgit - Convert agent transcripts to git repositories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentgit.core import (
    AssistantContext,
    FileOperation,
    OperationType,
    Prompt,
    SourceCommit,
    Transcript,
    TranscriptEntry,
)
from agentgit.git_builder import GitRepoBuilder, format_commit_message
from agentgit.plugins import get_plugin_manager, hookimpl, hookspec, register_builtin_plugins

if TYPE_CHECKING:
    from git import Repo

__version__ = "0.1.0"

__all__ = [
    # Core types
    "FileOperation",
    "Prompt",
    "AssistantContext",
    "TranscriptEntry",
    "Transcript",
    "OperationType",
    "SourceCommit",
    # Plugin system
    "hookspec",
    "hookimpl",
    # Main functions
    "parse_transcript",
    "build_repo",
    "transcript_to_repo",
    "format_commit_message",
    "discover_transcripts",
    "find_git_root",
]


def parse_transcript(path: Path | str, plugin_type: str | None = None) -> Transcript:
    """Parse a transcript file into structured data.

    Args:
        path: Path to the transcript file.
        plugin_type: Optional plugin type to use (e.g., 'claude_code').
            If not specified, auto-detects the format.

    Returns:
        Parsed Transcript object.

    Raises:
        ValueError: If the transcript format cannot be detected or plugin not found.
    """
    path = Path(path)

    pm = get_plugin_manager()
    register_builtin_plugins(pm)

    # Determine format - either from explicit type or auto-detection
    if plugin_type:
        # Find plugin with matching name and get its format
        format = None
        for info in pm.hook.agentgit_get_plugin_info():
            if info and info.get("name") == plugin_type:
                # Try to detect format using this knowledge
                detected = pm.hook.agentgit_detect_format(path=path)
                if detected:
                    format = detected
                    break
        if not format:
            # Fall back to using plugin_type as the format directly
            format = plugin_type
    else:
        format = pm.hook.agentgit_detect_format(path=path)
        if not format:
            raise ValueError(f"Could not detect transcript format for: {path}")

    transcript = pm.hook.agentgit_parse_transcript(path=path, format=format)
    if not transcript:
        raise ValueError(f"Failed to parse transcript: {path}")

    operations = []
    for ops in pm.hook.agentgit_extract_operations(transcript=transcript):
        operations.extend(ops)

    enriched_operations = []
    for op in operations:
        for enriched_op in pm.hook.agentgit_enrich_operation(operation=op, transcript=transcript):
            enriched_operations.append(enriched_op)
            break

    transcript.operations = enriched_operations
    return transcript


def build_repo(
    operations: list[FileOperation],
    output_dir: Path | None = None,
    author_name: str = "Agent",
    author_email: str = "agent@local",
    source_repo: Path | None = None,
) -> tuple[Repo, Path, dict[str, str]]:
    """Build a git repository from file operations.

    Args:
        operations: List of FileOperation objects.
        output_dir: Directory for the git repo. If None, creates a temp dir.
        author_name: Name for git commits.
        author_email: Email for git commits.
        source_repo: Optional source repository to interleave commits from.

    Returns:
        Tuple of (repo, repo_path, path_mapping).
    """
    builder = GitRepoBuilder(output_dir=output_dir)
    return builder.build(
        operations=operations,
        author_name=author_name,
        author_email=author_email,
        source_repo=source_repo,
    )


def transcript_to_repo(
    transcript_path: Path | str,
    output_dir: Path | None = None,
    author_name: str = "Agent",
    author_email: str = "agent@local",
    source_repo: Path | str | None = None,
    plugin_type: str | None = None,
) -> tuple[Repo, Path, Transcript]:
    """Parse a transcript and build a git repository.

    Convenience function combining parse_transcript and build_repo.

    Args:
        transcript_path: Path to the transcript file.
        output_dir: Directory for the git repo. If None, creates a temp dir.
        author_name: Name for git commits.
        author_email: Email for git commits.
        source_repo: Optional source repository to interleave commits from.
        plugin_type: Optional plugin type to use (e.g., 'claude_code').
            If not specified, auto-detects the format.

    Returns:
        Tuple of (repo, repo_path, transcript).
    """
    transcript = parse_transcript(transcript_path, plugin_type=plugin_type)

    source_repo_path = Path(source_repo) if source_repo else None

    repo, repo_path, _ = build_repo(
        operations=transcript.operations,
        output_dir=output_dir,
        author_name=author_name,
        author_email=author_email,
        source_repo=source_repo_path,
    )
    return repo, repo_path, transcript


def find_git_root(start_path: Path | str | None = None) -> Path | None:
    """Find the root of the git repository containing start_path.

    Args:
        start_path: Path to start searching from. Defaults to current directory.

    Returns:
        Path to git root, or None if not in a git repository.
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    current = start_path.resolve()

    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    # Check root directory
    if (current / ".git").exists():
        return current

    return None


def discover_transcripts(project_path: Path | str | None = None) -> list[Path]:
    """Discover transcript files for a project.

    If no project_path is provided, attempts to find the git root of the
    current directory.

    Args:
        project_path: Path to the project. Defaults to current git repo root.

    Returns:
        List of paths to discovered transcript files, sorted by modification
        time (most recent first).

    Raises:
        ValueError: If no project_path provided and not in a git repository.
    """
    if project_path is None:
        project_path = find_git_root()
        if project_path is None:
            raise ValueError(
                "Not in a git repository. Please specify a project path or transcript file."
            )
    else:
        project_path = Path(project_path)

    pm = get_plugin_manager()
    register_builtin_plugins(pm)

    # Collect transcripts from all plugins
    all_transcripts = []
    for transcripts in pm.hook.agentgit_discover_transcripts(project_path=project_path):
        all_transcripts.extend(transcripts)

    # Sort by modification time, most recent first
    all_transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_transcripts
