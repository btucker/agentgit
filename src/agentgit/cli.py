"""CLI for agentgit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click


def get_available_types() -> list[str]:
    """Get list of available plugin types."""
    from agentgit.plugins import get_configured_plugin_manager

    pm = get_configured_plugin_manager()

    types = []
    for info in pm.hook.agentgit_get_plugin_info():
        if info and "name" in info:
            types.append(info["name"])
    return types


def encode_path_as_name(path: Path) -> str:
    """Encode a path for use as a project name in directory names.

    Replaces path separators with dashes.
    e.g., /Users/name/project -> -Users-name-project

    Args:
        path: The path to encode.

    Returns:
        Encoded string safe for directory names.
    """
    return str(path.resolve()).replace("/", "-")


def get_default_output_dir(transcript_path: Path) -> Path:
    """Get the default output directory for a transcript.

    Uses ~/.agentgit/projects/{project_name} where project_name is derived
    from the project (similar to Claude Code's convention).

    The project name is determined by:
    1. Asking plugins to get the project name from transcript location
       (e.g., Claude Code returns "-Users-name-project" from
       ~/.claude/projects/-Users-name-project/session.jsonl)
    2. Falling back to encoding the current git root
    3. Falling back to encoding the transcript's parent directory

    Args:
        transcript_path: Path to the transcript file.

    Returns:
        Path to the default output directory.
    """
    from agentgit import find_git_root
    from agentgit.plugins import get_configured_plugin_manager

    pm = get_configured_plugin_manager()

    # Ask plugins to get the project name from transcript location
    project_name = pm.hook.agentgit_get_project_name(transcript_path=transcript_path)

    if not project_name:
        # Try to find git root from current directory
        git_root = find_git_root()
        if git_root:
            project_name = encode_path_as_name(git_root)

    if not project_name:
        # Fall back to transcript's parent directory
        project_name = encode_path_as_name(transcript_path.resolve().parent)

    return Path.home() / ".agentgit" / "projects" / project_name


def resolve_transcripts(transcript: Path | None) -> list[Path]:
    """Resolve transcript paths, using auto-discovery if not provided.

    If a transcript is explicitly provided, returns a single-item list.
    Otherwise, discovers all transcripts for the current project.
    """
    if transcript is not None:
        return [transcript]

    from agentgit import discover_transcripts

    transcripts = discover_transcripts()
    if not transcripts:
        raise click.ClickException(
            "No transcripts found for this project. "
            "Please specify a transcript file explicitly."
        )

    return transcripts


def load_and_display_transcript_header(transcript: Path | None, item_name: str) -> tuple:
    """Load transcripts and display a header with count.

    Common helper for prompts and operations commands.

    Args:
        transcript: Optional transcript path.
        item_name: Name of items being listed (e.g., "prompts", "operations").

    Returns:
        Tuple of (transcripts list, parsed Transcript).
    """
    from agentgit import parse_transcripts

    transcripts = resolve_transcripts(transcript)
    parsed = parse_transcripts(transcripts)

    if len(transcripts) == 1:
        click.echo(f"Transcript: {transcripts[0].name}")
    else:
        click.echo(f"Transcripts: {len(transcripts)} files")

    return transcripts, parsed


def get_agentgit_repo_path() -> Path | None:
    """Get the agentgit output repo path for the current project.

    Returns:
        Path to the agentgit repo, or None if not determinable.
    """
    from agentgit import discover_transcripts, find_git_root

    # Try to find transcripts for current project
    git_root = find_git_root()
    if not git_root:
        return None

    transcripts = discover_transcripts(git_root)
    if not transcripts:
        return None

    # Use the first transcript to determine output path
    return get_default_output_dir(transcripts[0])


def run_git_passthrough(args: list[str]) -> None:
    """Run a git command on the agentgit repo."""
    import subprocess

    repo_path = get_agentgit_repo_path()
    if not repo_path or not repo_path.exists():
        raise click.ClickException(
            "No agentgit repository found. Run 'agentgit' first to create one."
        )

    # Run git with the provided args in the agentgit repo
    result = subprocess.run(
        ["git", "-C", str(repo_path)] + args,
        capture_output=False,
    )
    raise SystemExit(result.returncode)


class DefaultGroup(click.Group):
    """A click Group that allows a default command and git passthrough."""

    def __init__(self, *args: Any, default_cmd: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_cmd = default_cmd

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Insert default command if no subcommand given."""
        # If no args at all, use the default command
        if not args and self.default_cmd:
            args = [self.default_cmd]
            return super().parse_args(ctx, args)

        if not args:
            return super().parse_args(ctx, args)

        # Don't intercept top-level options like --help, --version
        if args[0].startswith("-"):
            return super().parse_args(ctx, args)

        # Check if first arg is a known command
        if args[0] in self.commands:
            return super().parse_args(ctx, args)

        # Check if first arg looks like a file path (for process command)
        first_arg_path = Path(args[0])
        if first_arg_path.exists() and first_arg_path.is_file():
            # Prepend the default command for file arguments
            args = [self.default_cmd] + args
            return super().parse_args(ctx, args)

        # Otherwise, treat as git passthrough
        run_git_passthrough(args)
        return []  # Never reached due to SystemExit


@click.group(cls=DefaultGroup, default_cmd="process")
@click.version_option()
def main() -> None:
    """Process agent transcripts into git repositories.

    If no command is specified, 'process' is used by default.
    If no transcript is specified, automatically discovers transcripts
    for the current git repository.

    Any unrecognized command is passed through to git, running on the
    agentgit-created repository. This allows you to use familiar git
    commands like 'log', 'diff', 'show', etc.

    Examples:

        agentgit                           # Process all transcripts for current project

        agentgit session.jsonl -o ./output # Process specific transcript

        agentgit log --oneline -10         # View recent commits in agentgit repo

        agentgit diff HEAD~5..HEAD         # View changes in agentgit repo

        agentgit prompts                   # List prompts from transcripts
    """
    pass


@main.command()
@click.argument("transcript", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory for git repo. Defaults to ~/.agentgit/projects/{project}.",
)
@click.option(
    "-t",
    "--type",
    "plugin_type",
    help="Plugin type to use (e.g., 'claude_code'). Auto-detects if not specified.",
)
@click.option(
    "--author",
    default="Agent",
    help="Author name for git commits.",
)
@click.option(
    "--email",
    default="agent@local",
    help="Author email for git commits.",
)
@click.option(
    "--source-repo",
    type=click.Path(exists=True, path_type=Path),
    help="Source repository to interleave commits from.",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Watch transcript for changes and auto-commit.",
)
def process(
    transcript: Path | None,
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
    watch: bool,
) -> None:
    """Process a transcript into a git repository.

    If TRANSCRIPT is not provided, discovers and processes all transcripts
    for the current project, merging their operations by timestamp.

    With --watch, monitors the transcript file and automatically commits
    new operations as they are added (only works with a single transcript).
    """
    transcripts = resolve_transcripts(transcript)

    if watch:
        if len(transcripts) > 1:
            raise click.ClickException(
                "Watch mode only supports a single transcript. "
                "Please specify a transcript file explicitly."
            )
        _run_watch_mode(transcripts[0], output, author, email, source_repo)
    else:
        _run_process(transcripts, output, plugin_type, author, email, source_repo)


def _run_process(
    transcripts: list[Path],
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
) -> None:
    """Run processing of one or more transcripts."""
    from agentgit import build_repo, parse_transcripts

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcripts[0])

    if len(transcripts) == 1:
        click.echo(f"Processing transcript: {transcripts[0]}")
    else:
        click.echo(f"Processing {len(transcripts)} transcripts:")
        for t in transcripts:
            click.echo(f"  - {t.name}")

    parsed = parse_transcripts(transcripts, plugin_type=plugin_type)

    repo, repo_path, _ = build_repo(
        operations=parsed.operations,
        output_dir=output,
        author_name=author,
        author_email=email,
        source_repo=source_repo,
    )

    click.echo(f"Created git repository at: {repo_path}")
    click.echo(f"  Prompts: {len(parsed.prompts)}")
    click.echo(f"  Operations: {len(parsed.operations)}")
    click.echo(f"  Commits: {len(list(repo.iter_commits()))}")


def _run_watch_mode(
    transcript: Path,
    output: Path | None,
    author: str,
    email: str,
    source_repo: Path | None,
) -> None:
    """Run in watch mode."""
    from agentgit.watcher import TranscriptWatcher

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcript)

    click.echo(f"Watching transcript: {transcript}")
    click.echo(f"Output directory: {output}")
    click.echo("Press Ctrl+C to stop.\n")

    def on_update(new_commits: int) -> None:
        click.echo(f"  Added {new_commits} commit(s)")

    watcher = TranscriptWatcher(
        transcript_path=transcript,
        output_dir=output,
        author_name=author,
        author_email=email,
        source_repo=source_repo,
        on_update=on_update,
    )

    # Initial build status
    from git import Repo
    from git.exc import InvalidGitRepositoryError

    watcher.start()

    try:
        repo = Repo(output)
        commits = len(list(repo.iter_commits()))
        click.echo(f"Initial build: {commits} commit(s)")
    except InvalidGitRepositoryError:
        click.echo("Initial build: 0 commits")

    click.echo("Watching for changes...")

    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher...")
    finally:
        watcher.stop()


@main.command()
@click.argument("transcript", type=click.Path(exists=True, path_type=Path), required=False)
def prompts(transcript: Path | None) -> None:
    """List prompts from a transcript with their md5 IDs.

    If TRANSCRIPT is not provided, lists prompts from all transcripts
    discovered for the current project.
    """
    _, parsed = load_and_display_transcript_header(transcript, "prompts")
    click.echo(f"Found {len(parsed.prompts)} prompts:\n")

    for i, prompt in enumerate(parsed.prompts, 1):
        text_preview = prompt.text[:80].replace("\n", " ")
        if len(prompt.text) > 80:
            text_preview += "..."

        click.echo(f"{i}. [{prompt.short_id}] {text_preview}")
        click.echo(f"   Timestamp: {prompt.timestamp}")
        click.echo(f"   Full ID: {prompt.prompt_id}")
        click.echo()


@main.command()
@click.argument("transcript", type=click.Path(exists=True, path_type=Path), required=False)
def operations(transcript: Path | None) -> None:
    """List file operations from a transcript.

    If TRANSCRIPT is not provided, lists operations from all transcripts
    discovered for the current project.
    """
    _, parsed = load_and_display_transcript_header(transcript, "operations")
    click.echo(f"Found {len(parsed.operations)} operations:\n")

    for i, op in enumerate(parsed.operations, 1):
        prompt_ref = f"[{op.prompt.short_id}]" if op.prompt else "[no prompt]"

        click.echo(f"{i}. {op.operation_type.value.upper()} {op.file_path}")
        click.echo(f"   Timestamp: {op.timestamp}")
        click.echo(f"   Prompt: {prompt_ref}")
        if op.tool_id:
            click.echo(f"   Tool ID: {op.tool_id}")
        click.echo()


@main.command()
@click.option(
    "--project",
    type=click.Path(exists=True, path_type=Path),
    help="Project path to discover transcripts for. Defaults to current git repo.",
)
def discover(project: Path | None) -> None:
    """Discover available transcripts for a project.

    Lists all transcript files found for the current project (or specified
    project path), sorted by modification time.
    """
    from agentgit import discover_transcripts, find_git_root

    if project is None:
        project = find_git_root()
        if project is None:
            raise click.ClickException(
                "Not in a git repository. Use --project to specify a project path."
            )

    click.echo(f"Project: {project}")

    transcripts = discover_transcripts(project)

    if not transcripts:
        click.echo("No transcripts found for this project.")
        return

    click.echo(f"Found {len(transcripts)} transcript(s):\n")

    for i, path in enumerate(transcripts, 1):
        stat = path.stat()
        from datetime import datetime

        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = stat.st_size / 1024

        click.echo(f"{i}. {path.name}")
        click.echo(f"   Modified: {mtime}")
        click.echo(f"   Size: {size_kb:.1f} KB")
        click.echo(f"   Path: {path}")
        click.echo()


@main.command()
def types() -> None:
    """List available transcript format plugins."""
    from agentgit.plugins import get_configured_plugin_manager

    pm = get_configured_plugin_manager()

    click.echo("Available transcript format plugins:\n")

    for info in pm.hook.agentgit_get_plugin_info():
        if info:
            name = info.get("name", "unknown")
            description = info.get("description", "No description")
            click.echo(f"  {name}: {description}")


if __name__ == "__main__":
    main()
