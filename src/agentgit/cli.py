"""CLI for agentgit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click


def get_available_types() -> list[str]:
    """Get list of available plugin types."""
    from agentgit.plugins import get_plugin_manager, register_builtin_plugins

    pm = get_plugin_manager()
    register_builtin_plugins(pm)

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
    from agentgit.plugins import get_plugin_manager, register_builtin_plugins

    pm = get_plugin_manager()
    register_builtin_plugins(pm)

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


def resolve_transcript(transcript: Path | None) -> Path:
    """Resolve transcript path, using auto-discovery if not provided."""
    if transcript is not None:
        return transcript

    from agentgit import discover_transcripts

    transcripts = discover_transcripts()
    if not transcripts:
        raise click.ClickException(
            "No transcripts found for this project. "
            "Please specify a transcript file explicitly."
        )

    # Use most recent transcript
    return transcripts[0]


class DefaultGroup(click.Group):
    """A click Group that allows a default command."""

    def __init__(self, *args: Any, default_cmd: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_cmd = default_cmd

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Insert default command if no subcommand given."""
        if not args:
            return args

        # Don't intercept top-level options like --help, --version
        if args[0].startswith("-"):
            return super().parse_args(ctx, args)

        # Check if first arg is a known command
        if args[0] not in self.commands and self.default_cmd:
            # Prepend the default command
            args = [self.default_cmd] + args

        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup, default_cmd="process")
@click.version_option()
def main() -> None:
    """Process agent transcripts into git repositories.

    If no command is specified, 'process' is used by default.
    If no transcript is specified, automatically discovers transcripts
    for the current git repository.

    Examples:

        agentgit session.jsonl -o ./output

        agentgit process session.jsonl --watch

        agentgit prompts session.jsonl
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

    If TRANSCRIPT is not provided, uses the most recent transcript
    discovered for the current project.

    With --watch, monitors the transcript file and automatically commits
    new operations as they are added.
    """
    transcript = resolve_transcript(transcript)

    if watch:
        _run_watch_mode(transcript, output, author, email, source_repo)
    else:
        _run_process(transcript, output, plugin_type, author, email, source_repo)


def _run_process(
    transcript: Path,
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
) -> None:
    """Run a single processing."""
    from agentgit import transcript_to_repo

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcript)

    click.echo(f"Processing transcript: {transcript}")

    repo, repo_path, parsed = transcript_to_repo(
        transcript_path=transcript,
        output_dir=output,
        author_name=author,
        author_email=email,
        source_repo=source_repo,
        plugin_type=plugin_type,
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

    If TRANSCRIPT is not provided, uses the most recent transcript
    discovered for the current project.
    """
    from agentgit import parse_transcript

    transcript = resolve_transcript(transcript)
    parsed = parse_transcript(transcript)

    click.echo(f"Transcript: {transcript.name}")
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

    If TRANSCRIPT is not provided, uses the most recent transcript
    discovered for the current project.
    """
    from agentgit import parse_transcript

    transcript = resolve_transcript(transcript)
    parsed = parse_transcript(transcript)

    click.echo(f"Transcript: {transcript.name}")
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
    from agentgit.plugins import get_plugin_manager, register_builtin_plugins

    pm = get_plugin_manager()
    register_builtin_plugins(pm)

    click.echo("Available transcript format plugins:\n")

    for info in pm.hook.agentgit_get_plugin_info():
        if info:
            name = info.get("name", "unknown")
            description = info.get("description", "No description")
            click.echo(f"  {name}: {description}")


if __name__ == "__main__":
    main()
