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


def translate_paths_for_agentgit_repo(args: list[str], repo_path: Path) -> list[str]:
    """Translate local file paths to their equivalents in the agentgit repo.

    The agentgit repo uses normalized paths (common prefix stripped).
    This function finds matching files by filename.
    """
    translated = []
    for arg in args:
        # Skip options and things that don't look like paths
        if arg.startswith("-") or "/" not in arg:
            translated.append(arg)
            continue

        # Check if this looks like a local file path
        local_path = Path(arg)
        if not local_path.exists():
            translated.append(arg)
            continue

        # Try to find this file in the agentgit repo by filename
        filename = local_path.name
        matches = list(repo_path.rglob(filename))

        if len(matches) == 1:
            # Found exactly one match - use the relative path
            translated.append(str(matches[0].relative_to(repo_path)))
        elif len(matches) > 1:
            # Multiple matches - try to find best match by path suffix
            local_parts = local_path.parts
            best_match = None
            best_score = 0
            for match in matches:
                match_parts = match.relative_to(repo_path).parts
                # Count matching path components from the end
                score = 0
                for lp, mp in zip(reversed(local_parts), reversed(match_parts)):
                    if lp == mp:
                        score += 1
                    else:
                        break
                if score > best_score:
                    best_score = score
                    best_match = match
            if best_match:
                translated.append(str(best_match.relative_to(repo_path)))
            else:
                translated.append(arg)
        else:
            # No matches found - keep original
            translated.append(arg)

    return translated


def run_git_passthrough(args: list[str]) -> None:
    """Run a git command on the agentgit repo."""
    import subprocess

    repo_path = get_agentgit_repo_path()
    if not repo_path or not repo_path.exists():
        raise click.ClickException(
            "No agentgit repository found. Run 'agentgit' first to create one."
        )

    # Translate local file paths to agentgit repo paths
    translated_args = translate_paths_for_agentgit_repo(args, repo_path)

    # Run git with the provided args in the agentgit repo
    result = subprocess.run(
        ["git", "-C", str(repo_path)] + translated_args,
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
@click.option(
    "--single-repo",
    is_flag=True,
    help="Create agentgit output as an orphan branch in the source repo with a worktree.",
)
@click.option(
    "--branch",
    default="agentgit",
    help="Branch name for --single-repo mode. Defaults to 'agentgit'.",
)
def process(
    transcript: Path | None,
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
    watch: bool,
    single_repo: bool,
    branch: str,
) -> None:
    """Process a transcript into a git repository.

    If TRANSCRIPT is not provided, discovers and processes all transcripts
    for the current project, merging their operations by timestamp.

    With --watch, monitors the transcript file and automatically commits
    new operations as they are added (only works with a single transcript).

    With --single-repo, creates the agentgit output as an orphan branch
    in the source repository, using a git worktree at the output location.
    """
    transcripts = resolve_transcripts(transcript)

    if watch:
        if len(transcripts) > 1:
            raise click.ClickException(
                "Watch mode only supports a single transcript. "
                "Please specify a transcript file explicitly."
            )
        _run_watch_mode(
            transcripts[0], output, author, email, source_repo,
            single_repo=single_repo, branch=branch
        )
    else:
        _run_process(
            transcripts, output, plugin_type, author, email, source_repo,
            single_repo=single_repo, branch=branch
        )


def _run_process(
    transcripts: list[Path],
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
    single_repo: bool = False,
    branch: str = "agentgit",
) -> None:
    """Run processing of one or more transcripts."""
    from agentgit import build_repo, find_git_root, parse_transcripts

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcripts[0])

    # Handle --single-repo mode
    worktree_source_repo = None
    worktree_branch = None
    if single_repo:
        # Find the source repo
        worktree_source_repo = find_git_root()
        if worktree_source_repo is None:
            raise click.ClickException(
                "Cannot use --single-repo: not in a git repository. "
                "Run from within a git repository or specify --source-repo."
            )
        worktree_branch = branch
        click.echo(f"Single-repo mode: creating orphan branch '{branch}' in {worktree_source_repo}")

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
        source_repo=worktree_source_repo if single_repo else source_repo,
        branch=worktree_branch,
        orphan=single_repo,
    )

    click.echo(f"Created git repository at: {repo_path}")
    if single_repo:
        click.echo(f"  Branch: {branch} (orphan)")
        click.echo(f"  Linked to: {worktree_source_repo}")
    click.echo(f"  Prompts: {len(parsed.prompts)}")
    click.echo(f"  Operations: {len(parsed.operations)}")
    click.echo(f"  Commits: {len(list(repo.iter_commits()))}")


def _run_watch_mode(
    transcript: Path,
    output: Path | None,
    author: str,
    email: str,
    source_repo: Path | None,
    single_repo: bool = False,
    branch: str = "agentgit",
) -> None:
    """Run in watch mode."""
    from agentgit import find_git_root
    from agentgit.watcher import TranscriptWatcher

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcript)

    # Handle --single-repo mode
    worktree_source_repo = None
    worktree_branch = None
    if single_repo:
        worktree_source_repo = find_git_root()
        if worktree_source_repo is None:
            raise click.ClickException(
                "Cannot use --single-repo: not in a git repository."
            )
        worktree_branch = branch
        click.echo(f"Single-repo mode: using orphan branch '{branch}' in {worktree_source_repo}")

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
        source_repo=worktree_source_repo if single_repo else source_repo,
        branch=worktree_branch,
        orphan=single_repo,
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


@main.command()
@click.argument("session_id", required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory for git repo.",
)
@click.option(
    "--token",
    help="API access token (auto-detected from keychain on macOS).",
)
@click.option(
    "--org-uuid",
    help="Organization UUID (auto-detected from ~/.claude.json).",
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
def web(
    session_id: str | None,
    output: Path | None,
    token: str | None,
    org_uuid: str | None,
    author: str,
    email: str,
) -> None:
    """Process a Claude Code web session.

    Fetches sessions from the Claude API and processes them into a git repository.
    If SESSION_ID is not provided, displays an interactive session picker.

    Credentials are auto-detected on macOS from the keychain and ~/.claude.json.
    On other platforms, provide --token and --org-uuid manually.

    Examples:

        agentgit web                    # Interactive session picker

        agentgit web abc123             # Process specific session

        agentgit web --token=... --org-uuid=...  # Manual credentials
    """
    from agentgit.web_sessions import (
        WebSessionError,
        fetch_session_data,
        fetch_sessions,
        find_matching_local_project,
        resolve_credentials,
        session_to_jsonl_entries,
    )

    try:
        resolved_token, resolved_org_uuid = resolve_credentials(token, org_uuid)
    except WebSessionError as e:
        raise click.ClickException(str(e)) from e

    if session_id is None:
        # Interactive session picker
        click.echo("Fetching web sessions...")
        try:
            sessions = fetch_sessions(resolved_token, resolved_org_uuid)
        except WebSessionError as e:
            raise click.ClickException(str(e)) from e

        if not sessions:
            raise click.ClickException("No web sessions found.")

        click.echo(f"\nFound {len(sessions)} session(s):\n")

        # Show sessions with local project detection
        for i, session in enumerate(sessions, 1):
            local_project = find_matching_local_project(session)
            local_indicator = " [LOCAL]" if local_project else ""
            title_preview = session.title[:60] if session.title else "Untitled"
            if len(session.title) > 60:
                title_preview += "..."

            click.echo(f"  {i}. {title_preview}{local_indicator}")
            click.echo(f"     ID: {session.id}")
            click.echo(f"     Created: {session.created_at}")
            if session.project_path:
                click.echo(f"     Project: {session.project_path}")
            click.echo()

        # Prompt for selection
        while True:
            try:
                choice = click.prompt(
                    "Select a session (number or ID)",
                    type=str,
                )
                # Try as number first
                try:
                    idx = int(choice)
                    if 1 <= idx <= len(sessions):
                        selected_session = sessions[idx - 1]
                        break
                    click.echo(f"Please enter a number between 1 and {len(sessions)}")
                except ValueError:
                    # Try as session ID
                    matching = [s for s in sessions if s.id == choice]
                    if matching:
                        selected_session = matching[0]
                        break
                    click.echo(f"No session found with ID: {choice}")
            except click.Abort:
                click.echo("\nCancelled.")
                return

        session_id = selected_session.id
        click.echo(f"\nSelected: {selected_session.title}")
    else:
        selected_session = None

    # Fetch the session data
    click.echo(f"Fetching session {session_id}...")
    try:
        session_data = fetch_session_data(resolved_token, resolved_org_uuid, session_id)
    except WebSessionError as e:
        raise click.ClickException(str(e)) from e

    # Convert to JSONL entries and process
    entries = session_to_jsonl_entries(session_data)
    if not entries:
        raise click.ClickException("No transcript entries found in session.")

    # Create a temporary transcript file for processing
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        prefix=f"web-session-{session_id[:8]}-",
    ) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        temp_path = Path(f.name)

    try:
        # Determine output directory
        if output is None:
            # Check for local project match
            project_path = session_data.get("project_path")
            if project_path:
                output = Path.home() / ".agentgit" / "projects" / encode_path_as_name(
                    Path(project_path)
                )
                local_project = Path(project_path)
                if local_project.exists():
                    click.echo(f"Matched local project: {local_project}")
            else:
                # Fall back to session-based naming
                output = Path.home() / ".agentgit" / "web-sessions" / session_id

        _run_process(
            transcripts=[temp_path],
            output=output,
            plugin_type=None,
            author=author,
            email=email,
            source_repo=None,
        )

    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
