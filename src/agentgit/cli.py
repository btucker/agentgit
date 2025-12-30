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
@click.option(
    "--web",
    "web_session_id",
    type=str,
    help="Process a Claude Code web session by ID. Use 'list' to show available sessions.",
)
@click.option(
    "--token",
    help="API access token for web sessions (auto-detected from keychain on macOS).",
)
@click.option(
    "--org-uuid",
    help="Organization UUID for web sessions (auto-detected from ~/.claude.json).",
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
    web_session_id: str | None,
    token: str | None,
    org_uuid: str | None,
) -> None:
    """Process a transcript into a git repository.

    If TRANSCRIPT is not provided, discovers and processes all transcripts
    for the current project, merging their operations by timestamp.

    With --watch, monitors the transcript file and automatically commits
    new operations as they are added (only works with a single transcript).

    With --single-repo, creates the agentgit output as an orphan branch
    in the source repository, using a git worktree at the output location.

    With --web, processes a Claude Code web session. Use --web=list to show
    available sessions, or --web=SESSION_ID to process a specific session.

    Examples:

        agentgit process --web=list          # List web sessions

        agentgit process --web=abc123        # Process session by ID
    """
    # Handle web session processing
    if web_session_id is not None:
        _run_web_process(
            web_session_id=web_session_id,
            output=output,
            author=author,
            email=email,
            token=token,
            org_uuid=org_uuid,
        )
        return

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


def _run_web_process(
    web_session_id: str,
    output: Path | None,
    author: str,
    email: str,
    token: str | None,
    org_uuid: str | None,
) -> None:
    """Process a web session by ID or list available sessions."""
    import json
    import tempfile

    from agentgit.plugins import get_configured_plugin_manager

    pm = get_configured_plugin_manager()

    # Resolve credentials through plugin
    credentials = pm.hook.agentgit_resolve_web_credentials(
        token=token, org_uuid=org_uuid
    )

    if not credentials:
        raise click.ClickException(
            "Could not resolve web session credentials. "
            "Please provide --token and --org-uuid manually."
        )

    # Handle 'list' command
    if web_session_id.lower() == "list":
        click.echo("Fetching web sessions...")
        all_sessions = []
        for sessions in pm.hook.agentgit_discover_web_sessions(
            project_path=None, credentials=credentials
        ):
            all_sessions.extend(sessions)

        if not all_sessions:
            click.echo("No web sessions found.")
            return

        click.echo(f"\nFound {len(all_sessions)} session(s):\n")
        for i, session in enumerate(all_sessions, 1):
            title = session.name
            click.echo(f"  {i}. {title}")
            click.echo(f"     ID: {session.session_id}")
            click.echo(f"     Created: {session.mtime_formatted}")
            if session.project_path:
                click.echo(f"     Project: {session.project_path}")
            click.echo()
        return

    # Fetch and process specific session
    click.echo(f"Fetching session {web_session_id}...")
    entries = pm.hook.agentgit_fetch_web_session(
        session_id=web_session_id, credentials=credentials
    )

    if not entries:
        raise click.ClickException(f"No transcript entries found in session {web_session_id}.")

    # Create a temporary transcript file for processing
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        prefix=f"web-session-{web_session_id[:8]}-",
    ) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        temp_path = Path(f.name)

    try:
        # Determine output directory if not specified
        if output is None:
            # Try to get project path from session data (first entry with cwd)
            project_path = None
            for entry in entries:
                cwd = entry.get("cwd")
                if cwd:
                    project_path = cwd
                    break

            if project_path:
                output = Path.home() / ".agentgit" / "projects" / encode_path_as_name(
                    Path(project_path)
                )
            else:
                output = Path.home() / ".agentgit" / "web-sessions" / web_session_id

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
@click.option(
    "--all",
    "all_projects",
    is_flag=True,
    help="Show transcripts from all projects, not just the current one.",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    help="List transcripts without interactive selection.",
)
@click.option(
    "--type",
    "filter_type",
    type=str,
    help="Filter by transcript type (e.g., claude_code, codex).",
)
@click.option(
    "--web",
    "include_web",
    is_flag=True,
    help="Include web sessions from the Claude API.",
)
@click.option(
    "--token",
    help="API access token for web sessions (auto-detected from keychain on macOS).",
)
@click.option(
    "--org-uuid",
    help="Organization UUID for web sessions (auto-detected from ~/.claude.json).",
)
def discover(
    project: Path | None,
    all_projects: bool,
    list_only: bool,
    filter_type: str | None,
    include_web: bool,
    token: str | None,
    org_uuid: str | None,
) -> None:
    """Discover and process transcripts interactively.

    Shows all transcript files found for the current project in a tabular view.
    Enter a number to process a transcript into a git repository.

    Use --all to show transcripts from all projects.
    Use --web to include web sessions from the Claude API.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from agentgit import DiscoveredWebSession, discover_transcripts_enriched, find_git_root

    console = Console()
    home = Path.home()

    if all_projects:
        # Discover from all projects
        transcripts = discover_transcripts_enriched(all_projects=True)
        header_path = "all projects"
    else:
        if project is None:
            project = find_git_root()
            if project is None and not include_web:
                raise click.ClickException(
                    "Not in a git repository. Use --project to specify a project path, --all for all projects, or --web for web sessions."
                )
        if project:
            transcripts = discover_transcripts_enriched(project)
            header_path = str(project)
        else:
            transcripts = []
            header_path = "web sessions"

    # Fetch web sessions if requested
    web_sessions: list[DiscoveredWebSession] = []
    credentials: tuple[str, str] | None = None

    if include_web:
        from agentgit.plugins import get_configured_plugin_manager

        pm = get_configured_plugin_manager()

        # Resolve credentials through plugin
        credentials = pm.hook.agentgit_resolve_web_credentials(
            token=token, org_uuid=org_uuid
        )

        if credentials:
            console.print("[dim]Fetching web sessions...[/dim]")
            # Discover web sessions through plugins
            for sessions in pm.hook.agentgit_discover_web_sessions(
                project_path=project, credentials=credentials
            ):
                web_sessions.extend(sessions)
        else:
            console.print("[yellow]Warning: Could not resolve web session credentials.[/yellow]")

    # Combine local transcripts and web sessions
    # Create a unified list with type information
    all_items: list[tuple[str, Any]] = []  # ("local", transcript) or ("web", web_session)

    for t in transcripts:
        all_items.append(("local", t))
    for ws in web_sessions:
        all_items.append(("web", ws))

    if not all_items:
        if include_web:
            msg = "No transcripts or web sessions found."
        elif all_projects:
            msg = "No transcripts found."
        else:
            msg = "No transcripts found for this project."
        console.print(f"[yellow]{msg}[/yellow]")
        return

    # Filter by type if specified
    if filter_type:
        all_items = [
            (item_type, item)
            for item_type, item in all_items
            if filter_type.lower() in item.format_type.lower()
        ]
        if not all_items:
            console.print(
                f"[yellow]No transcripts found matching type '{filter_type}'.[/yellow]"
            )
            return

    # Display header
    header_suffix = " (including web)" if include_web and web_sessions else ""
    console.print(
        Panel(f"[bold]agentgit discover[/bold] - {header_path}{header_suffix}", border_style="blue")
    )
    console.print()

    # Build unified table
    item_count = len(all_items)
    count_label = "item" if item_count == 1 else "items"
    table = Table(title=f"{item_count} {count_label}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="magenta")
    table.add_column("Path", style="cyan")
    table.add_column("Modified", style="green")
    table.add_column("Size", style="yellow", justify="right")

    for i, (item_type, item) in enumerate(all_items, 1):
        if item_type == "local":
            # Convert path to ~/... format
            try:
                rel_path = "~/" + str(item.path.relative_to(home))
            except ValueError:
                rel_path = str(item.path)
            table.add_row(str(i), item.plugin_name, rel_path, item.mtime_formatted, item.size_human)
        else:
            # Web session
            table.add_row(
                str(i),
                item.plugin_name,
                item.path_display,
                item.mtime_formatted,
                item.size_human,
            )

    console.print(table)
    console.print()

    # If list-only mode, stop here
    if list_only:
        return

    # Interactive selection via number input
    try:
        choice = click.prompt(
            "Enter number to process (or press Enter to cancel)",
            type=str,
            default="",
            show_default=False,
        )
    except click.Abort:
        console.print("[dim]Cancelled.[/dim]")
        return

    if not choice.strip():
        console.print("[dim]Cancelled.[/dim]")
        return

    try:
        idx = int(choice)
        if idx < 1 or idx > len(all_items):
            console.print(f"[red]Invalid number. Enter 1-{len(all_items)}.[/red]")
            return
    except ValueError:
        console.print("[red]Invalid input. Enter a number.[/red]")
        return

    item_type, selected = all_items[idx - 1]

    if item_type == "local":
        # Process local transcript
        _process_local_transcript(console, selected)
    else:
        # Process web session
        if credentials is None:
            console.print("[red]Cannot process web session: credentials not available.[/red]")
            return
        _process_web_session(console, selected, credentials)


def _process_local_transcript(console: Any, selected: Any) -> None:
    """Process a local transcript file."""
    from agentgit import transcript_to_repo

    console.print()
    console.print(f"[bold]Processing[/bold] {selected.path.name}...")

    output_dir = get_default_output_dir(selected.path)
    try:
        repo, repo_path, transcript = transcript_to_repo(
            selected.path,
            output_dir=output_dir,
        )

        # Count commits and files
        commit_count = sum(1 for _ in repo.iter_commits())
        prompt_count = len(transcript.prompt_responses)
        file_count = len(set(op.file_path for op in transcript.operations))

        console.print(f"[green]Created git repository at[/green] {repo_path}")
        console.print(f"  - {commit_count} commits ({prompt_count} prompts)")
        console.print(f"  - {file_count} files modified")

    except Exception as e:
        console.print(f"[red]Error processing transcript:[/red] {e}")
        raise click.ClickException(str(e))


def _process_web_session(
    console: Any,
    selected: Any,
    credentials: tuple[str, str],
) -> None:
    """Process a web session from the Claude API."""
    import json
    import tempfile

    from agentgit.plugins import get_configured_plugin_manager

    console.print()
    console.print(f"[bold]Processing web session[/bold] {selected.name}...")

    # Fetch the session data through plugin
    console.print("[dim]Fetching session data...[/dim]")
    pm = get_configured_plugin_manager()
    entries = pm.hook.agentgit_fetch_web_session(
        session_id=selected.session_id, credentials=credentials
    )

    if not entries:
        raise click.ClickException("No transcript entries found in session.")

    # Create a temporary transcript file for processing
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        prefix=f"web-session-{selected.session_id[:8]}-",
    ) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        temp_path = Path(f.name)

    try:
        # Determine output directory
        if selected.project_path:
            output_dir = Path.home() / ".agentgit" / "projects" / encode_path_as_name(
                Path(selected.project_path)
            )
        else:
            # Fall back to session-based naming
            output_dir = Path.home() / ".agentgit" / "web-sessions" / selected.session_id

        _run_process(
            transcripts=[temp_path],
            output=output_dir,
            plugin_type=None,
            author="Agent",
            email="agent@local",
            source_repo=None,
        )

    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


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
