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


def get_repo_id(code_repo: Path) -> str | None:
    """Get the repository ID from the first (root) commit SHA.

    The repo ID is the first 12 characters of the root commit SHA.
    This provides a stable identifier that survives repo moves/renames.

    Args:
        code_repo: Path to the code repository.

    Returns:
        12-character repo ID, or None if not a git repo or has no commits.
    """
    from git import Repo
    from git.exc import InvalidGitRepositoryError, GitCommandError

    try:
        repo = Repo(code_repo)
        # Get root commit(s) - repos can have multiple roots from orphan branches
        # Use HEAD's root commit
        root_commits = list(repo.iter_commits(rev="HEAD", max_parents=0))
        if root_commits:
            return root_commits[0].hexsha[:12]
        return None
    except (InvalidGitRepositoryError, GitCommandError, ValueError):
        return None


def get_default_output_dir(transcript_path: Path) -> Path:
    """Get the default output directory for a transcript.

    Uses ~/.agentgit/projects/{repo_id} where repo_id is the first 12
    characters of the repository's root commit SHA. This provides a stable
    identifier across renames and clones.

    The repo is determined by:
    1. Asking plugins to get the project name from transcript location,
       then finding the corresponding git repo (e.g., Claude Code returns
       "-Users-name-project" which maps to a git root)
    2. Using the current directory's git root
    3. Falling back to path encoding for non-git directories

    Args:
        transcript_path: Path to the transcript file.

    Returns:
        Path to the default output directory.
    """
    from agentgit import find_git_root
    from agentgit.plugins import get_configured_plugin_manager

    pm = get_configured_plugin_manager()

    # Strategy 1: Ask plugins for project name, then find git repo
    project_name = pm.hook.agentgit_get_project_name(transcript_path=transcript_path)
    if project_name:
        # Project name might be path-encoded like "-Users-name-project"
        # Try to decode it back to a path and check if it's a git repo
        if project_name.startswith("-"):
            # Convert "-Users-name-project" -> "/Users/name/project"
            potential_path = Path(project_name.replace("-", "/", 1))
            if potential_path.exists():
                repo_id = get_repo_id(potential_path)
                if repo_id:
                    return Path.home() / ".agentgit" / "projects" / repo_id

    # Strategy 2: Use current directory's git root
    git_root = find_git_root()
    if git_root:
        repo_id = get_repo_id(git_root)
        if repo_id:
            return Path.home() / ".agentgit" / "projects" / repo_id

    # Strategy 3: Fall back to path encoding for non-git directories
    if project_name:
        return Path.home() / ".agentgit" / "projects" / project_name

    # Last resort: encode the transcript's parent directory
    fallback_name = encode_path_as_name(transcript_path.resolve().parent)
    return Path.home() / ".agentgit" / "projects" / fallback_name


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
    from agentgit import find_git_root

    git_root = find_git_root()
    translated = []

    for arg in args:
        # Skip options
        if arg.startswith("-"):
            translated.append(arg)
            continue

        # Try to resolve the path
        local_path = Path(arg)

        # If path doesn't exist in cwd, try relative to git root
        if not local_path.exists() and git_root:
            potential_path = git_root / arg
            if potential_path.exists():
                local_path = potential_path

        # If still doesn't exist, try to find by filename in agentgit repo
        if not local_path.exists():
            # Search for the filename in agentgit repo
            filename = Path(arg).name
            matches = list(repo_path.rglob(filename))

            if len(matches) == 1:
                translated.append(str(matches[0].relative_to(repo_path)))
                continue
            elif len(matches) > 1:
                # Multiple matches - try to find best match by path suffix
                arg_parts = Path(arg).parts
                best_match = None
                best_score = 0
                for match in matches:
                    match_parts = match.relative_to(repo_path).parts
                    score = 0
                    for ap, mp in zip(reversed(arg_parts), reversed(match_parts)):
                        if ap == mp:
                            score += 1
                        else:
                            break
                    if score > best_score:
                        best_score = score
                        best_match = match
                if best_match:
                    translated.append(str(best_match.relative_to(repo_path)))
                    continue

            # No matches found - keep original
            translated.append(arg)
            continue

        # Path exists - try to find it in the agentgit repo by filename
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
    "--enhancer",
    default=None,
    help="Enhancer plugin ('rules' for heuristics, 'claude_code' for AI). Saved per-project.",
)
@click.option(
    "--llm-model",
    "enhance_model",
    default=None,
    help="Model for LLM enhancer (e.g., 'haiku', 'sonnet'). Saved per-project.",
)
def process(
    transcript: Path | None,
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
    watch: bool,
    enhancer: str | None,
    enhance_model: str | None,
) -> None:
    """Process a transcript into a git repository.

    If TRANSCRIPT is not provided, discovers and processes all transcripts
    for the current project, merging their operations by timestamp.

    With --watch, monitors the transcript file and automatically commits
    new operations as they are added (only works with a single transcript).

    Use --enhancer to generate better commit messages. The preference is saved
    per-project and used automatically on future runs.
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
            enhancer=enhancer, enhance_model=enhance_model
        )
    else:
        _run_process(
            transcripts, output, plugin_type, author, email, source_repo,
            enhancer=enhancer, enhance_model=enhance_model
        )


def _run_process(
    transcripts: list[Path],
    output: Path | None,
    plugin_type: str | None,
    author: str,
    email: str,
    source_repo: Path | None,
    enhancer: str | None = None,
    enhance_model: str | None = None,
) -> None:
    """Run processing of one or more transcripts."""
    from agentgit import build_repo_grouped, parse_transcript
    from agentgit.config import ProjectConfig, load_config, save_config
    from agentgit.enhance import EnhanceConfig

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcripts[0])

    # Load saved config and merge with CLI options
    saved_config = load_config(output)
    # Auto-set enhancer to 'llm' if --llm-model is provided
    if enhance_model and not enhancer:
        enhancer = "llm"
    effective_enhancer = enhancer or saved_config.enhancer
    effective_model = enhance_model or saved_config.enhance_model or "haiku"

    # Configure enhancement if an enhancer is set
    enhance_config = None
    if effective_enhancer:
        enhance_config = EnhanceConfig(
            enhancer=effective_enhancer, model=effective_model, enabled=True
        )
        click.echo(f"Enhancement: {effective_enhancer} (model: {effective_model})")

    if len(transcripts) == 1:
        click.echo(f"Processing transcript: {transcripts[0]}")
    else:
        click.echo(f"Processing {len(transcripts)} transcripts:")
        for t in transcripts:
            click.echo(f"  - {t.name}")

    # Process each transcript into its own session branch
    repo = None
    total_prompts = 0
    total_operations = 0
    for transcript_path in transcripts:
        parsed = parse_transcript(transcript_path, plugin_type=plugin_type)

        if not parsed.prompt_responses:
            click.echo(f"  Skipping {transcript_path.name}: no operations found")
            continue

        # Extract agent name from format (e.g., "claude_code_jsonl" -> "claude-code")
        agent_name = None
        if parsed.source_format:
            # Remove suffixes like "_jsonl", "_web"
            agent_name = parsed.source_format.replace("_jsonl", "").replace("_web", "")

        # Build into session branch
        repo, repo_path, _ = build_repo_grouped(
            prompt_responses=parsed.prompt_responses,
            output_dir=output,
            author_name=author,
            author_email=email,
            enhance_config=enhance_config,
            session_id=parsed.session_id or transcript_path.stem,
            agent_name=agent_name,
        )

        total_prompts += len(parsed.prompts)
        total_operations += len(parsed.operations)

    if repo is None:
        click.echo("No operations found in any transcript")
        return

    # Save new preferences if explicitly provided (after repo exists)
    if enhancer is not None or enhance_model is not None:
        new_config = ProjectConfig(
            enhancer=enhancer or saved_config.enhancer,
            enhance_model=enhance_model or saved_config.enhance_model,
        )
        save_config(output, new_config)

    # Count branches (sessions)
    session_branches = [b.name for b in repo.heads if b.name.startswith('session/')]

    click.echo(f"Created git repository at: {repo_path}")
    click.echo(f"  Sessions (branches): {len(session_branches)}")
    click.echo(f"  Total prompts: {total_prompts}")
    click.echo(f"  Total operations: {total_operations}")
    click.echo(f"  Total commits: {len(list(repo.iter_commits('--all')))}")


def _run_watch_mode(
    transcript: Path,
    output: Path | None,
    author: str,
    email: str,
    source_repo: Path | None,
    enhancer: str | None = None,
    enhance_model: str | None = None,
) -> None:
    """Run in watch mode."""
    from agentgit.config import ProjectConfig, load_config, save_config
    from agentgit.enhance import EnhanceConfig
    from agentgit.watcher import TranscriptWatcher

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcript)

    # Load saved config and merge with CLI options
    saved_config = load_config(output)
    # Auto-set enhancer to 'llm' if --llm-model is provided
    if enhance_model and not enhancer:
        enhancer = "llm"
    effective_enhancer = enhancer or saved_config.enhancer
    effective_model = enhance_model or saved_config.enhance_model or "haiku"

    # Configure enhancement if an enhancer is set
    enhance_config = None
    if effective_enhancer:
        enhance_config = EnhanceConfig(
            enhancer=effective_enhancer, model=effective_model, enabled=True
        )
        click.echo(f"Enhancement: {effective_enhancer} (model: {effective_model})")

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
        enhance_config=enhance_config,
    )

    # Initial build status
    from git import Repo
    from git.exc import InvalidGitRepositoryError

    watcher.start()

    # Save new preferences if explicitly provided (after repo exists)
    if enhancer is not None or enhance_model is not None:
        new_config = ProjectConfig(
            enhancer=enhancer or saved_config.enhancer,
            enhance_model=enhance_model or saved_config.enhance_model,
        )
        save_config(output, new_config)

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
@click.option(
    "--project",
    type=click.Path(exists=True, path_type=Path),
    help="Project path to find sessions for. Defaults to current git repo.",
)
@click.option(
    "--all",
    "all_projects",
    is_flag=True,
    help="Show sessions from all projects, not just the current one.",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    help="List sessions without interactive selection.",
)
@click.option(
    "--type",
    "filter_type",
    type=str,
    help="Filter by session type (e.g., claude_code, codex, claude_code_web).",
)
def sessions(
    project: Path | None,
    all_projects: bool,
    list_only: bool,
    filter_type: str | None,
) -> None:
    """List and process AI coding sessions.

    Shows all available sessions for the current project in a tabular view.
    This includes both local transcript files and web sessions (if credentials
    are available). Sessions that have been processed into the git repo are
    marked with [REPO]. Enter a number to process a session.

    Use --all to show sessions from all projects.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from agentgit import discover_transcripts_enriched, find_git_root

    console = Console()
    home = Path.home()

    if all_projects:
        # Discover from all projects
        transcripts = discover_transcripts_enriched(all_projects=True)
        header_path = "all projects"
    else:
        if project is None:
            project = find_git_root()
            if project is None:
                raise click.ClickException(
                    "Not in a git repository. Use --project to specify a project path, or --all for all projects."
                )
        transcripts = discover_transcripts_enriched(project)
        header_path = str(project)

    if not transcripts:
        msg = "No transcripts found." if all_projects else "No transcripts found for this project."
        console.print(f"[yellow]{msg}[/yellow]")
        return

    # Filter by type if specified
    if filter_type:
        transcripts = [
            t for t in transcripts if filter_type.lower() in t.format_type.lower()
            or filter_type.lower() in t.plugin_name.lower()
        ]
        if not transcripts:
            console.print(
                f"[yellow]No transcripts found matching type '{filter_type}'.[/yellow]"
            )
            return

    # Display header
    console.print(
        Panel(f"[bold]agentgit sessions[/bold] - {header_path}", border_style="blue")
    )
    console.print()

    # Build unified table
    count_label = "session" if len(transcripts) == 1 else "sessions"
    table = Table(title=f"{len(transcripts)} {count_label}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="magenta")
    table.add_column("Name/Path", style="cyan")
    table.add_column("Modified", style="green")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Status", style="blue", width=6)

    for i, t in enumerate(transcripts, 1):
        # Use display name if available, otherwise path
        if t.display_name:
            name = t.display_name
        else:
            # Convert path to ~/... format
            try:
                name = "~/" + str(t.path.relative_to(home))
            except ValueError:
                name = str(t.path)

        # Check if this session has been processed into git repo
        output_dir = get_default_output_dir(t.path)
        status = ""
        if output_dir.exists() and (output_dir / ".git").exists():
            status = "REPO"

        table.add_row(str(i), t.plugin_name, name, t.mtime_formatted, t.size_human, status)

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
        if idx < 1 or idx > len(transcripts):
            console.print(f"[red]Invalid number. Enter 1-{len(transcripts)}.[/red]")
            return
    except ValueError:
        console.print("[red]Invalid input. Enter a number.[/red]")
        return

    selected = transcripts[idx - 1]

    # Process the selected transcript
    console.print()
    console.print(f"[bold]Processing[/bold] {selected.display_name or selected.path.name}...")

    output_dir = get_default_output_dir(selected.path)
    try:
        from agentgit import transcript_to_repo

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


@main.group()
def config() -> None:
    """Manage agentgit configuration.

    Examples:

    \b
      agentgit config agents          # List installed agent plugins
      agentgit config agents add pkg  # Install an agent plugin
    """
    pass


@config.group(invoke_without_command=True)
@click.pass_context
def agents(ctx: click.Context) -> None:
    """Manage agent plugins.

    Without a subcommand, lists all available plugins.

    Examples:

    \b
      agentgit config agents                      # List all plugins
      agentgit config agents add agentgit-aider   # Install a plugin
      agentgit config agents remove agentgit-aider
    """
    if ctx.invoked_subcommand is None:
        # Default behavior: list plugins
        ctx.invoke(agents_list)


@agents.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show additional details.")
def agents_list(verbose: bool = False) -> None:
    """List available agent plugins."""
    from agentgit.plugins import list_configured_plugins

    plugins = list_configured_plugins()

    if not plugins:
        click.echo("No agent plugins found.")
        return

    click.echo("Available agent plugins:\n")

    for plugin in plugins:
        name = plugin.get("name", "unknown")
        description = plugin.get("description", "No description")
        source = plugin.get("source", "unknown")

        if verbose:
            click.echo(f"  {name}: {description}")
            click.echo(f"    source: {source}")
            click.echo()
        else:
            source_tag = f" [{source}]" if source != "builtin" else ""
            click.echo(f"  {name}: {description}{source_tag}")


@agents.command("add")
@click.argument("package")
def agents_add(package: str) -> None:
    """Install an agent plugin package.

    Installs the package using pip/uv and registers it with agentgit.
    The package must define an agentgit entry point.

    Examples:

    \b
      agentgit config agents add agentgit-aider
      agentgit config agents add agentgit-cursor
    """
    from agentgit.plugins import add_plugin

    click.echo(f"Installing {package}...")
    success, message = add_plugin(package)

    if success:
        click.echo(message)
    else:
        raise click.ClickException(message)


@agents.command("remove")
@click.argument("package")
def agents_remove(package: str) -> None:
    """Uninstall an agent plugin package.

    Uninstalls the package and removes it from agentgit's registry.

    Examples:

    \b
      agentgit config agents remove agentgit-aider
    """
    from agentgit.plugins import remove_plugin

    success, message = remove_plugin(package)

    if success:
        click.echo(message)
    else:
        raise click.ClickException(message)


# Alias for backward compatibility
@main.command(hidden=True)
@click.option("--project", type=click.Path(exists=True, path_type=Path))
@click.option("--all", "all_projects", is_flag=True)
@click.option("--list", "list_only", is_flag=True)
@click.option("--type", "filter_type", type=str)
def discover(
    project: Path | None,
    all_projects: bool,
    list_only: bool,
    filter_type: str | None,
) -> None:
    """Alias for 'sessions' command (deprecated, use 'sessions' instead)."""
    from click import get_current_context
    ctx = get_current_context()
    ctx.invoke(sessions, project=project, all_projects=all_projects,
               list_only=list_only, filter_type=filter_type)


@main.command()
@click.argument("file_path", type=str)
@click.option(
    "--lines",
    "-L",
    type=str,
    help="Show blame for specific line range (e.g., '10,20' or '10,+5').",
)
@click.option(
    "--no-context",
    is_flag=True,
    help="Disable showing agent context inline.",
)
@click.option(
    "--session",
    "-s",
    type=str,
    help="Specific session branch to blame (e.g., session/claude-code/add-auth).",
)
def blame(file_path: str, lines: str | None, no_context: bool, session: str | None) -> None:
    """Show git blame with agent context for each line.

    By default, blames your code repo and maps commits to session branches.
    Falls back to agentgit repo if not in a code repo.

    Examples:

        agentgit blame auth.py
        agentgit blame src/utils.py -L 10,20
        agentgit blame auth.py --session session/claude-code/add-auth
    """
    import re
    from git import Repo, GitCommandError
    from pathlib import Path
    from agentgit import find_git_root

    agentgit_repo_path = get_agentgit_repo_path()
    if not agentgit_repo_path or not agentgit_repo_path.exists():
        raise click.ClickException(
            "No agentgit repository found. Run 'agentgit' first to create one."
        )

    # Check if we're in a code repo
    code_repo = None
    try:
        code_repo_path = find_git_root(Path.cwd())
        if code_repo_path and code_repo_path != agentgit_repo_path:
            code_repo = Repo(code_repo_path)
            click.echo(f"Using code repo: {code_repo_path}", err=True)
    except Exception:
        pass

    # Decide which repo to blame
    if session:
        # Explicit session specified - use agentgit repo
        repo = Repo(agentgit_repo_path)
        translated = translate_paths_for_agentgit_repo([file_path], agentgit_repo_path)
        target_file = translated[0]
        blame_ref = session
        use_session_mapping = False
        click.echo(f"Blaming session: {session}", err=True)
    elif code_repo:
        # Use code repo and map to sessions
        repo = code_repo
        target_file = file_path
        blame_ref = 'HEAD'
        use_session_mapping = True
    else:
        # Fall back to agentgit repo
        repo = Repo(agentgit_repo_path)
        translated = translate_paths_for_agentgit_repo([file_path], agentgit_repo_path)
        target_file = translated[0]
        blame_ref = 'HEAD'
        use_session_mapping = False
        click.echo("No code repo found, using agentgit repo", err=True)

    try:
        # Parse line range if provided
        start_line = None
        end_line = None
        if lines:
            parts = lines.split(',')
            if len(parts) == 2:
                start_line = int(parts[0]) - 1  # GitPython uses 0-based indexing
                # Handle both "start,end" and "start,+count" formats
                if parts[1].startswith('+'):
                    end_line = start_line + int(parts[1])
                else:
                    end_line = int(parts[1]) - 1

        # Get blame data
        blame_data = repo.blame(blame_ref, target_file)
    except Exception as e:
        raise click.ClickException(f"Failed to get blame data: {e}")

    # Set up session mapping if needed
    agentgit_repo = None
    blob_to_session_cache = {}  # Cache blob SHA → session mapping

    if use_session_mapping:
        agentgit_repo = Repo(agentgit_repo_path)
        # Build index of blob SHAs to sessions
        click.echo("Indexing sessions...", err=True)
        blob_to_session_cache = _build_blob_to_session_index(agentgit_repo, agentgit_repo_path)

    def find_session_for_commit(commit, file_path_in_commit) -> tuple[str | None, str | None]:
        """Find which session branch wrote this blob.

        Returns: (session_name, context)
        """
        if not use_session_mapping:
            return None, None

        try:
            # Get the blob SHA for this file in this commit
            blob_sha = commit.tree[file_path_in_commit].hexsha

            # Look up in cache
            if blob_sha in blob_to_session_cache:
                return blob_to_session_cache[blob_sha]
        except (KeyError, AttributeError):
            pass

        return None, None

    def get_commit_context(commit) -> str | None:
        """Extract context from commit message."""
        try:
            message = commit.message

            # Extract context section
            context_match = re.search(
                r'Context:\s*\n(.+?)(?:\n\n|\nPrompt-Id:|\nOperation:|\nTimestamp:|\Z)',
                message,
                re.DOTALL
            )
            if context_match:
                context = context_match.group(1).strip()
                # Truncate to first sentence or 80 chars
                first_sentence = re.split(r'[.!?]\s', context)[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:77] + "..."
                return first_sentence
        except Exception:
            pass

        return None

    # Process and display blame output
    line_num = 0
    for commit, lines_list in blame_data:
        for line_content in lines_list:
            # Apply line range filter if specified
            if start_line is not None and (line_num < start_line or line_num > end_line):
                line_num += 1
                continue

            # Format blame line
            short_sha = commit.hexsha[:7]
            author = commit.author.name[:12]  # Truncate long names
            date = commit.committed_datetime.strftime('%Y-%m-%d')

            # Remove trailing newline from content
            content = line_content.rstrip('\n')

            # Find session if in mapping mode
            session_name, session_context = find_session_for_commit(commit, target_file)

            if session_name:
                # Format session name compactly: session/claude_code/foo -> cc/foo
                blame_line = _format_blame_line_with_session(
                    short_sha, session_name, session_context, content, no_context
                )
            else:
                # Standard blame format
                blame_line = f"{short_sha} ({author:12} {date:10}) {content}"

            click.echo(blame_line)

            line_num += 1


def _format_blame_line_with_session(
    sha: str,
    session_name: str,
    context: str | None,
    code: str,
    no_context: bool,
) -> str:
    """Format a blame line with session information.

    Format: {sha} {agent}/{branch}...: {context...}   {code}

    Terminal-width aware - wider terminals show more detail.
    """
    import shutil

    # Get terminal width, default to 80 if not available
    term_width = shutil.get_terminal_size((80, 24)).columns

    # Abbreviate session name: session/claude_code/foo -> cc/foo
    parts = session_name.split('/')
    if len(parts) >= 3 and parts[0] == 'session':
        agent = parts[1]
        # Abbreviate agent name to 2-3 chars
        if agent == 'claude_code':
            agent_abbrev = 'cc'
        elif agent == 'codex':
            agent_abbrev = 'cx'
        else:
            # Use first 2 chars for unknown agents
            agent_abbrev = agent[:2]

        branch = '/'.join(parts[2:])
        session_short = f"{agent_abbrev}/{branch}"
    else:
        session_short = session_name

    # Calculate available space for session+context
    # Format: "{sha} {session}: {context}   {code}"
    # Fixed parts: sha(7) + space(1) + ": " (2) + "   " (3) = 13
    fixed_space = 7 + 1 + 2 + 3
    code_space = len(code)
    available = term_width - fixed_space - code_space

    # If we have context and it's not disabled, include it
    if context and not no_context:
        # Reserve space for session (min 10 chars for "cc/branch")
        min_session = 10
        session_max = min(len(session_short), max(min_session, available // 3))
        context_max = available - session_max - 2  # -2 for ": "

        if context_max > 10:  # Only show context if we have reasonable space
            session_display = session_short[:session_max]
            if len(session_short) > session_max:
                session_display = session_display[:-2] + '..'

            context_display = context[:context_max]
            if len(context) > context_max:
                context_display = context_display[:-3] + '...'

            return f"{sha} {session_display}: {context_display}   {code}"

    # No context or not enough space - just show session
    session_max = max(10, available)
    session_display = session_short[:session_max]
    if len(session_short) > session_max:
        session_display = session_display[:-2] + '..'

    return f"{sha} {session_display}:   {code}"


def _build_blob_to_session_index(agentgit_repo: "Repo", agentgit_repo_path) -> dict[str, tuple[str, str]]:
    """Build an index mapping blob SHAs to sessions.

    Returns: {blob_sha: (session_name, context)}
    """
    import re
    from pathlib import Path

    blob_index = {}

    # Get path mapping to know how to look up files
    from agentgit.git_builder import normalize_file_paths

    # Find all session branches
    for branch in agentgit_repo.heads:
        if not branch.name.startswith('session/'):
            continue

        # Iterate through commits in this session
        for commit in agentgit_repo.iter_commits(branch.name):
            # Extract context from commit message
            context_match = re.search(
                r'Context:\s*\n(.+?)(?:\n\n|\nPrompt-Id:|\nOperation:|\nTimestamp:|\Z)',
                commit.message,
                re.DOTALL
            )
            context = None
            if context_match:
                context = context_match.group(1).strip()
                first_sentence = re.split(r'[.!?]\s', context)[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:77] + "..."
                context = first_sentence

            # Index all blobs in this commit's tree
            try:
                for item in commit.tree.traverse():
                    if item.type == 'blob':
                        blob_sha = item.hexsha
                        # Store first session that wrote this blob (could be overwritten)
                        if blob_sha not in blob_index:
                            blob_index[blob_sha] = (branch.name, context)
            except Exception:
                continue

    return blob_index


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
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging to inspect session data structure.",
)
@click.option(
    "--dump-json",
    type=click.Path(path_type=Path),
    help="Dump raw session JSON to this file for inspection.",
)
def web(
    session_id: str | None,
    output: Path | None,
    token: str | None,
    org_uuid: str | None,
    author: str,
    email: str,
    debug: bool,
    dump_json: Path | None,
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

        agentgit web --debug abc123     # Show all available session fields
    """
    import logging

    # Enable debug logging if requested
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s [%(name)s] %(message)s'
        )
        click.echo("Debug logging enabled - will show session data structure\n")

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

    # Dump raw JSON if requested
    if dump_json:
        import json
        with open(dump_json, 'w') as f:
            json.dump(session_data, f, indent=2)
        click.echo(f"Dumped raw session data to: {dump_json}")
        click.echo(f"Top-level keys: {', '.join(sorted(session_data.keys()))}")
        return

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
