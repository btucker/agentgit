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

    Args:
        transcript_path: Path to the transcript file.

    Returns:
        Path to the default output directory.

    Raises:
        click.ClickException: If no git repository can be found.
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

    # No git repository found
    raise click.ClickException(
        f"Could not determine git repository for transcript: {transcript_path}\n"
        "Make sure you're running this command from within a git repository."
    )


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


def _find_best_match_by_suffix(
    path_parts: tuple[str, ...], matches: list[Path], repo_path: Path
) -> Path | None:
    """Find the best matching file by comparing path suffixes.

    Args:
        path_parts: Parts of the path to match (from Path.parts).
        matches: List of candidate paths in the repo.
        repo_path: Root path of the repo (for computing relative paths).

    Returns:
        Best matching path, or None if no good match found.
    """
    best_match = None
    best_score = 0

    for match in matches:
        match_parts = match.relative_to(repo_path).parts
        # Count matching path components from the end
        score = 0
        for path_part, match_part in zip(reversed(path_parts), reversed(match_parts)):
            if path_part == match_part:
                score += 1
            else:
                break
        if score > best_score:
            best_score = score
            best_match = match

    return best_match


def _translate_single_path(arg: str, repo_path: Path, git_root: Path | None) -> str:
    """Translate a single path argument to its equivalent in the agentgit repo.

    Args:
        arg: The argument to translate (could be a path or other argument).
        repo_path: Path to the agentgit repository.
        git_root: Path to the git root of the current project (or None).

    Returns:
        Translated path, or original argument if not translatable.
    """
    # Skip options
    if arg.startswith("-"):
        return arg

    # Try to resolve the path
    local_path = Path(arg)

    # If path doesn't exist in cwd, try relative to git root
    if not local_path.exists() and git_root:
        potential_path = git_root / arg
        if potential_path.exists():
            local_path = potential_path

    # Search for the filename in agentgit repo
    filename = Path(arg).name
    matches = list(repo_path.rglob(filename))

    if not matches:
        # No matches found - keep original
        return arg
    elif len(matches) == 1:
        # Single match - use it
        return str(matches[0].relative_to(repo_path))
    else:
        # Multiple matches - find best match by path suffix
        # Use local_path if it exists, otherwise use arg
        path_parts = local_path.parts if local_path.exists() else Path(arg).parts
        best_match = _find_best_match_by_suffix(path_parts, matches, repo_path)
        if best_match:
            return str(best_match.relative_to(repo_path))
        else:
            return arg


def translate_paths_for_agentgit_repo(args: list[str], repo_path: Path) -> list[str]:
    """Translate local file paths to their equivalents in the agentgit repo.

    The agentgit repo uses normalized paths (common prefix stripped).
    This function finds matching files by filename.
    """
    from agentgit import find_git_root

    git_root = find_git_root()
    return [_translate_single_path(arg, repo_path, git_root) for arg in args]


def run_git_passthrough(args: list[str]) -> None:
    """Run a git command on the agentgit repo."""
    import os
    import subprocess

    repo_path = get_agentgit_repo_path()
    if not repo_path or not repo_path.exists():
        raise click.ClickException(
            "No agentgit repository found. Run 'agentgit' first to create one."
        )

    # Smart path translation: only translate arguments that are actual file paths
    # This avoids translating branch names, refs, etc.
    from agentgit import find_git_root

    git_root = find_git_root()
    translated_args = []

    for arg in args:
        # Skip options (start with -)
        if arg.startswith("-"):
            translated_args.append(arg)
            continue

        # Check if this looks like a file path that exists
        local_path = Path(arg)

        # Try relative to cwd or git root
        if local_path.exists() and local_path.is_file():
            # This is an actual file - translate it
            translated_args.append(_translate_single_path(arg, repo_path, git_root))
        elif git_root and (git_root / arg).exists():
            # Exists relative to git root - translate it
            translated_args.append(_translate_single_path(arg, repo_path, git_root))
        else:
            # Not a file path - pass through unchanged
            # This handles branch names, commit SHAs, refs, etc.
            translated_args.append(arg)

    # Set up less with ANSI color support as the pager
    # -R: interpret ANSI color codes
    # -F: quit if output fits on one screen
    # -X: don't clear screen on exit
    env = os.environ.copy()
    env["GIT_PAGER"] = "less -RFX"

    # Run git with -C to specify the agentgit repo directory
    # Use --paginate to enable paging for commands that support it
    # Explicitly inherit stdin/stdout/stderr to ensure terminal interaction works
    result = subprocess.run(
        ["git", "--paginate", "-C", str(repo_path)] + translated_args,
        env=env,
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
@click.argument(
    "transcript", type=click.Path(exists=True, path_type=Path), required=False
)
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
            transcripts[0],
            output,
            author,
            email,
            source_repo,
            enhancer=enhancer,
            enhance_model=enhance_model,
        )
    else:
        _run_process(
            transcripts,
            output,
            plugin_type,
            author,
            email,
            source_repo,
            enhancer=enhancer,
            enhance_model=enhance_model,
        )


def _resolve_enhance_config(
    output_dir: Path,
    enhancer: str | None = None,
    enhance_model: str | None = None,
) -> tuple[str | None, str | None, "EnhanceConfig | None"]:
    """Resolve enhancement configuration from CLI args and saved config.

    Args:
        output_dir: Output directory where config is stored.
        enhancer: CLI-provided enhancer name (if any).
        enhance_model: CLI-provided model name (if any).

    Returns:
        Tuple of (effective_enhancer, effective_model, enhance_config).
    """
    from agentgit.config import load_config
    from agentgit.enhance import EnhanceConfig

    # Load saved config and merge with CLI options
    saved_config = load_config(output_dir)

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

    return effective_enhancer, effective_model, enhance_config


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
    from agentgit.config import ProjectConfig, save_config

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcripts[0])

    # Resolve enhancement configuration
    effective_enhancer, effective_model, enhance_config = _resolve_enhance_config(
        output, enhancer, enhance_model
    )

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
    session_branches = [b.name for b in repo.heads if b.name.startswith("session/")]

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
    from agentgit.config import ProjectConfig, save_config
    from agentgit.watcher import TranscriptWatcher

    # Use default output directory if not specified
    if output is None:
        output = get_default_output_dir(transcript)

    # Resolve enhancement configuration
    effective_enhancer, effective_model, enhance_config = _resolve_enhance_config(
        output, enhancer, enhance_model
    )

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
        msg = (
            "No transcripts found."
            if all_projects
            else "No transcripts found for this project."
        )
        console.print(f"[yellow]{msg}[/yellow]")
        return

    # Filter by type if specified
    if filter_type:
        transcripts = [
            t
            for t in transcripts
            if filter_type.lower() in t.format_type.lower()
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
    table.add_column("Name/Path", style="cyan", no_wrap=True, overflow="fold")
    table.add_column("Modified", style="green")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Status", style="blue", no_wrap=True)

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
            # Find the branch for this transcript by reading its session ref
            try:
                # Get the transcript's session ID from the filename
                transcript_id = t.path.stem  # Filename without extension

                # Check if a session ref file exists
                session_ref_path = (
                    output_dir / ".git" / "refs" / "sessions" / transcript_id
                )
                if session_ref_path.exists():
                    # Read the symbolic ref and extract the branch name
                    ref_content = session_ref_path.read_text().strip()
                    if ref_content.startswith("ref: refs/heads/"):
                        status = ref_content[len("ref: refs/heads/") :]
            except (OSError, IOError):
                status = "REPO"

        table.add_row(
            str(i), t.plugin_name, name, t.mtime_formatted, t.size_human, status
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
        if idx < 1 or idx > len(transcripts):
            console.print(f"[red]Invalid number. Enter 1-{len(transcripts)}.[/red]")
            return
    except ValueError:
        console.print("[red]Invalid input. Enter a number.[/red]")
        return

    selected = transcripts[idx - 1]

    # Process the selected transcript
    console.print()
    console.print(
        f"[bold]Processing[/bold] {selected.display_name or selected.path.name}..."
    )

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
    ctx.invoke(
        sessions,
        project=project,
        all_projects=all_projects,
        list_only=list_only,
        filter_type=filter_type,
    )


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
def blame(
    file_path: str, lines: str | None, no_context: bool, session: str | None
) -> None:
    """Show git blame with agent context for each line.

    By default, blames your code repo and maps commits to session branches.
    Falls back to agentgit repo if not in a code repo.

    Examples:

        agentgit blame auth.py
        agentgit blame src/utils.py -L 10,20
        agentgit blame auth.py --session session/claude-code/add-auth
    """
    from agentgit.cmd.blame import blame_command

    blame_command(
        file_path=file_path,
        lines=lines,
        no_context=no_context,
        session=session,
        get_agentgit_repo_path=get_agentgit_repo_path,
        translate_paths_for_agentgit_repo=translate_paths_for_agentgit_repo,
    )


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
            level=logging.DEBUG, format="%(levelname)s [%(name)s] %(message)s"
        )
        click.echo("Debug logging enabled - will show session data structure\n")

    from agentgit.formats.claude_code_web import (
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

        with open(dump_json, "w") as f:
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
                output = (
                    Path.home()
                    / ".agentgit"
                    / "projects"
                    / encode_path_as_name(Path(project_path))
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


@main.command()
def repo() -> None:
    """Print the path to the agentgit repository for the current project.

    This command finds the git root of the current directory and prints
    the path to the corresponding agentgit repository using the repo ID
    (first 12 chars of the root commit hash).

    Example:

        cd /path/to/my/project
        agentgit repo
        # Outputs: /Users/username/.agentgit/projects/a12e842947dd
    """
    from agentgit import find_git_root

    git_root = find_git_root()
    if not git_root:
        raise click.ClickException(
            "Not in a git repository. Please run this command from within a git repository."
        )

    # Get repo ID from the first commit
    repo_id = get_repo_id(git_root)
    if not repo_id:
        raise click.ClickException(
            f"Could not determine repository ID for: {git_root}\n"
            "Make sure the repository has at least one commit."
        )

    agentgit_repo = Path.home() / ".agentgit" / "projects" / repo_id

    if not agentgit_repo.exists():
        raise click.ClickException(
            f"No agentgit repository found for this project.\n"
            f"Expected location: {agentgit_repo}\n"
            f"Repository ID: {repo_id}\n"
            f"Run 'agentgit process' first to create the repository."
        )

    click.echo(str(agentgit_repo))


@main.command()
@click.argument("shell", type=click.Choice(["bash", "zsh"], case_sensitive=False))
def completion(shell: str) -> None:
    """Output shell completion script for agentgit.

    This outputs a completion script that delegates to git's native completion
    for git passthrough commands.

    Examples:

        # Install for current session
        source <(agentgit completion bash)

        # Install permanently (bash)
        agentgit completion bash >> ~/.bashrc

        # Install permanently (zsh)
        agentgit completion zsh >> ~/.zshrc
    """
    if shell == "bash":
        script = '''# Bash completion for agentgit
# This delegates to git's existing completion for git passthrough commands

# Wrapper function that sets up git context for agentgit repo
_agentgit_git_wrapper() {
    # Find the agentgit repo path
    local agentgit_repo=""
    local project_name=""
    local git_root=$(git rev-parse --show-toplevel 2>/dev/null)

    if [ -n "$git_root" ]; then
        # Get repo ID (first 12 chars of root commit)
        local repo_id=$(git -C "$git_root" rev-list --max-parents=0 HEAD 2>/dev/null | head -1 | cut -c1-12)
        if [ -n "$repo_id" ]; then
            agentgit_repo="$HOME/.agentgit/projects/$repo_id"
        fi
    fi

    # If agentgit repo exists, provide completions
    if [ -d "$agentgit_repo/.git" ]; then
        # Check if git completion is available
        if declare -f __git_main &>/dev/null; then
            # Use git's native completion
            local saved_pwd="$PWD"
            local saved_git_dir="$__git_dir"

            cd "$agentgit_repo" 2>/dev/null
            __git_dir="$agentgit_repo/.git"

            __git_main

            __git_dir="$saved_git_dir"
            cd "$saved_pwd" 2>/dev/null
        else
            # Fallback: provide basic branch/ref completion
            local git_cmd="${words[1]}"
            case "$git_cmd" in
                log|show|diff|checkout|switch|rebase|merge|cherry-pick)
                    # Complete with branches and refs
                    local refs=$(git -C "$agentgit_repo" for-each-ref --format='%(refname:short)' refs/heads/ refs/remotes/ refs/tags/ 2>/dev/null)
                    COMPREPLY=( $(compgen -W "$refs" -- "$cur") )
                    ;;
                branch)
                    local branches=$(git -C "$agentgit_repo" for-each-ref --format='%(refname:short)' refs/heads/ 2>/dev/null)
                    COMPREPLY=( $(compgen -W "$branches" -- "$cur") )
                    ;;
                *)
                    # Default to file completion
                    _filedir
                    ;;
            esac
        fi
    fi
}

_agentgit() {
    local cur prev words cword
    _init_completion || return

    # Known agentgit commands
    local agentgit_commands="process sessions discover repo completion"

    # If we're completing the first argument
    if [ $cword -eq 1 ]; then
        # Offer both agentgit commands and git commands
        local git_commands=$(git help -a 2>/dev/null | awk '/^  [a-z]/ {print $1}')
        COMPREPLY=( $(compgen -W "$agentgit_commands $git_commands" -- "$cur") )
        return 0
    fi

    # If the first argument is an agentgit command, use default completion
    case "${words[1]}" in
        process|sessions|discover)
            # Use default file completion
            _filedir
            return 0
            ;;
        repo)
            # No arguments for repo command
            return 0
            ;;
        completion)
            # Complete shell types
            COMPREPLY=( $(compgen -W "bash zsh" -- "$cur") )
            return 0
            ;;
        *)
            # For git passthrough commands, use wrapper that sets git context
            # Replace command name with git in the completion arrays
            local saved_word0="${words[0]}"
            local saved_comp_word0="${COMP_WORDS[0]}"

            words[0]="git"
            COMP_WORDS[0]="git"

            # Call our wrapper that handles the agentgit repo context
            _agentgit_git_wrapper

            # Restore command name
            COMP_WORDS[0]="$saved_comp_word0"
            words[0]="$saved_word0"
            return 0
            ;;
    esac
}

complete -F _agentgit agentgit
complete -F _agentgit agit
'''
    else:  # zsh
        script = '''#compdef agentgit agit

# Zsh completion for agentgit and agit
# This delegates to git's existing completion for git passthrough commands

_agentgit() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    # If we're at the first argument position, show agentgit + git commands
    if (( CURRENT == 2 )); then
        local agentgit_commands=(
            'process:Process agent transcripts into git repositories'
            'sessions:Manage and list web sessions'
            'discover:Discover transcripts for the current project'
            'repo:Print path to agentgit repository'
            'completion:Output shell completion script'
        )
        local git_commands
        git_commands=(${(f)"$(git help -a 2>/dev/null | awk '/^  [a-z]/ {print $1}')"})

        _describe -t agentgit-commands 'agentgit commands' agentgit_commands
        _describe -t git-commands 'git commands' git_commands
        return 0
    fi

    # For positions after the first argument, provide git-aware completions
    # Find the agentgit repo path using repo ID
    local agentgit_repo=""
    local git_root=$(git rev-parse --show-toplevel 2>/dev/null)

    if [[ -n "$git_root" ]]; then
        local repo_id=$(git -C "$git_root" rev-list --max-parents=0 HEAD 2>/dev/null | head -1 | cut -c1-12)
        if [[ -n "$repo_id" ]]; then
            agentgit_repo="$HOME/.agentgit/projects/$repo_id"
        fi
    fi

    # If repo exists, provide git-aware completions
    if [[ -d "$agentgit_repo/.git" ]]; then
        local -a branches
        branches=(${(f)"$(git -C "$agentgit_repo" for-each-ref --format='%(refname:short)' refs/heads/ refs/remotes/ refs/tags/ 2>/dev/null)"})
        _describe -t branches 'git branches' branches
    else
        _files
    fi
}

# Manually register the completion (needed when sourcing via process substitution)
compdef _agentgit agit
compdef _agentgit agentgit
'''

    click.echo(script)


if __name__ == "__main__":
    main()
