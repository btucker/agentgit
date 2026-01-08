"""Blame command implementation for agentgit.

Maps lines in your code to the agent sessions that wrote them by using
git's native blame across all session branches.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import click

if TYPE_CHECKING:
    from git import Repo


class SessionBlameEntry(NamedTuple):
    """Blame information for a line from a session."""

    commit_sha: str
    session: str
    timestamp: datetime
    context: str | None


def blame_command(
    file_path: str,
    lines: str | None,
    no_context: bool,
    session: str | None,
    get_agentgit_repo_path,
    translate_paths_for_agentgit_repo,
) -> None:
    """Show git blame with agent context for each line.

    By default, blames your code repo and maps commits to session branches.
    For each line, finds the EARLIEST commit across all sessions that wrote it.
    Falls back to agentgit repo if not in a code repo.

    Examples:

        agentgit blame auth.py
        agentgit blame src/utils.py -L 10,20
        agentgit blame auth.py --session sessions/claude-code/add-auth

    Args:
        file_path: Path to the file to blame
        lines: Optional line range in format "start,end" or "start,+count"
        no_context: If True, don't show agent context
        session: Optional session branch to blame
        get_agentgit_repo_path: Function to get agentgit repo path
        translate_paths_for_agentgit_repo: Function to translate paths
    """
    from git import Repo

    from agentgit import find_git_root

    agentgit_repo_path = get_agentgit_repo_path()
    if not agentgit_repo_path or not agentgit_repo_path.exists():
        raise click.ClickException(
            "No agentgit repository found. Run 'agentgit' first to create one."
        )

    # Check if we're in a code repo
    code_repo = None
    code_repo_path = None
    try:
        code_repo_path = find_git_root(Path.cwd())
        if code_repo_path and code_repo_path != agentgit_repo_path:
            code_repo = Repo(code_repo_path)
            click.echo(f"Using code repo: {code_repo_path}", err=True)
    except Exception:
        pass

    # Decide which repo to blame
    if session:
        # Explicit session specified - use agentgit repo directly
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

    # Parse line range if provided
    start_line = None
    end_line = None
    if lines:
        parts = lines.split(',')
        if len(parts) == 2:
            start_line = int(parts[0]) - 1  # GitPython uses 0-based indexing
            if parts[1].startswith('+'):
                end_line = start_line + int(parts[1])
            else:
                end_line = int(parts[1]) - 1

    try:
        blame_data = repo.blame(blame_ref, target_file)
    except Exception as e:
        raise click.ClickException(f"Failed to get blame data: {e}")

    # Build session index if mapping mode
    session_index: dict[str, list[SessionBlameEntry]] = {}
    if use_session_mapping:
        agentgit_repo = Repo(agentgit_repo_path)
        click.echo("Building session index...", err=True)
        session_index = build_session_index(agentgit_repo, target_file)
        click.echo(f"Indexed {len(session_index)} unique lines", err=True)

    # Flatten blame data for line-by-line processing
    all_lines = []
    for commit, lines_list in blame_data:
        for line_content in lines_list:
            all_lines.append((commit, line_content))

    # Process and display blame output
    line_num = 0
    for commit, line_content in all_lines:
        # Apply line range filter if specified
        if start_line is not None and end_line is not None:
            if line_num < start_line or line_num > end_line:
                line_num += 1
                continue

        content = line_content.rstrip('\n')

        # Find session if in mapping mode
        session_entry = find_earliest_session(content, session_index)

        if session_entry:
            blame_line = format_blame_line_with_session(
                session_entry.commit_sha,
                session_entry.session,
                session_entry.context,
                content,
                no_context,
            )
        else:
            # Standard blame format (no session match)
            short_sha = commit.hexsha[:7]
            author = commit.author.name[:12]
            date = commit.committed_datetime.strftime('%Y-%m-%d')
            blame_line = f"{short_sha} ({author:12} {date:10}) {content}"

        click.echo(blame_line)
        line_num += 1


def build_session_index(
    agentgit_repo: "Repo",
    code_relative_path: str,
) -> dict[str, list[SessionBlameEntry]]:
    """Build an index of all lines across all session branches.

    Uses git's native blame on each session branch to find the commit
    that introduced each line.

    Args:
        agentgit_repo: The agentgit repository
        code_relative_path: Relative path of the file in code repo (e.g., 'src/cli.py')

    Returns:
        Dict mapping line content -> list of SessionBlameEntry from all sessions
    """
    results: dict[str, list[SessionBlameEntry]] = defaultdict(list)

    # Find all session branches
    session_branches = [b for b in agentgit_repo.heads if b.name.startswith('sessions/')]

    # Find all path variants for this file across sessions
    agentgit_paths = find_agentgit_paths(code_relative_path, session_branches)

    # Blame each session branch
    for branch in session_branches:
        for file_path in agentgit_paths:
            try:
                # Check if file exists in this branch
                branch.commit.tree[file_path]
            except KeyError:
                continue

            try:
                blame_data = agentgit_repo.blame(branch.name, file_path)
            except Exception:
                continue

            for commit, lines in blame_data:
                context = extract_context(commit)
                for line in lines:
                    line_stripped = line.rstrip('\n')
                    results[line_stripped].append(SessionBlameEntry(
                        commit_sha=commit.hexsha[:7],
                        session=branch.name,
                        timestamp=commit.committed_datetime,
                        context=context,
                    ))

    return dict(results)


def find_agentgit_paths(
    code_relative_path: str,
    session_branches: list,
) -> list[str]:
    """Find all path variants for a code file in the agentgit repo.

    Session branches may have files at different absolute paths like:
        Documents/projects/myapp/src/cli.py
        Users/name/Documents/projects/myapp/src/cli.py

    Args:
        code_relative_path: Relative path from code repo (e.g., 'src/cli.py')
        session_branches: List of session branches to search

    Returns:
        List of unique paths in agentgit that match the code file
    """
    all_paths = set()
    for branch in session_branches:
        try:
            for item in branch.commit.tree.traverse():
                if item.type == 'blob' and item.path.endswith(code_relative_path):
                    all_paths.add(item.path)
        except Exception:
            continue
    return list(all_paths)


def find_earliest_session(
    line_content: str,
    session_index: dict[str, list[SessionBlameEntry]],
) -> SessionBlameEntry | None:
    """Find the earliest session commit that wrote this line.

    Args:
        line_content: The line of code to look up
        session_index: Index from build_session_index()

    Returns:
        SessionBlameEntry for the earliest commit, or None if not found
    """
    entries = session_index.get(line_content)
    if not entries:
        return None

    # Return the earliest by timestamp
    return min(entries, key=lambda e: e.timestamp)


def extract_context(commit) -> str | None:
    """Extract context from commit message.

    Args:
        commit: GitPython commit object

    Returns:
        First sentence of context, or None if not found
    """
    message = commit.message
    match = re.search(
        r'Context:\s*\n(.+?)(?:\n\n|\nPrompt-Id:|\nOperation:|\nTimestamp:|\Z)',
        message,
        re.DOTALL
    )
    if match:
        context = match.group(1).strip()
        first_sentence = re.split(r'[.!?]\s', context)[0]
        if len(first_sentence) > 80:
            return first_sentence[:77] + "..."
        return first_sentence
    return None


def format_blame_line_with_session(
    sha: str,
    session_name: str,
    context: str | None,
    code: str,
    no_context: bool,
) -> str:
    """Format a blame line with session information.

    Format: {sha} {agent}/{branch}...: {context...}   {code}

    Terminal-width aware - wider terminals show more detail.

    Args:
        sha: Commit SHA (short form)
        session_name: Full session name like "sessions/claude_code/feature"
        context: Optional context from commit message
        code: The line of code
        no_context: If True, don't show context

    Returns:
        Formatted blame line
    """
    import shutil

    term_width = shutil.get_terminal_size((80, 24)).columns

    # Abbreviate session name: sessions/claude_code/foo -> cc/foo
    parts = session_name.split('/')
    if len(parts) >= 3 and parts[0] == 'sessions':
        agent = parts[1]
        if agent == 'claude_code':
            agent_abbrev = 'cc'
        elif agent == 'codex':
            agent_abbrev = 'cx'
        else:
            agent_abbrev = agent[:2]

        branch = '/'.join(parts[2:])
        session_short = f"{agent_abbrev}/{branch}"
    else:
        session_short = session_name

    # Calculate available space
    # Format: "{sha} {session}: {context}   {code}"
    fixed_space = 7 + 1 + 2 + 3  # sha + space + ": " + "   "
    code_space = len(code)
    available = term_width - fixed_space - code_space

    if context and not no_context:
        min_session = 10
        session_max = min(len(session_short), max(min_session, available // 3))
        context_max = available - session_max - 2

        if context_max > 10:
            session_display = session_short[:session_max]
            if len(session_short) > session_max:
                session_display = session_display[:-2] + '..'

            context_display = context[:context_max]
            if len(context) > context_max:
                context_display = context_display[:-3] + '...'

            return f"{sha} {session_display}: {context_display}   {code}"

    # No context or not enough space
    session_max = max(10, available)
    session_display = session_short[:session_max]
    if len(session_short) > session_max:
        session_display = session_display[:-2] + '..'

    return f"{sha} {session_display}:   {code}"
