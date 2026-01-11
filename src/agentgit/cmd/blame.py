"""Blame command implementation for agentgit.

Maps lines in your code to the agent sessions that wrote them by running
git blame across all session branches and finding the earliest agent commit.
"""

from __future__ import annotations

import re
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

    Runs git blame across all session branches to find which agent commits
    introduced each line of code. For each line, finds the EARLIEST agent
    commit across all sessions.

    Examples:

        agentgit blame auth.py
        agentgit blame src/utils.py -L 10,20
        agentgit blame auth.py --session sessions/claude-code/add-auth

    Args:
        file_path: Path to the file to blame
        lines: Optional line range in format "start,end" or "start,+count"
        no_context: If True, don't show agent context
        session: Optional session branch to blame (uses standard git blame)
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

    # If explicit session specified, use standard git blame on that branch
    if session:
        agentgit_repo = Repo(agentgit_repo_path)
        translated = translate_paths_for_agentgit_repo([file_path], agentgit_repo_path)
        target_file = translated[0]
        click.echo(f"Blaming session: {session}", err=True)
        _blame_single_branch(agentgit_repo, session, target_file, lines, no_context)
        return

    # Check if we're in a code repo
    code_repo_path = find_git_root(Path.cwd())
    if not code_repo_path or code_repo_path == agentgit_repo_path:
        # Fall back to agentgit repo with standard blame
        agentgit_repo = Repo(agentgit_repo_path)
        translated = translate_paths_for_agentgit_repo([file_path], agentgit_repo_path)
        target_file = translated[0]
        click.echo("No code repo found, using agentgit repo", err=True)
        _blame_single_branch(agentgit_repo, "HEAD", target_file, lines, no_context)
        return

    click.echo(f"Using code repo: {code_repo_path}", err=True)

    # Read the file from the code repo
    full_path = code_repo_path / file_path
    if not full_path.exists():
        raise click.ClickException(f"File not found: {full_path}")

    content = full_path.read_text()
    file_lines = content.split('\n')

    # Parse line range if provided
    start_line, end_line = _parse_line_range(lines)

    # Use git blame across all session branches
    agentgit_repo = Repo(agentgit_repo_path)
    line_to_session = blame_across_sessions(
        agentgit_repo,
        file_path,
    )

    click.echo(f"Found {len(line_to_session)} lines with session attribution", err=True)

    # Output blame
    for i, line in enumerate(file_lines):
        if start_line is not None and end_line is not None:
            if i < start_line or i > end_line:
                continue

        stripped = line.rstrip('\n')
        entry = line_to_session.get(stripped)

        if entry:
            blame_line = format_blame_line_with_session(
                entry.commit_sha,
                entry.session,
                entry.context,
                stripped,
                no_context,
            )
        else:
            # No session match - show placeholder
            blame_line = f"??????? {'':30}   {stripped}"

        click.echo(blame_line)


def blame_across_sessions(
    agentgit_repo: "Repo",
    relative_path: str,
) -> dict[str, SessionBlameEntry]:
    """Find earliest agent commit for each line using git blame across sessions.

    Runs git blame on each session branch, then merges results to find
    the earliest agent commit for each line content.

    Args:
        agentgit_repo: The agentgit repository
        relative_path: Relative path of the file in the code repo

    Returns:
        Dict mapping line content -> SessionBlameEntry for earliest commit
    """
    # Find session branches
    session_branches = [
        ref.name for ref in agentgit_repo.refs
        if ref.name.startswith('sessions/')
    ]

    if not session_branches:
        return {}

    click.echo(f"Searching {len(session_branches)} session branches...", err=True)

    # Find file paths that match in each branch using GitPython
    def find_file_in_branch(branch: str) -> tuple[str, str | None]:
        """Find the file path in this branch that ends with relative_path."""
        try:
            # Use GitPython's ls-tree equivalent
            tree_contents = agentgit_repo.git.ls_tree('-r', '--name-only', branch)
            for path in tree_contents.split('\n'):
                if path.endswith(relative_path):
                    return branch, path
            return branch, None
        except Exception:
            return branch, None

    # Find files sequentially (GitPython operations on same repo not thread-safe)
    branch_paths: dict[str, str] = {}
    for branch in session_branches:
        _, path = find_file_in_branch(branch)
        if path:
            branch_paths[branch] = path

    if not branch_paths:
        return {}

    click.echo(f"Found file in {len(branch_paths)} branches", err=True)

    # Run git blame on each branch using GitPython
    def blame_branch(branch_path: tuple[str, str]) -> dict[str, tuple[str, int, str]]:
        """Run git blame on a branch and return line -> (sha, timestamp, branch)."""
        branch, path = branch_path
        results: dict[str, tuple[str, int, str]] = {}

        try:
            # Use GitPython's blame method which returns parsed data
            blame_data = agentgit_repo.blame(branch, path)

            for commit, lines_list in blame_data:
                # Skip initial state commits (bulk imports)
                if commit.summary == 'Initial state (pre-session)':
                    continue

                timestamp = int(commit.committed_date)

                for line_content in lines_list:
                    line_content = line_content.rstrip('\n')
                    if not line_content:
                        continue

                    # Only record if we don't have this line yet,
                    # or if this commit is earlier
                    if line_content not in results or timestamp < results[line_content][1]:
                        results[line_content] = (commit.hexsha[:7], timestamp, branch)

        except Exception:
            pass

        return results

    # Blame all branches sequentially (GitPython's blame is not thread-safe)
    all_results: dict[str, tuple[str, int, str]] = {}
    for branch_path in branch_paths.items():
        branch_results = blame_branch(branch_path)
        # Merge: keep earliest timestamp for each line
        for line_content, (sha, timestamp, branch) in branch_results.items():
            if line_content not in all_results or timestamp < all_results[line_content][1]:
                all_results[line_content] = (sha, timestamp, branch)

    # Convert to SessionBlameEntry with context
    final_results: dict[str, SessionBlameEntry] = {}
    for line_content, (sha, timestamp, branch) in all_results.items():
        # Get context from commit message
        try:
            commit = agentgit_repo.commit(sha)
            context = extract_context(commit)
        except Exception:
            context = None

        final_results[line_content] = SessionBlameEntry(
            commit_sha=sha,
            session=branch,
            timestamp=datetime.fromtimestamp(timestamp),
            context=context,
        )

    return final_results


def _blame_single_branch(
    repo: "Repo",
    ref: str,
    file_path: str,
    lines: str | None,
    no_context: bool,
) -> None:
    """Standard git blame on a single branch."""
    start_line, end_line = _parse_line_range(lines)

    try:
        blame_data = repo.blame(ref, file_path)
    except Exception as e:
        raise click.ClickException(f"Failed to get blame data: {e}")

    line_num = 0
    for commit, lines_list in blame_data:
        for line_content in lines_list:
            if start_line is not None and end_line is not None:
                if line_num < start_line or line_num > end_line:
                    line_num += 1
                    continue

            content = line_content.rstrip('\n')
            context = extract_context(commit)
            short_sha = commit.hexsha[:7]

            if context and not no_context:
                click.echo(f"{short_sha}: {context[:50]}   {content}")
            else:
                author = commit.author.name[:12]
                date = commit.committed_datetime.strftime('%Y-%m-%d')
                click.echo(f"{short_sha} ({author:12} {date:10}) {content}")

            line_num += 1


def _parse_line_range(lines: str | None) -> tuple[int | None, int | None]:
    """Parse line range from -L argument."""
    if not lines:
        return None, None

    parts = lines.split(',')
    if len(parts) != 2:
        return None, None

    start_line = int(parts[0]) - 1  # 0-indexed
    if parts[1].startswith('+'):
        end_line = start_line + int(parts[1])
    else:
        end_line = int(parts[1]) - 1

    return start_line, end_line


def extract_context(commit) -> str | None:
    """Extract context from commit message.

    Prefers the commit subject line (which now contains the contextual summary
    from the previous assistant message). Falls back to the Context section
    for backward compatibility with older commits.

    Args:
        commit: GitPython commit object

    Returns:
        Subject line or first sentence of context, or None if generic
    """
    message = commit.message
    subject = message.split('\n')[0].strip()

    # Skip generic subjects that don't provide useful context
    generic_patterns = [
        r'^(Create|Edit|Delete|Modify) \S+$',  # "Edit file.py"
        r'^Initial (commit|state)',  # Initial commits
        r'^Call \d+ tools?$',  # "Call 3 tools"
        r'^Assistant response$',
        r'^User prompt$',
    ]
    is_generic = any(re.match(pattern, subject, re.IGNORECASE) for pattern in generic_patterns)

    if not is_generic and subject:
        if len(subject) > 80:
            return subject[:77] + "..."
        return subject

    # Fallback: extract from Context section for backward compatibility
    match = re.search(
        r'Context:\s*\n(.+?)(?:\n\n|\nThinking:|\nPrompt-Id:|\nOperation:|\nTimestamp:|\Z)',
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

    Format: {sha} {session}: {context}   {code}

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
