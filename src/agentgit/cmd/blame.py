"""Blame command implementation for agentgit.

Maps lines in your code to the agent sessions that wrote them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from git import Repo


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

    if use_session_mapping:
        agentgit_repo = Repo(agentgit_repo_path)

        # Check if grep index exists, build if needed
        try:
            agentgit_repo.git.rev_parse('refs/heads/agentgit-index')
            click.echo("Using line index...", err=True)
        except Exception:
            # Index doesn't exist, build it
            click.echo("Building line index (first time only)...", err=True)
            build_line_grep_index(agentgit_repo, agentgit_repo_path)

    def find_session_for_line(
        line_content: str,
        file_path: str,
        prev_line: str = "",
        next_line: str = ""
    ) -> tuple[str | None, str | None, str | None]:
        """Find which session branch wrote this line.

        Returns: (session_name, context, commit_sha)
        """
        if not use_session_mapping or not agentgit_repo:
            return None, None, None

        # Hash the line with sliding window context
        line_hash = hash_line(line_content, prev_line, next_line)

        # Look up using git grep
        result = lookup_line_with_grep(agentgit_repo, file_path, line_hash)

        if result:
            return result['session'], result['context'], result['commit']

        return None, None, None

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

    # First, collect all lines with their commits (for sliding window context)
    all_lines = []
    for commit, lines_list in blame_data:
        for line_content in lines_list:
            all_lines.append((commit, line_content))

    # Process and display blame output with sliding window
    line_num = 0
    for i, (commit, line_content) in enumerate(all_lines):
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

        # Get prev/next lines for sliding window
        prev_line = all_lines[i-1][1] if i > 0 else ""
        next_line = all_lines[i+1][1] if i < len(all_lines) - 1 else ""

        # Find session if in mapping mode (using line-level matching)
        session_name, session_context, agentgit_commit_sha = find_session_for_line(
            line_content, target_file, prev_line, next_line
        )

        if session_name:
            # Format session name compactly: sessions/claude_code/foo -> cc/foo
            # Use agentgit commit SHA instead of code repo commit SHA
            # Note: agentgit_commit_sha is already the short SHA (7 chars)
            agentgit_short_sha = agentgit_commit_sha if agentgit_commit_sha else short_sha
            blame_line = format_blame_line_with_session(
                agentgit_short_sha, session_name, session_context, content, no_context
            )
        else:
            # Standard blame format
            blame_line = f"{short_sha} ({author:12} {date:10}) {content}"

        click.echo(blame_line)

        line_num += 1


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

    # Get terminal width, default to 80 if not available
    term_width = shutil.get_terminal_size((80, 24)).columns

    # Abbreviate session name: sessions/claude_code/foo -> cc/foo
    parts = session_name.split('/')
    if len(parts) >= 3 and parts[0] == 'sessions':
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


def hash_line(line: str, prev_line: str = "", next_line: str = "") -> str:
    """Create consistent hash for a line using a sliding 3-line window.

    Uses prev + current + next lines to provide positional context.
    This disambiguates duplicate lines (like 'pass' or '}') based on
    their surrounding code.

    - Strip trailing newlines for consistency
    - Keep leading/trailing whitespace (significant in code)
    - Use UTF-8 encoding
    - Return first 16 hex chars (64 bits - enough for uniqueness)

    Args:
        line: Current line of code to hash
        prev_line: Previous line for context (empty string if first line)
        next_line: Next line for context (empty string if last line)

    Returns:
        16-character hex hash of the 3-line window
    """
    import hashlib

    # Normalize each line (strip trailing newline)
    prev_norm = prev_line.rstrip('\n')
    curr_norm = line.rstrip('\n')
    next_norm = next_line.rstrip('\n')

    # Hash the 3-line window
    window = f"{prev_norm}\n{curr_norm}\n{next_norm}"
    return hashlib.sha256(window.encode('utf-8')).hexdigest()[:16]


def normalize_session_path(session_path: str) -> str:
    """Normalize session file path to code repo path.

    Session paths have full directories like:
        Documents/projects/agentgit/src/file.py
        Users/btucker/Documents/projects/agentgit/src/file.py

    Code repo paths are relative:
        src/file.py

    Args:
        session_path: Path from session commit tree

    Returns:
        Normalized path matching code repo structure
    """
    # Common prefixes to strip
    prefixes = [
        'Documents/projects/agentgit/',
        'Users/btucker/Documents/projects/agentgit/',
        # Add more patterns as needed
    ]

    for prefix in prefixes:
        if session_path.startswith(prefix):
            return session_path[len(prefix):]

    # If no prefix matches, return as-is
    return session_path


def build_line_grep_index(agentgit_repo: "Repo", agentgit_repo_path: Path) -> None:
    """Build searchable line index using git grep.

    Creates a branch 'agentgit-index' with text files containing line mappings.
    Each file maps line hashes to sessions for efficient git grep lookups.

    Format: .agentgit/lines/<normalized_path>
    Content: line_hash|session_name|commit_sha|context

    Args:
        agentgit_repo: The agentgit repository
        agentgit_repo_path: Path to the agentgit repository
    """
    import os
    import shutil
    import tempfile
    from collections import defaultdict

    # Group line mappings by source file
    by_file = defaultdict(list)

    # Traverse all session branches
    for branch in agentgit_repo.heads:
        if not branch.name.startswith('sessions/'):
            continue

        # Checkout the branch to run git blame
        try:
            current_head = agentgit_repo.head.commit
        except Exception:
            current_head = None

        try:
            # Checkout the session branch
            agentgit_repo.git.checkout(branch.name)

            # Get all files in this branch
            for item in branch.commit.tree.traverse():
                if item.type != 'blob':
                    continue

                # Normalize the file path
                normalized_path = normalize_session_path(item.path)
                file_path = item.path

                # Run git blame on this file to get commit per line
                try:
                    blame_data = agentgit_repo.blame(branch.name, file_path)
                except Exception:
                    continue

                # Process each blamed line
                for commit, lines in blame_data:
                    # Only index Write/Edit operations (actual file modifications)
                    # Skip merge commits, initial state, and other non-file operations
                    # Check for Tool-Id trailer (indicates this is an operation commit)
                    if "Tool-Id:" not in commit.message:
                        continue

                    # Check if this is a Write or Edit operation
                    # Look for tool name in the commit message or Operation trailer
                    tool_match = re.search(r'\bOperation:\s*(Write|Edit)\b', commit.message)
                    if not tool_match:
                        # Also check in the first line of the commit message
                        first_line = commit.message.split('\n')[0]
                        if not any(op in first_line for op in ['Write', 'Edit', 'write', 'edit']):
                            continue

                    # Extract context from this commit's message
                    context_match = re.search(
                        r'Context:\s*(.+?)(?:\n\n|\nPrompt-Id:|\nOperation:|\nTimestamp:|\Z)',
                        commit.message,
                        re.DOTALL
                    )
                    context = ""
                    if context_match:
                        ctx = context_match.group(1).strip()
                        first_sentence = re.split(r'[.!?]\s', ctx)[0]
                        if len(first_sentence) > 80:
                            first_sentence = first_sentence[:77] + "..."
                        context = first_sentence

                    # Convert lines to list for indexing
                    lines_list = list(lines)

                    # Index each line with sliding window
                    for i, line in enumerate(lines_list):
                        prev_line = lines_list[i-1] if i > 0 else ""
                        next_line = lines_list[i+1] if i < len(lines_list) - 1 else ""

                        line_hash = hash_line(line, prev_line, next_line)

                        # Escape pipes in context
                        escaped_context = context.replace('|', '\\|')

                        # Format: line_hash|session_name|commit_sha|context
                        index_line = f"{line_hash}|{branch.name}|{commit.hexsha[:7]}|{escaped_context}\n"

                        by_file[normalized_path].append(index_line)

        except Exception:
            continue
        finally:
            # Return to original HEAD
            if current_head:
                try:
                    agentgit_repo.git.checkout(current_head.hexsha)
                except Exception:
                    pass

    # Now create the index branch with these files
    if not by_file:
        # No lines to index
        return

    # Use git commands directly for simplicity
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the file structure
        lines_dir = os.path.join(tmpdir, '.agentgit', 'lines')
        os.makedirs(lines_dir, exist_ok=True)

        for file_path, lines in by_file.items():
            index_filename = file_path.replace('/', '_')
            index_file_path = os.path.join(lines_dir, index_filename)

            content = ''.join(sorted(set(lines)))
            with open(index_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # Use git commands to create the index branch
        # Save current HEAD
        try:
            current_head = agentgit_repo.head.commit
        except Exception:
            current_head = None

        # Create orphan branch for the index
        agentgit_repo.git.checkout('--orphan', 'agentgit-index')

        # Remove all files
        agentgit_repo.git.rm('-rf', '.', force=True)

        # Copy our index files
        dest_lines_dir = os.path.join(agentgit_repo.working_dir, '.agentgit', 'lines')
        os.makedirs(dest_lines_dir, exist_ok=True)

        for file_path, lines in by_file.items():
            index_filename = file_path.replace('/', '_')
            src = os.path.join(lines_dir, index_filename)
            dst = os.path.join(dest_lines_dir, index_filename)
            shutil.copy(src, dst)

        # Add and commit
        agentgit_repo.git.add('.agentgit')
        agentgit_repo.git.commit('-m', 'Update line index', '--allow-empty')

        # Return to original HEAD
        if current_head:
            agentgit_repo.git.checkout(current_head.hexsha)


def lookup_line_with_grep(agentgit_repo: "Repo", file_path: str, line_hash: str) -> dict | None:
    """Use git grep to find which session wrote this line.

    Args:
        agentgit_repo: The agentgit repository
        file_path: Normalized file path (e.g., 'src/agentgit/cli.py')
        line_hash: Hash of the line to look up

    Returns:
        Dict with session, commit, and context, or None if not found
    """
    # Convert file path to index file path
    # src/agentgit/cli.py -> .agentgit/lines/src_agentgit_cli.py
    index_filename = file_path.replace('/', '_')
    index_file = f".agentgit/lines/{index_filename}"

    try:
        # Git grep searches the index branch (not checked out!)
        # Pattern: lines starting with our line hash
        # Use -F for fixed-string (literal) matching
        result = agentgit_repo.git.grep(
            '-F',  # Fixed-string (literal) matching
            f"{line_hash}|",  # Search for: hash|
            'refs/heads/agentgit-index',  # Search this ref
            '--',  # Separator
            index_file  # In this file
        )

        # Parse result
        # Format: "refs/heads/agentgit-index:.agentgit/lines/src_agentgit_cli.py:abc123|sessions/cc/add-auth|xyz|Adding JWT..."
        if result:
            # Git grep may return multiple lines if there are multiple matches
            # Take the first match
            first_line = result.split('\n')[0] if '\n' in result else result

            # Extract the matched line (after second colon)
            matched_line = first_line.split(':', 2)[2]

            # Parse the pipe-delimited format
            parts = matched_line.split('|', 3)

            if len(parts) >= 4:
                return {
                    'session': parts[1],
                    'commit': parts[2],
                    'context': parts[3].replace('\\|', '|').strip()
                }

    except Exception:
        # File might not be in index, or line not found
        pass

    return None
