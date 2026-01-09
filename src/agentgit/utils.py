"""Shared utilities for agentgit."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# Pattern to match rm commands
RM_COMMAND_PATTERN = re.compile(r"^\s*rm\s+(?:-[rfivI]+\s+)*(.+)$")


def extract_deleted_paths(command: str) -> list[str]:
    """Extract file paths deleted by an rm command.

    Parses rm commands handling quoted paths and spaces correctly.

    Args:
        command: The shell command string (e.g., "rm -rf /path/to/file")

    Returns:
        List of file paths that would be deleted by the command.
    """
    paths = []
    match = RM_COMMAND_PATTERN.match(command)
    if not match:
        return paths

    args_str = match.group(1).strip()
    current_path = ""
    in_quotes: str | None = None

    for char in args_str:
        if in_quotes:
            if char == in_quotes:
                if current_path:
                    paths.append(current_path)
                    current_path = ""
                in_quotes = None
            else:
                current_path += char
        elif char in ('"', "'"):
            in_quotes = char
        elif char == " ":
            if current_path:
                paths.append(current_path)
                current_path = ""
        else:
            current_path += char

    if current_path:
        paths.append(current_path)

    return paths


def normalize_git_url(url: str) -> str:
    """Normalize a git URL for comparison.

    Handles both SSH and HTTPS formats:
    - git@github.com:user/repo.git -> github.com/user/repo
    - https://github.com/user/repo -> github.com/user/repo
    - https://github.com/user/repo.git -> github.com/user/repo

    Args:
        url: Git URL in any format

    Returns:
        Normalized URL in format: host/user/repo
    """
    # Remove trailing slashes and .git suffix
    url = url.rstrip("/").removesuffix(".git")

    # Try parsing as a URL (handles https:// and http://)
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        # HTTPS/HTTP format: https://github.com/user/repo
        path = parsed.path.lstrip("/")
        return f"{parsed.netloc}/{path}"

    # Handle SSH format: git@github.com:user/repo
    # SSH URLs have format [user@]host:path
    if ":" in url and not url.startswith("/"):
        # Split on last @ to get user@host:path -> host:path
        after_at = url.split("@")[-1] if "@" in url else url
        # Split on first : to get host:path
        if ":" in after_at and "/" not in after_at.split(":")[0]:
            host, path = after_at.split(":", 1)
            return f"{host}/{path}"

    # Return as-is if no pattern matches
    return url


def get_git_remotes(project_path: Path) -> list[str]:
    """Get all git remote URLs for a project.

    Args:
        project_path: Path to the project directory

    Returns:
        List of git remote URLs (empty if not a git repo or on error)
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(project_path), "remote", "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        # Parse output: "origin  git@github.com:user/repo.git (fetch)"
        remotes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                url = parts[1]
                if url not in remotes:
                    remotes.append(url)

        return remotes
    except (subprocess.SubprocessError, OSError):
        return []


# Tool names that modify files
FILE_MODIFYING_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})


def has_file_operations(file_path: Path) -> bool:
    """Quickly check if a JSONL transcript contains file-modifying operations.

    This performs a fast scan of the file looking for tool_use entries with
    Write, Edit, or NotebookEdit tools. Stops as soon as one is found.

    Args:
        file_path: Path to the JSONL transcript file.

    Returns:
        True if the file contains file-modifying operations, False otherwise.
    """
    import json

    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Quick string check before full JSON parse
                if '"tool_use"' not in line:
                    continue

                try:
                    data = json.loads(line)
                    msg = data.get("message", {})
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if (
                                item.get("type") == "tool_use"
                                and item.get("name") in FILE_MODIFYING_TOOLS
                            ):
                                return True
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        pass

    return False
