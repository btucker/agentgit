"""Shared utilities for agentgit."""

from __future__ import annotations

import re

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
