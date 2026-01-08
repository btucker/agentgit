"""Rich markdown pager for git output.

This module provides a pager that renders markdown using Rich
and pipes the output to `less -RFX` for terminal viewing.
"""

import json
import re
import subprocess
import sys
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes and other escape sequences from text.

    Args:
        text: Text that may contain ANSI escape sequences.

    Returns:
        Text with ANSI codes removed.
    """
    # Remove OSC 8 hyperlinks: \x1b]8;params\x1b\ ... \x1b]8;;\x1b\
    text = re.sub(r'\x1b\]8;[^\x1b]*\x1b\\', '', text)
    # Remove color codes like \x1b[31m, \x1b[0m, etc.
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    return text


def format_todo_list(todos: list, tool_id: str) -> str:
    """Format a TodoWrite tool call as a nice list.

    Args:
        todos: List of todo items with status and content.
        tool_id: Tool ID for reference.

    Returns:
        Formatted todo list as markdown.
    """
    if not todos:
        return ""

    lines = ["\n### ☰ Task List\n"]

    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        # Choose icon based on status (markdown will render these)
        if status == "completed":
            icon = "✓"  # Checkmark for completed
        elif status == "in_progress":
            icon = "→"  # Arrow for in progress
        else:  # pending
            icon = "○"  # Circle for pending

        lines.append(f"- {icon} {content}")

    return "\n".join(lines)


def format_bash_tool(command: str, description: str, tool_id: str) -> str:
    """Format a Bash tool call nicely.

    Args:
        command: The bash command.
        description: Optional description of what the command does.
        tool_id: Tool ID for reference.

    Returns:
        Formatted bash tool string.
    """
    lines = ["\n### $ Bash\n"]

    if description:
        lines.append(f"*{description}*\n")

    # Show command in code block
    lines.append(f"```bash\n{command}\n```")

    return "\n".join(lines)


def enhance_tool_calls(text: str) -> str:
    """Enhance tool call formatting in markdown text.

    Detects tool calls in JSON format and replaces them with
    nicely formatted output for TodoWrite and Bash tools.

    Args:
        text: Markdown text that may contain tool calls.

    Returns:
        Text with enhanced tool call formatting.
    """
    # Pattern to match tool call sections with optional leading whitespace
    # Format: [spaces]- **ToolName** (`tool_id`)\n[spaces]```json\n{...}\n[spaces]```
    pattern = r'\s*- \*\*(\w+)\*\* \(`([^`]+)`\)\s*\n\s*```json\s*\n(.*?)\n\s*```'

    def replace_tool(match):
        tool_name = match.group(1)
        tool_id = match.group(2)
        json_str = match.group(3)

        try:
            tool_input = json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, return original
            return match.group(0)

        # Format based on tool type
        if tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])
            return format_todo_list(todos, tool_id)
        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            description = tool_input.get("description", "")
            return format_bash_tool(command, description, tool_id)
        else:
            # Keep other tools as-is
            return match.group(0)

    return re.sub(pattern, replace_tool, text, flags=re.DOTALL)


def preprocess_git_output(text: str) -> str:
    """Preprocess git log/show output to remove commit message indentation.

    Git indents commit message bodies with 4 spaces, which Markdown interprets
    as code blocks. This function removes that indentation so Rich can properly
    render markdown formatting in commit messages.

    When git log --graph is used, preserves the ASCII art graph characters.

    Args:
        text: Raw git output text.

    Returns:
        Preprocessed text with commit message indentation removed.
    """
    lines = text.split('\n')
    result = []

    for line in lines:
        # Check if this line contains git graph characters
        # Graph lines have characters like |, *, \, / mixed with spaces
        stripped = line.lstrip()
        has_graph = any(c in line[:10] for c in ['|', '*', '\\', '/'])

        # Preserve lines that:
        # 1. Start with commit metadata (commit, Author:, Date:, Merge:)
        # 2. Contain git graph characters
        # 3. Are empty
        if (line.startswith(('commit ', 'Author:', 'Commit:', 'Date:', 'Merge:')) or
            has_graph or
            not line.strip()):
            result.append(line)
        # Lines with 4-space indent that aren't graph lines are commit messages
        elif line.startswith('    ') and not line.startswith('        '):
            # Remove the 4-space indent for markdown rendering
            result.append(line[4:])  # Strip first 4 spaces
        # Lines with more than 4 spaces keep relative indentation
        elif line.startswith('        '):
            # Keep relative indentation (e.g., nested lists, code blocks)
            result.append(line[4:])  # Remove base 4-space indent
        else:
            # Non-indented lines or other content
            result.append(line)

    return '\n'.join(result)


def parse_graph_line(line: str) -> tuple[str, str]:
    """Parse a git graph line into graph prefix and content.

    Args:
        line: A line from git log --graph output.

    Returns:
        Tuple of (graph_prefix, content) where graph_prefix contains the ASCII art
        and content contains the actual commit info/message.
    """
    # Find where the graph ends and content begins
    # Graph characters are: space, |, *, \, /, and decorations like (
    graph_chars = set(' |*\\/()_-')

    # Scan from left to find where non-graph content starts
    content_start = 0
    for i, char in enumerate(line):
        # If we hit a non-graph character that's not whitespace after a graph char
        if char not in graph_chars:
            content_start = i
            break
        # Special case: if we see 'commit' or other metadata, that's content
        if line[i:].startswith(('commit ', 'Author:', 'Date:', 'Merge:')):
            content_start = i
            break

    graph_prefix = line[:content_start] if content_start > 0 else line
    content = line[content_start:] if content_start > 0 else ''

    return graph_prefix, content


def render_graph_with_markdown(text: str) -> str:
    """Render git graph output with markdown in commit messages.

    Preserves the ASCII art graph while rendering markdown in commit messages.

    Args:
        text: Git log --graph output.

    Returns:
        Rendered output with graph and markdown combined.
    """
    lines = text.split('\n')
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120, _environ={"NO_COLOR": ""})

    # Process lines in groups (commit blocks)
    current_graph_lines = []
    current_content_lines = []
    in_commit_message = False

    for line in lines:
        graph_prefix, content = parse_graph_line(line)

        # Detect if we're in a commit message (indented content after metadata)
        if content.strip() and not content.startswith(('commit ', 'Author:', 'Commit:', 'Date:', 'Merge:')):
            in_commit_message = True
        elif content.startswith(('commit ', 'Author:', 'Commit:', 'Date:', 'Merge:')):
            in_commit_message = False

        # For each line, preserve graph and optionally render content as markdown
        if in_commit_message and content.strip():
            # This is commit message content - render as markdown
            # Create a mini markdown string just for this content
            md = Markdown(content)
            temp_buffer = StringIO()
            temp_console = Console(file=temp_buffer, force_terminal=True, width=120)
            temp_console.print(md, end='')
            rendered_content = temp_buffer.getvalue()
            # Combine graph prefix with rendered markdown
            console.print(f"{graph_prefix}{rendered_content}", end='')
        else:
            # Metadata or graph-only line - print as-is
            console.print(line, end='')

        if line != lines[-1]:  # Don't add newline after last line
            console.print()

    return buffer.getvalue()


def render_markdown(text: str) -> str:
    """Render markdown text using Rich.

    Args:
        text: The markdown text to render.

    Returns:
        The rendered text with ANSI formatting codes (hyperlinks stripped).
    """
    if not text.strip():
        return ""

    # Check if this is git graph output (has graph characters in first 100 chars of multiple lines)
    lines = text.split('\n')[:min(20, len(text.split('\n')))]
    graph_line_count = sum(1 for line in lines if any(c in line[:15] for c in ['|', '*', '\\', '/']))

    # If more than 2 lines have graph characters, this is likely git log --graph output
    # For graph output, just preprocess to remove indentation but don't render markdown
    # because the graph structure makes it too complex to render markdown reliably
    if graph_line_count > 2:
        return preprocess_git_output(text)

    # Preprocess git output to remove commit message indentation
    text = preprocess_git_output(text)

    # Enhance tool call formatting (TodoWrite, Bash, etc.)
    text = enhance_tool_calls(text)

    # Create a string buffer to capture output
    buffer = StringIO()

    # Create console that writes to the buffer with force_terminal for ANSI codes
    console = Console(file=buffer, force_terminal=True, width=120)

    # Render the markdown
    md = Markdown(text)
    console.print(md)

    rendered = buffer.getvalue()

    # Strip hyperlinks from Rich's output (they don't work well in less)
    rendered = re.sub(r'\x1b\]8;[^\x1b]*\x1b\\', '', rendered)

    return rendered


def main() -> int:
    """Main entry point for the pager.

    Reads from stdin, renders markdown with Rich, and pipes to less.

    Returns:
        Exit code from less, or 0 on success.
    """
    try:
        # Read all input from stdin
        text = sys.stdin.read()

        # Strip ANSI color codes from git output before markdown rendering
        # This prevents git's colors from interfering with Rich's markdown formatting
        text = strip_ansi_codes(text)

        # Render markdown
        rendered = render_markdown(text)

        # Pipe to less with:
        # -R: interpret ANSI color codes
        # -F: quit if output fits on one screen
        # -X: don't clear screen on exit
        result = subprocess.run(
            ["less", "-RFX"],
            input=rendered,
            text=True,
        )
        return result.returncode

    except BrokenPipeError:
        # Handle case where less exits early
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
