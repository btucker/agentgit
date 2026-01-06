"""Rich markdown pager for git output.

This module provides a pager that renders markdown using Rich
and pipes the output to `less -RFX` for terminal viewing.
"""

import subprocess
import sys
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown


def render_markdown(text: str) -> str:
    """Render markdown text using Rich.

    Args:
        text: The markdown text to render.

    Returns:
        The rendered text with ANSI formatting codes.
    """
    if not text.strip():
        return ""

    # Create a string buffer to capture output
    buffer = StringIO()

    # Create console that writes to the buffer with force_terminal for ANSI codes
    console = Console(file=buffer, force_terminal=True, width=120)

    # Render the markdown
    md = Markdown(text)
    console.print(md)

    return buffer.getvalue()


def main() -> int:
    """Main entry point for the pager.

    Reads from stdin, renders markdown with Rich, and pipes to less.

    Returns:
        Exit code from less, or 0 on success.
    """
    try:
        # Read all input from stdin
        text = sys.stdin.read()

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
