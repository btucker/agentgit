"""Tests for the Rich markdown pager."""

import io
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestStripAnsiCodes:
    """Tests for ANSI code stripping."""

    def test_strips_basic_color_codes(self):
        """Test that basic ANSI color codes are removed."""
        from agentgit.pager import strip_ansi_codes

        text = "\x1b[31mRed text\x1b[0m normal text"
        result = strip_ansi_codes(text)
        assert result == "Red text normal text"

    def test_strips_multiple_codes(self):
        """Test that multiple ANSI codes are removed."""
        from agentgit.pager import strip_ansi_codes

        text = "\x1b[1m\x1b[31m||\x1b[0m\x1b[m text"
        result = strip_ansi_codes(text)
        assert result == "|| text"

    def test_preserves_plain_text(self):
        """Test that text without ANSI codes is unchanged."""
        from agentgit.pager import strip_ansi_codes

        text = "Just plain text with no codes"
        result = strip_ansi_codes(text)
        assert result == text

    def test_strips_git_graph_colors(self):
        """Test stripping ANSI codes from git graph output."""
        from agentgit.pager import strip_ansi_codes

        text = "\x1b[1m\x1b[31m|\x1b[m commit abc123"
        result = strip_ansi_codes(text)
        assert result == "| commit abc123"


class TestPreprocessGitOutput:
    """Tests for git output preprocessing."""

    def test_removes_commit_message_indentation(self):
        """Test that 4-space indentation is removed from commit messages."""
        from agentgit.pager import preprocess_git_output

        git_output = """commit abc123
Author: Test <test@example.com>
Date:   Mon Jan 1 00:00:00 2025 +0000

    This is the subject

    This is the body with **markdown**.

    - Item 1
    - Item 2
"""
        result = preprocess_git_output(git_output)

        # Indentation should be removed
        assert "\nThis is the subject\n" in result
        assert "\nThis is the body with **markdown**.\n" in result
        assert "\n- Item 1\n" in result
        assert "    This is the subject" not in result

    def test_preserves_nested_indentation(self):
        """Test that nested indentation (8+ spaces) keeps relative spacing."""
        from agentgit.pager import preprocess_git_output

        git_output = """commit abc123

    Subject

    Some text:
        Nested item
            More nested
"""
        result = preprocess_git_output(git_output)

        # Base 4 spaces removed, but relative indentation preserved
        assert "\nSome text:\n" in result
        assert "\n    Nested item\n" in result  # 8 spaces -> 4 spaces
        assert "\n        More nested\n" in result  # 12 spaces -> 8 spaces

    def test_preserves_non_commit_content(self):
        """Test that non-commit content is left unchanged."""
        from agentgit.pager import preprocess_git_output

        text = """Some random output
No commit info here
Just plain text"""

        result = preprocess_git_output(text)
        assert result == text

    def test_handles_multiple_commits(self):
        """Test preprocessing of multiple commits in git log output."""
        from agentgit.pager import preprocess_git_output

        git_output = """commit abc123

    First commit

commit def456

    Second commit
"""
        result = preprocess_git_output(git_output)

        assert "\nFirst commit\n" in result
        assert "\nSecond commit\n" in result
        assert "    First commit" not in result
        assert "    Second commit" not in result

    def test_preserves_git_graph_formatting(self):
        """Test that git graph ASCII art is preserved."""
        from agentgit.pager import preprocess_git_output

        git_output = """* commit abc123
| Author: Test <test@example.com>
| Date:   Mon Jan 1 00:00:00 2025 +0000
|
|     Add feature X
|
|     This adds **markdown**.
|
* commit def456
  Author: Test <test@example.com>

      Another commit
"""
        result = preprocess_git_output(git_output)

        # Graph characters should be preserved
        assert "* commit abc123" in result
        assert "| Author:" in result
        assert "| Date:" in result
        # Lines with graph characters are preserved as-is (not de-indented)
        assert "|     Add feature X" in result
        assert "|     This adds **markdown**." in result


class TestRenderMarkdown:
    """Tests for markdown rendering function."""

    def test_renders_markdown_headers(self):
        """Test that markdown headers are rendered."""
        from agentgit.pager import render_markdown

        result = render_markdown("# Header\n\nSome text")
        # Rich adds formatting - check that output contains the header text
        assert "Header" in result
        assert "Some text" in result

    def test_renders_code_blocks(self):
        """Test that code blocks are rendered."""
        from agentgit.pager import render_markdown

        result = render_markdown("```python\nprint('hello')\n```")
        assert "print" in result
        assert "hello" in result

    def test_renders_inline_code(self):
        """Test that inline code is rendered."""
        from agentgit.pager import render_markdown

        result = render_markdown("Use `git status` to check")
        assert "git status" in result

    def test_preserves_plain_text(self):
        """Test that plain text without markdown is preserved."""
        from agentgit.pager import render_markdown

        result = render_markdown("Just plain text here")
        assert "Just plain text here" in result

    def test_empty_input(self):
        """Test handling of empty input."""
        from agentgit.pager import render_markdown

        result = render_markdown("")
        # Should return empty or whitespace
        assert result.strip() == ""


class TestPagerMain:
    """Tests for the pager main function."""

    def test_reads_from_stdin_and_pipes_to_less(self, monkeypatch):
        """Test that main reads stdin and pipes to less."""
        from agentgit.pager import main

        # Mock stdin
        mock_stdin = io.StringIO("# Test Header\n\nSome content")
        monkeypatch.setattr("sys.stdin", mock_stdin)

        # Mock subprocess.run
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = main()

        # Should have called subprocess.run with less
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] == ["less", "-RFX"]
        # Check that input was provided
        assert "input" in args[1]
        assert "Test Header" in args[1]["input"]

    def test_returns_less_exit_code(self, monkeypatch):
        """Test that main returns the exit code from less."""
        from agentgit.pager import main

        mock_stdin = io.StringIO("test")
        monkeypatch.setattr("sys.stdin", mock_stdin)

        mock_run = MagicMock(return_value=MagicMock(returncode=42))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = main()
        assert result == 42

    def test_handles_broken_pipe(self, monkeypatch):
        """Test graceful handling when less exits early (broken pipe)."""
        from agentgit.pager import main

        mock_stdin = io.StringIO("test content")
        monkeypatch.setattr("sys.stdin", mock_stdin)

        # Simulate broken pipe
        mock_run = MagicMock(side_effect=BrokenPipeError)
        monkeypatch.setattr("subprocess.run", mock_run)

        result = main()
        assert result == 0  # Should exit cleanly

    def test_handles_keyboard_interrupt(self, monkeypatch):
        """Test graceful handling of Ctrl+C (KeyboardInterrupt)."""
        from agentgit.pager import main

        mock_stdin = io.StringIO("test content")
        monkeypatch.setattr("sys.stdin", mock_stdin)

        # Simulate keyboard interrupt
        mock_run = MagicMock(side_effect=KeyboardInterrupt)
        monkeypatch.setattr("subprocess.run", mock_run)

        result = main()
        assert result == 130  # Standard exit code for SIGINT


class TestGraphDetection:
    """Tests for detecting git graph output."""

    def test_detects_graph_output(self):
        """Test that git graph output is detected and preserved (markdown not rendered)."""
        from agentgit.pager import render_markdown

        # Git log --graph output with multiple graph characters
        graph_output = """* commit abc123
| Author: Test <test@example.com>
| Date:   Mon Jan 1 00:00:00 2025 +0000
|
|     Add feature with **bold**
|
* commit def456
  Author: Test <test@example.com>
"""
        result = render_markdown(graph_output)
        # Graph structure should be preserved exactly as-is
        assert "* commit abc123" in result
        assert "| Author:" in result
        # For graph output, content is preserved without markdown rendering
        # Graph lines keep their indentation to preserve the visual structure
        assert "|     Add feature with **bold**" in result
        # Markdown is not processed - bold syntax remains literal
        assert "**bold**" in result

    def test_renders_non_graph_output(self):
        """Test that non-graph output is rendered as markdown."""
        from agentgit.pager import render_markdown

        # Git show output without graph characters
        show_output = """commit abc123
Author: Test <test@example.com>
Date:   Mon Jan 1 00:00:00 2025 +0000

    Add feature

    This adds **bold text**.
"""
        result = render_markdown(show_output)
        # Should be rendered as markdown (bold text gets ANSI codes)
        assert result != show_output
        assert "**bold text**" not in result  # Should be rendered, not literal


class TestParseGraphLine:
    """Tests for parsing graph lines."""

    def test_parses_commit_line(self):
        """Test parsing a commit line with graph."""
        from agentgit.pager import parse_graph_line

        line = "* commit abc123def456"
        graph, content = parse_graph_line(line)

        assert graph == "* "
        assert content == "commit abc123def456"

    def test_parses_graph_only_line(self):
        """Test parsing a line with only graph characters."""
        from agentgit.pager import parse_graph_line

        line = "|  "
        graph, content = parse_graph_line(line)

        assert graph == line
        assert content == ""

    def test_parses_message_line(self):
        """Test parsing a message line with graph prefix."""
        from agentgit.pager import parse_graph_line

        line = "|     Add new feature"
        graph, content = parse_graph_line(line)

        assert "Add new feature" in content
        assert "|" in graph


class TestPagerIntegration:
    """Integration tests for pager with git output."""

    def test_renders_git_log_style_output(self):
        """Test rendering of git log style output with trailers."""
        from agentgit.pager import render_markdown

        git_output = """commit abc123
Author: Test User <test@example.com>
Date:   Mon Jan 1 00:00:00 2025 +0000

    Add new feature

    This commit adds a **new feature** with:
    - Better error handling
    - Improved performance

    Tool-Id: tool_123
    Prompt-Id: prompt_456
"""
        result = render_markdown(git_output)
        # Key content should be preserved
        assert "abc123" in result
        assert "new feature" in result
        assert "Tool-Id" in result

    def test_renders_markdown_formatting_in_commits(self):
        """Test that markdown in commit messages is properly rendered."""
        from agentgit.pager import render_markdown

        git_output = """commit abc123
Author: Test <test@example.com>

    Implement feature X

    This adds **bold text** and *italic text*.

    - List item 1
    - List item 2

    ```python
    print('code block')
    ```
"""
        result = render_markdown(git_output)

        # Markdown should be rendered, not displayed as literal text
        # Bold text should have ANSI codes (not literal **)
        assert "**bold text**" not in result  # Should be rendered, not literal
        # List items should have bullet points
        assert "•" in result or "●" in result  # Rich adds bullet points
