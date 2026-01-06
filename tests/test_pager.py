"""Tests for the Rich markdown pager."""

import io
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


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
