"""Tests for the blame command."""

from __future__ import annotations

from datetime import datetime, timezone

from agentgit.cmd.blame import (
    SessionBlameEntry,
    extract_context,
    format_blame_line_with_session,
)


class TestSessionBlameEntry:
    """Tests for SessionBlameEntry NamedTuple."""

    def test_creates_entry(self):
        """Test creating a SessionBlameEntry."""
        entry = SessionBlameEntry(
            commit_sha="abc1234",
            session="sessions/claude_code/feature",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            context="Adding feature X",
        )
        assert entry.commit_sha == "abc1234"
        assert entry.session == "sessions/claude_code/feature"
        assert entry.context == "Adding feature X"


class TestExtractContext:
    """Tests for extract_context function."""

    def test_uses_subject_line_when_informative(self):
        """Test uses subject line when it's not a generic operation description."""

        class MockCommit:
            message = "I'll add a validation helper function\n\nContext:\nSome context here\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result == "I'll add a validation helper function"

    def test_falls_back_to_context_section_for_generic_subject(self):
        """Test falls back to Context section when subject is generic like 'Edit file.py'."""

        class MockCommit:
            message = "Edit auth.py\n\nContext:\nThis is the context explaining the change.\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        # Should fall back to Context section
        assert result == "This is the context explaining the change."

    def test_skips_generic_create_delete_modify_subjects(self):
        """Test skips generic Create/Delete/Modify subjects."""

        class MockCommit:
            message = "Create helper.py\n\nContext:\nAdding a helper.\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result == "Adding a helper."

    def test_skips_initial_commit_subjects(self):
        """Test skips Initial commit/state subjects."""

        class MockCommit:
            message = "Initial state (pre-session)\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result is None

    def test_returns_none_when_no_context_and_generic_subject(self):
        """Test returns None when both subject is generic and no Context section."""

        class MockCommit:
            message = "Edit file.py\n\nSome other content\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result is None

    def test_truncates_long_subject(self):
        """Test that long subject is truncated."""

        class MockCommit:
            message = ("x" * 100) + "\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result is not None
        assert len(result) <= 80
        assert result.endswith("...")

    def test_handles_thinking_section_boundary(self):
        """Test correctly stops at Thinking section."""

        class MockCommit:
            message = "Edit file.py\n\nContext:\nExplanation here.\n\nThinking:\nSome thinking\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result == "Explanation here."


class TestFormatBlameLineWithSession:
    """Tests for format_blame_line_with_session function."""

    def test_abbreviates_claude_code_session(self):
        """Test claude_code session is abbreviated to cc."""
        result = format_blame_line_with_session(
            "abc1234",
            "sessions/claude_code/feature-branch",
            None,
            "code line",
            no_context=True,
        )
        assert "cc/feature-branch" in result
        assert "abc1234" in result

    def test_abbreviates_codex_session(self):
        """Test codex session is abbreviated to cx."""
        result = format_blame_line_with_session(
            "abc1234",
            "sessions/codex/feature-branch",
            None,
            "code line",
            no_context=True,
        )
        assert "cx/feature-branch" in result

    def test_includes_context_when_available(self):
        """Test context is included when available and not disabled."""
        result = format_blame_line_with_session(
            "abc1234",
            "sessions/claude_code/feat",
            "Adding new feature",
            "code",
            no_context=False,
        )
        # Context might be truncated but should be present
        assert "Adding" in result or "cc/feat" in result

    def test_excludes_context_when_disabled(self):
        """Test context is excluded when no_context is True."""
        result = format_blame_line_with_session(
            "abc1234",
            "sessions/claude_code/feature",
            "This context should not appear in full",
            "code line here",
            no_context=True,
        )
        # Should still have session but not the full context
        assert "abc1234" in result
        assert "cc/feature" in result
