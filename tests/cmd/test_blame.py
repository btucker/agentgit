"""Tests for the blame command."""

from __future__ import annotations

from datetime import datetime, timezone

from agentgit.cmd.blame import (
    SessionBlameEntry,
    build_session_index,
    extract_context,
    find_agentgit_paths,
    find_earliest_session,
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

    def test_extracts_context_from_message(self):
        """Test extracting context section from commit message."""

        class MockCommit:
            message = "Subject\n\nContext:\nThis is the context explaining the change.\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        # Result contains first sentence (includes trailing period if no sentence break found)
        assert result == "This is the context explaining the change."

    def test_returns_none_when_no_context(self):
        """Test returns None when no Context section."""

        class MockCommit:
            message = "Subject\n\nSome other content\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result is None

    def test_truncates_long_context(self):
        """Test that long context is truncated."""

        class MockCommit:
            message = "Subject\n\nContext:\n" + ("x" * 100) + "\n\nPrompt-Id: abc123"

        result = extract_context(MockCommit())
        assert result is not None
        assert len(result) <= 80
        assert result.endswith("...")


class TestFindEarliestSession:
    """Tests for find_earliest_session function."""

    def test_finds_earliest_by_timestamp(self):
        """Test that the earliest entry by timestamp is returned."""
        index = {
            "def foo():": [
                SessionBlameEntry("aaa1111", "sessions/cc/first", datetime(2025, 1, 3, tzinfo=timezone.utc), "Third"),
                SessionBlameEntry("bbb2222", "sessions/cc/second", datetime(2025, 1, 1, tzinfo=timezone.utc), "First"),
                SessionBlameEntry("ccc3333", "sessions/cc/third", datetime(2025, 1, 2, tzinfo=timezone.utc), "Second"),
            ]
        }

        result = find_earliest_session("def foo():", index)
        assert result is not None
        assert result.commit_sha == "bbb2222"
        assert result.context == "First"

    def test_returns_none_when_line_not_found(self):
        """Test returns None when line is not in index."""
        index = {
            "def bar():": [
                SessionBlameEntry("aaa1111", "sessions/cc/feature", datetime(2025, 1, 1, tzinfo=timezone.utc), "Ctx"),
            ]
        }

        result = find_earliest_session("def foo():", index)
        assert result is None

    def test_handles_empty_index(self):
        """Test handles empty index gracefully."""
        result = find_earliest_session("anything", {})
        assert result is None


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


class TestFindAgentgitPaths:
    """Tests for find_agentgit_paths function."""

    def test_finds_matching_paths(self):
        """Test finding path variants that end with the code relative path."""

        class MockBlob:
            type = "blob"

            def __init__(self, path):
                self.path = path

        class MockTree:
            def traverse(self):
                return [
                    MockBlob("Documents/project/src/cli.py"),
                    MockBlob("Users/name/Documents/project/src/cli.py"),
                    MockBlob("src/other.py"),
                ]

        class MockCommit:
            tree = MockTree()

        class MockBranch:
            name = "sessions/cc/feature"
            commit = MockCommit()

        paths = find_agentgit_paths("src/cli.py", [MockBranch()])
        assert len(paths) == 2
        assert "Documents/project/src/cli.py" in paths
        assert "Users/name/Documents/project/src/cli.py" in paths
        assert "src/other.py" not in paths

    def test_handles_empty_branches(self):
        """Test handles empty branch list."""
        paths = find_agentgit_paths("src/cli.py", [])
        assert paths == []


class TestBuildSessionIndex:
    """Tests for build_session_index function."""

    def test_returns_empty_dict_when_no_sessions(self):
        """Test returns empty dict when repo has no session branches."""

        class MockRepo:
            heads = []

        result = build_session_index(MockRepo(), "src/cli.py")
        assert result == {}

    def test_returns_empty_dict_when_file_not_in_sessions(self):
        """Test returns empty dict when file doesn't exist in any session."""

        class MockTree:
            def traverse(self):
                return []

            def __getitem__(self, key):
                raise KeyError(key)

        class MockCommit:
            tree = MockTree()

        class MockBranch:
            name = "sessions/cc/feature"
            commit = MockCommit()

        class MockRepo:
            heads = [MockBranch()]

            def blame(self, _ref, _path):
                return []

        result = build_session_index(MockRepo(), "nonexistent.py")
        assert result == {}
