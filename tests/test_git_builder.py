"""Tests for agentgit.git_builder module."""

import tempfile
from pathlib import Path

import pytest
from git import Repo

from agentgit.core import (
    AssistantContext,
    AssistantTurn,
    FileOperation,
    OperationType,
    Prompt,
    PromptResponse,
)
from agentgit.git_builder import (
    CommitMetadata,
    GitRepoBuilder,
    format_commit_message,
    get_processed_operations,
    normalize_file_paths,
    parse_commit_trailers,
)


class TestFormatCommitMessage:
    """Tests for format_commit_message function."""

    def test_write_operation_subject(self):
        """Should format write operation with Create verb."""
        op = FileOperation(
            file_path="/path/to/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_commit_message(op)
        assert message.startswith("Create file.py")

    def test_edit_operation_subject(self):
        """Should format edit operation with Edit verb."""
        op = FileOperation(
            file_path="/path/to/file.py",
            operation_type=OperationType.EDIT,
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_commit_message(op)
        assert message.startswith("Edit file.py")

    def test_delete_operation_subject(self):
        """Should format delete operation with Delete verb."""
        op = FileOperation(
            file_path="/path/to/file.py",
            operation_type=OperationType.DELETE,
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_commit_message(op)
        assert message.startswith("Delete file.py")

    def test_includes_prompt_text_truncated(self):
        """Should include prompt text truncated to 200 chars."""
        long_prompt = "x" * 1000
        prompt = Prompt(text=long_prompt, timestamp="2025-01-01T00:00:00Z")
        op = FileOperation(
            file_path="/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            prompt=prompt,
        )
        message = format_commit_message(op)
        assert "User Prompt:" in message
        # Should be truncated with ellipsis
        assert ("x" * 197 + "...") in message
        # Should NOT contain full prompt
        assert long_prompt not in message

    def test_uses_contextual_summary_as_subject(self):
        """Should use contextual summary from previous assistant message as subject."""
        op = FileOperation(
            file_path="/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            assistant_context=AssistantContext(
                previous_message_text="I'll create a utility function to handle authentication",
                thinking="Let me think about the approach...",
            ),
        )
        message = format_commit_message(op)
        # Subject should be the first line of previous_message_text
        assert message.startswith("I'll create a utility function to handle authentication")

    def test_includes_context_section_with_previous_message(self):
        """Should include Context section with previous assistant message."""
        op = FileOperation(
            file_path="/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            assistant_context=AssistantContext(
                previous_message_text="This is the explanation of what I'm about to do.",
            ),
        )
        message = format_commit_message(op)
        assert "Context:\nThis is the explanation of what I'm about to do." in message

    def test_includes_thinking_section(self):
        """Should include Thinking section when thinking is present."""
        op = FileOperation(
            file_path="/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            assistant_context=AssistantContext(
                thinking="I need to consider error handling here...",
            ),
        )
        message = format_commit_message(op)
        assert "Thinking:\nI need to consider error handling here..." in message

    def test_falls_back_to_operation_subject_when_no_context(self):
        """Should fall back to operation-based subject when no contextual summary."""
        op = FileOperation(
            file_path="/path/to/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_commit_message(op)
        assert message.startswith("Create file.py")

    def test_includes_trailers(self):
        """Should include machine-parseable trailers."""
        prompt = Prompt(text="test", timestamp="2025-01-01T00:00:00Z")
        op = FileOperation(
            file_path="/path/to/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            tool_id="toolu_001",
            prompt=prompt,
        )
        message = format_commit_message(op)

        assert f"Prompt-Id: {prompt.prompt_id}" in message
        assert "Operation: write" in message
        assert "File: /path/to/file.py" in message
        assert "Timestamp: 2025-01-01T00:00:00Z" in message
        assert "Tool-Id: toolu_001" in message


class TestNormalizeFilePaths:
    """Tests for normalize_file_paths function."""

    def test_single_file(self):
        """Should handle single file."""
        ops = [
            FileOperation(
                file_path="/path/to/file.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
            )
        ]
        prefix, mapping = normalize_file_paths(ops)
        assert prefix == "/path/to"
        assert mapping["/path/to/file.py"] == "file.py"

    def test_multiple_files_common_prefix(self):
        """Should find common prefix for multiple files."""
        ops = [
            FileOperation(
                file_path="/project/src/a.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
            ),
            FileOperation(
                file_path="/project/src/b.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
            ),
        ]
        prefix, mapping = normalize_file_paths(ops)
        assert prefix == "/project/src"
        assert mapping["/project/src/a.py"] == "a.py"
        assert mapping["/project/src/b.py"] == "b.py"

    def test_empty_operations(self):
        """Should handle empty operations list."""
        prefix, mapping = normalize_file_paths([])
        assert prefix == ""
        assert mapping == {}



# TestGitRepoBuilder deleted - tests for removed build() method

class TestParseCommitTrailers:
    """Tests for parse_commit_trailers function."""

    def test_parses_tool_id(self):
        """Should parse Tool-Id trailer."""
        message = "Create file.py\n\nTool-Id: toolu_001"
        metadata = parse_commit_trailers(message)
        assert metadata.tool_id == "toolu_001"

    def test_parses_timestamp(self):
        """Should parse Timestamp trailer."""
        message = "Create file.py\n\nTimestamp: 2025-01-01T00:00:00Z"
        metadata = parse_commit_trailers(message)
        assert metadata.timestamp == "2025-01-01T00:00:00Z"

    def test_parses_operation(self):
        """Should parse Operation trailer."""
        message = "Edit file.py\n\nOperation: edit"
        metadata = parse_commit_trailers(message)
        assert metadata.operation == "edit"

    def test_parses_file_path(self):
        """Should parse File trailer."""
        message = "Create file.py\n\nFile: /path/to/file.py"
        metadata = parse_commit_trailers(message)
        assert metadata.file_path == "/path/to/file.py"

    def test_parses_prompt_id(self):
        """Should parse Prompt-Id trailer."""
        message = "Create file.py\n\nPrompt-Id: abc123def456"
        metadata = parse_commit_trailers(message)
        assert metadata.prompt_id == "abc123def456"

    def test_parses_source_commit(self):
        """Should parse Source-Commit trailer."""
        message = "Merge commit\n\nSource-Commit: abc123"
        metadata = parse_commit_trailers(message)
        assert metadata.source_commit == "abc123"

    def test_parses_multiple_trailers(self):
        """Should parse multiple trailers."""
        message = """Create file.py

Tool-Id: toolu_001
Timestamp: 2025-01-01T00:00:00Z
Operation: write
File: /path/to/file.py"""
        metadata = parse_commit_trailers(message)
        assert metadata.tool_id == "toolu_001"
        assert metadata.timestamp == "2025-01-01T00:00:00Z"
        assert metadata.operation == "write"
        assert metadata.file_path == "/path/to/file.py"

    def test_returns_empty_metadata_for_no_trailers(self):
        """Should return empty metadata when no trailers present."""
        message = "Simple commit message"
        metadata = parse_commit_trailers(message)
        assert metadata.tool_id is None
        assert metadata.timestamp is None


class TestGetProcessedOperations:
    """Tests for get_processed_operations function."""

    def test_returns_empty_for_new_repo(self, tmp_path):
        """Should return empty set for empty repo."""
        repo = Repo.init(tmp_path)
        result = get_processed_operations(repo)
        assert result == set()

    def test_extracts_tool_ids(self, tmp_path):
        """Should extract Tool-Id values from commit trailers."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        # Create a file and commit with tool_id trailer
        test_file = tmp_path / "test.py"
        test_file.write_text("content")
        repo.index.add(["test.py"])
        repo.index.commit("Create test.py\n\nTool-Id: toolu_001\nTimestamp: 2025-01-01T00:00:00Z")

        result = get_processed_operations(repo)
        assert "toolu_001" in result

    def test_extracts_multiple_tool_ids(self, tmp_path):
        """Should extract multiple Tool-Ids from different commits."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        # Create first commit
        file1 = tmp_path / "file1.py"
        file1.write_text("content1")
        repo.index.add(["file1.py"])
        repo.index.commit("Create file1.py\n\nTool-Id: toolu_001")

        # Create second commit
        file2 = tmp_path / "file2.py"
        file2.write_text("content2")
        repo.index.add(["file2.py"])
        repo.index.commit("Create file2.py\n\nTool-Id: toolu_002")

        result = get_processed_operations(repo)
        assert "toolu_001" in result
        assert "toolu_002" in result

    def test_falls_back_to_timestamp(self, tmp_path):
        """Should use timestamp if no Tool-Id present."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        test_file = tmp_path / "test.py"
        test_file.write_text("content")
        repo.index.add(["test.py"])
        repo.index.commit("Create test.py\n\nTimestamp: 2025-01-01T00:00:00Z")

        result = get_processed_operations(repo)
        assert "ts:2025-01-01T00:00:00Z" in result

    def test_extracts_tool_ids_from_all_branches(self, tmp_path):
        """Should extract Tool-Ids from all branches, not just current branch.

        This is critical for incremental processing - session branches contain
        the actual operation commits, while main may only have merges or be empty.
        """
        repo = Repo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        # Create initial commit on main
        readme = tmp_path / "README.md"
        readme.write_text("# Project")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Create a session branch with commits
        session_branch = repo.create_head("sessions/claude_code/test-session")
        session_branch.checkout()

        # Add commits with Tool-Ids on the session branch
        file1 = tmp_path / "file1.py"
        file1.write_text("content1")
        repo.index.add(["file1.py"])
        repo.index.commit("Create file1.py\n\nTool-Id: toolu_session_001")

        file2 = tmp_path / "file2.py"
        file2.write_text("content2")
        repo.index.add(["file2.py"])
        repo.index.commit("Create file2.py\n\nTool-Id: toolu_session_002")

        # Switch back to main branch
        repo.heads.main.checkout()

        # get_processed_operations should find Tool-Ids from ALL branches
        result = get_processed_operations(repo)

        # These should be found even though we're on main
        assert "toolu_session_001" in result, "Should find Tool-Id from session branch"
        assert "toolu_session_002" in result, "Should find Tool-Id from session branch"



# TestIncrementalBuild deleted - tests for removed build() method

class TestFormatGitDate:
    """Tests for format_git_date function."""

    def test_formats_utc_timestamp(self):
        """Should format UTC timestamp correctly."""
        from agentgit.git_builder import format_git_date

        result = format_git_date("2025-01-15T14:30:45Z")
        assert result == "2025-01-15 14:30:45 +0000"

    def test_formats_timestamp_with_offset(self):
        """Should format timestamp with timezone offset."""
        from agentgit.git_builder import format_git_date

        result = format_git_date("2025-01-15T14:30:45+05:30")
        assert result == "2025-01-15 14:30:45 +0530"

    def test_formats_timestamp_with_negative_offset(self):
        """Should format timestamp with negative timezone offset."""
        from agentgit.git_builder import format_git_date

        result = format_git_date("2025-01-15T14:30:45-08:00")
        assert result == "2025-01-15 14:30:45 -0800"



# TestCommitDates, TestSessionBranchIsolation, TestEnsureInitialStateForEdit deleted
# - tests for removed build() and build_from_prompt_responses() methods
