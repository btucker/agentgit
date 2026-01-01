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
    format_prompt_merge_message,
    format_turn_commit_message,
    get_last_processed_timestamp,
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

    def test_includes_prompt_text(self):
        """Should include full prompt text without truncation."""
        long_prompt = "x" * 1000
        prompt = Prompt(text=long_prompt, timestamp="2025-01-01T00:00:00Z")
        op = FileOperation(
            file_path="/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            prompt=prompt,
        )
        message = format_commit_message(op)
        assert f"Prompt #{prompt.short_id}:" in message
        assert long_prompt in message

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


class TestGitRepoBuilder:
    """Tests for GitRepoBuilder class."""

    def test_build_creates_repo(self, tmp_path):
        """Should create a git repository."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            )
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, repo_path, mapping = builder.build(ops)

        assert repo is not None
        assert repo_path == tmp_path
        assert (tmp_path / ".git").exists()

    def test_build_creates_temp_dir_if_none(self):
        """Should create temp directory if none provided."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            )
        ]

        builder = GitRepoBuilder()
        repo, repo_path, mapping = builder.build(ops)

        assert repo_path.exists()
        assert "agentgit_" in str(repo_path)

    def test_write_operation_creates_file(self, tmp_path):
        """Should create file for write operation."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            )
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        assert (tmp_path / "hello.py").exists()
        assert (tmp_path / "hello.py").read_text() == "print('hello')"

    def test_edit_operation_modifies_file(self, tmp_path):
        """Should modify file for edit operation."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            ),
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.EDIT,
                timestamp="2025-01-01T00:00:01Z",
                old_string="hello",
                new_string="world",
            ),
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        assert (tmp_path / "hello.py").read_text() == "print('world')"

    def test_delete_operation_removes_file(self, tmp_path):
        """Should remove file for delete operation."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            ),
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.DELETE,
                timestamp="2025-01-01T00:00:01Z",
            ),
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        assert not (tmp_path / "hello.py").exists()

    def test_creates_commits(self, tmp_path):
        """Should create commits for each operation."""
        prompt = Prompt(text="Add hello function", timestamp="2025-01-01T00:00:00Z")
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="def hello(): pass",
                prompt=prompt,
            ),
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.EDIT,
                timestamp="2025-01-01T00:00:01Z",
                old_string="pass",
                new_string="return 'hello'",
                prompt=prompt,
            ),
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build(ops)

        commits = list(repo.iter_commits())
        assert len(commits) == 2

    def test_custom_author(self, tmp_path):
        """Should use custom author name and email."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            )
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build(ops, author_name="Test Author", author_email="test@example.com")

        commit = list(repo.iter_commits())[0]
        assert commit.author.name == "Test Author"
        assert commit.author.email == "test@example.com"

    def test_edit_with_original_content(self, tmp_path):
        """Should use original_content for files not yet created."""
        ops = [
            FileOperation(
                file_path="/project/existing.py",
                operation_type=OperationType.EDIT,
                timestamp="2025-01-01T00:00:00Z",
                old_string="old",
                new_string="new",
                original_content="some old content here",
            ),
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        # Should have created the file with edited content
        content = (tmp_path / "existing.py").read_text()
        assert "new" in content
        assert "old" not in content

    def test_replace_all_edit(self, tmp_path):
        """Should replace all occurrences when replace_all is True."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="foo foo foo",
            ),
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.EDIT,
                timestamp="2025-01-01T00:00:01Z",
                old_string="foo",
                new_string="bar",
                replace_all=True,
            ),
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        assert (tmp_path / "hello.py").read_text() == "bar bar bar"

    def test_creates_nested_directories(self, tmp_path):
        """Should create nested directories for files with different paths."""
        ops = [
            FileOperation(
                file_path="/project/src/lib/utils/helper.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="# helper",
            ),
            FileOperation(
                file_path="/project/src/main.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:01Z",
                content="# main",
            ),
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        # Common prefix is /project/src, so paths are relative to that
        assert (tmp_path / "lib" / "utils" / "helper.py").exists()
        assert (tmp_path / "main.py").exists()


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


class TestGetLastProcessedTimestamp:
    """Tests for get_last_processed_timestamp function."""

    def test_returns_none_for_empty_repo(self, tmp_path):
        """Should return None for empty repo."""
        repo = Repo.init(tmp_path)
        result = get_last_processed_timestamp(repo)
        assert result is None

    def test_returns_most_recent_timestamp(self, tmp_path):
        """Should return timestamp from most recent commit."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        # Create first commit with older timestamp
        file1 = tmp_path / "file1.py"
        file1.write_text("content1")
        repo.index.add(["file1.py"])
        repo.index.commit("Create file1.py\n\nTimestamp: 2025-01-01T00:00:00Z")

        # Create second commit with newer timestamp
        file2 = tmp_path / "file2.py"
        file2.write_text("content2")
        repo.index.add(["file2.py"])
        repo.index.commit("Create file2.py\n\nTimestamp: 2025-01-02T00:00:00Z")

        result = get_last_processed_timestamp(repo)
        # Most recent commit is returned first by iter_commits
        assert result == "2025-01-02T00:00:00Z"


class TestIncrementalBuild:
    """Tests for incremental build functionality."""

    def test_skips_already_processed_operations_by_tool_id(self, tmp_path):
        """Should skip operations already processed based on Tool-Id."""
        ops1 = [
            FileOperation(
                file_path="/project/file1.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="content1",
                tool_id="toolu_001",
            )
        ]

        # First build
        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build(ops1)
        initial_commits = len(list(repo.iter_commits()))
        assert initial_commits == 1

        # Second build with same operation plus a new one
        ops2 = [
            FileOperation(
                file_path="/project/file1.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="content1",
                tool_id="toolu_001",  # Already processed
            ),
            FileOperation(
                file_path="/project/file2.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:01Z",
                content="content2",
                tool_id="toolu_002",  # New
            ),
        ]

        builder2 = GitRepoBuilder(output_dir=tmp_path)
        repo2, _, _ = builder2.build(ops2)

        final_commits = len(list(repo2.iter_commits()))
        # Should only add 1 new commit (toolu_002)
        assert final_commits == 2

    def test_skips_already_processed_operations_by_timestamp(self, tmp_path):
        """Should skip operations by timestamp when no Tool-Id."""
        ops1 = [
            FileOperation(
                file_path="/project/file1.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="content1",
            )
        ]

        # First build
        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build(ops1)
        initial_commits = len(list(repo.iter_commits()))
        assert initial_commits == 1

        # Second build with same timestamp
        ops2 = [
            FileOperation(
                file_path="/project/file1.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",  # Same timestamp
                content="content1",
            ),
            FileOperation(
                file_path="/project/file2.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:01Z",  # New timestamp
                content="content2",
            ),
        ]

        builder2 = GitRepoBuilder(output_dir=tmp_path)
        repo2, _, _ = builder2.build(ops2)

        final_commits = len(list(repo2.iter_commits()))
        assert final_commits == 2

    def test_incremental_false_reprocesses_all(self, tmp_path):
        """Should reprocess all operations when incremental=False."""
        ops = [
            FileOperation(
                file_path="/project/file1.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="content1",
                tool_id="toolu_001",
            )
        ]

        # First build
        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        # Clear the repo and rebuild
        import shutil
        shutil.rmtree(tmp_path / ".git")

        # Re-init with incremental=False on existing dir
        builder2 = GitRepoBuilder(output_dir=tmp_path)
        repo2, _, _ = builder2.build(ops, incremental=False)

        commits = len(list(repo2.iter_commits()))
        assert commits == 1

    def test_loads_file_states_from_existing_repo(self, tmp_path):
        """Should preserve file state when adding incremental edits."""
        # First build: create a file
        ops1 = [
            FileOperation(
                file_path="/project/file.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="original content",
                tool_id="toolu_001",
            )
        ]

        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops1)

        # Second build: edit the file
        ops2 = [
            FileOperation(
                file_path="/project/file.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="original content",
                tool_id="toolu_001",  # Skip this
            ),
            FileOperation(
                file_path="/project/file.py",
                operation_type=OperationType.EDIT,
                timestamp="2025-01-01T00:00:01Z",
                old_string="original",
                new_string="modified",
                tool_id="toolu_002",
            ),
        ]

        builder2 = GitRepoBuilder(output_dir=tmp_path)
        builder2.build(ops2)

        # File should have modified content
        content = (tmp_path / "file.py").read_text()
        assert content == "modified content"

    def test_returns_early_if_no_new_operations(self, tmp_path):
        """Should return existing repo if all operations already processed."""
        ops = [
            FileOperation(
                file_path="/project/file.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="content",
                tool_id="toolu_001",
            )
        ]

        # First build
        builder = GitRepoBuilder(output_dir=tmp_path)
        builder.build(ops)

        # Second build with same operations
        builder2 = GitRepoBuilder(output_dir=tmp_path)
        repo, path, mapping = builder2.build(ops)

        # Should still return valid repo
        assert repo is not None
        assert len(list(repo.iter_commits())) == 1


class TestFormatTurnCommitMessage:
    """Tests for format_turn_commit_message function."""

    def test_uses_summary_line(self):
        """Should use assistant context text as summary."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/path/to/file.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                )
            ],
            context=AssistantContext(text="Creating a new helper function"),
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_turn_commit_message(turn)
        assert message.startswith("Creating a new helper function")

    def test_falls_back_to_operation_summary(self):
        """Should describe operation when no context."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/path/to/helper.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_turn_commit_message(turn)
        assert message.startswith("Create helper.py")

    def test_includes_file_lists(self):
        """Should list created and modified files."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/path/to/new.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                ),
                FileOperation(
                    file_path="/path/to/existing.py",
                    operation_type=OperationType.EDIT,
                    timestamp="2025-01-01T00:00:01Z",
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_turn_commit_message(turn)
        assert "Created: new.py" in message
        assert "Modified: existing.py" in message

    def test_includes_trailers(self):
        """Should include timestamp trailer."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/path/to/file.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    tool_id="toolu_001",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_turn_commit_message(turn)
        assert "Timestamp: 2025-01-01T00:00:00Z" in message
        assert "Tool-Id: toolu_001" in message


class TestFormatPromptMergeMessage:
    """Tests for format_prompt_merge_message function."""

    def test_uses_prompt_first_line(self):
        """Should use first line of prompt as subject."""
        prompt = Prompt(
            text="Add user authentication\nWith login and logout support",
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_prompt_merge_message(prompt, [])
        assert message.startswith("Add user authentication")

    def test_truncates_long_subject(self):
        """Should truncate long first line."""
        long_text = "x" * 100
        prompt = Prompt(text=long_text, timestamp="2025-01-01T00:00:00Z")
        message = format_prompt_merge_message(prompt, [])
        first_line = message.split("\n")[0]
        assert len(first_line) <= 72
        assert first_line.endswith("...")

    def test_includes_full_prompt(self):
        """Should include full prompt text in body."""
        prompt = Prompt(
            text="Add user authentication\nWith login and logout support",
            timestamp="2025-01-01T00:00:00Z",
        )
        message = format_prompt_merge_message(prompt, [])
        assert f"Prompt #{prompt.short_id}:" in message
        assert "With login and logout support" in message

    def test_includes_prompt_id_trailer(self):
        """Should include Prompt-Id trailer."""
        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        message = format_prompt_merge_message(prompt, [])
        assert f"Prompt-Id: {prompt.prompt_id}" in message


class TestBuildFromPromptResponses:
    """Tests for build_from_prompt_responses method."""

    def test_creates_initial_commit(self, tmp_path):
        """Should create initial commit if repo is empty."""
        prompt = Prompt(text="Add hello function", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/hello.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="def hello(): pass",
                    prompt=prompt,
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn])

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build_from_prompt_responses([pr])

        # Should have: initial commit + turn commit + merge commit
        commits = list(repo.iter_commits())
        assert len(commits) >= 2

    def test_creates_merge_commits(self, tmp_path):
        """Should create merge commits for each prompt."""
        prompt = Prompt(text="Add hello function", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/hello.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="def hello(): pass",
                    prompt=prompt,
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn])

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build_from_prompt_responses([pr])

        # Check that HEAD is a merge commit
        head = repo.head.commit
        assert len(head.parents) == 2 or "Prompt #" in head.message

    def test_groups_multiple_operations_in_turn(self, tmp_path):
        """Should create single commit for multiple operations in a turn."""
        prompt = Prompt(text="Setup project", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/file1.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="# file1",
                    prompt=prompt,
                    tool_id="toolu_001",
                ),
                FileOperation(
                    file_path="/project/file2.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:01Z",
                    content="# file2",
                    prompt=prompt,
                    tool_id="toolu_002",
                ),
            ],
            context=AssistantContext(text="Creating project files"),
            timestamp="2025-01-01T00:00:00Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn])

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build_from_prompt_responses([pr])

        # Both files should exist
        assert (tmp_path / "file1.py").exists()
        assert (tmp_path / "file2.py").exists()

        # The turn commit should contain both Tool-Ids
        for commit in repo.iter_commits():
            if "Creating project files" in commit.message:
                assert "Tool-Id: toolu_001" in commit.message
                assert "Tool-Id: toolu_002" in commit.message
                break

    def test_handles_empty_prompt_responses(self, tmp_path):
        """Should handle empty list of prompt responses."""
        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, path, mapping = builder.build_from_prompt_responses([])

        assert repo is not None
        assert path == tmp_path

    def test_first_parent_shows_prompts(self, tmp_path):
        """git log --first-parent should show only prompt merges."""
        prompt1 = Prompt(text="First task", timestamp="2025-01-01T10:00:00Z")
        turn1 = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/first.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T10:00:01Z",
                    content="# first",
                    prompt=prompt1,
                    tool_id="toolu_001",
                )
            ],
            timestamp="2025-01-01T10:00:01Z",
        )
        pr1 = PromptResponse(prompt=prompt1, turns=[turn1])

        prompt2 = Prompt(text="Second task", timestamp="2025-01-01T11:00:00Z")
        turn2 = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/second.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T11:00:01Z",
                    content="# second",
                    prompt=prompt2,
                    tool_id="toolu_002",
                )
            ],
            timestamp="2025-01-01T11:00:01Z",
        )
        pr2 = PromptResponse(prompt=prompt2, turns=[turn2])

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build_from_prompt_responses([pr1, pr2])

        # Get first-parent commits (simulating git log --first-parent)
        first_parent_commits = []
        commit = repo.head.commit
        while commit:
            first_parent_commits.append(commit)
            if commit.parents:
                commit = commit.parents[0]
            else:
                break

        # Should have merge commits for each prompt
        prompt_messages = [c.message for c in first_parent_commits if "Prompt #" in c.message]
        assert len(prompt_messages) == 2

    def test_edit_operations_in_grouped_build(self, tmp_path):
        """Should handle edit operations in grouped build."""
        prompt = Prompt(text="Refactor code", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/code.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="def hello(): pass",
                    tool_id="toolu_001",
                ),
                FileOperation(
                    file_path="/project/code.py",
                    operation_type=OperationType.EDIT,
                    timestamp="2025-01-01T00:00:01Z",
                    old_string="pass",
                    new_string="return 'hello'",
                    tool_id="toolu_002",
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn])

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build_from_prompt_responses([pr])

        # File should have edited content
        assert (tmp_path / "code.py").read_text() == "def hello(): return 'hello'"

    def test_delete_operations_in_grouped_build(self, tmp_path):
        """Should handle delete operations in grouped build."""
        prompt = Prompt(text="Cleanup files", timestamp="2025-01-01T00:00:00Z")
        turn1 = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/temp.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="# temp file",
                    tool_id="toolu_001",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        turn2 = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/temp.py",
                    operation_type=OperationType.DELETE,
                    timestamp="2025-01-01T00:00:01Z",
                    tool_id="toolu_002",
                )
            ],
            timestamp="2025-01-01T00:00:01Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn1, turn2])

        builder = GitRepoBuilder(output_dir=tmp_path)
        repo, _, _ = builder.build_from_prompt_responses([pr])

        # File should be deleted
        assert not (tmp_path / "temp.py").exists()
