"""Tests for agentgit.formats.claude_code module."""

import json
import tempfile
from pathlib import Path

import pytest

from agentgit.core import OperationType, Transcript
from agentgit.formats.claude_code import ClaudeCodePlugin


@pytest.fixture
def plugin():
    """Create a ClaudeCodePlugin instance."""
    return ClaudeCodePlugin()


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file."""
    content = [
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Create a hello world function"},
            "sessionId": "test-session",
            "cwd": "/test/project",
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:05.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll create a hello world function."},
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/project/hello.py",
                            "content": "def hello():\n    return 'Hello, World!'",
                        },
                    },
                ]
            },
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:10.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "Now I'll add a greeting parameter."},
                    {
                        "type": "tool_use",
                        "id": "toolu_002",
                        "name": "Edit",
                        "input": {
                            "file_path": "/test/project/hello.py",
                            "old_string": "def hello():",
                            "new_string": "def hello(name='World'):",
                        },
                    },
                ]
            },
        },
    ]

    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


class TestClaudeCodePlugin:
    """Tests for ClaudeCodePlugin."""

    def test_detect_format_jsonl(self, plugin, sample_jsonl):
        """Should detect Claude Code JSONL format."""
        result = plugin.agentgit_detect_format(sample_jsonl)
        assert result == "claude_code_jsonl"

    def test_detect_format_non_jsonl(self, plugin, tmp_path):
        """Should return None for non-JSONL files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("just some text")
        result = plugin.agentgit_detect_format(txt_file)
        assert result is None

    def test_detect_format_invalid_jsonl(self, plugin, tmp_path):
        """Should return None for invalid JSONL."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text("not valid json\n")
        result = plugin.agentgit_detect_format(jsonl_file)
        assert result is None

    def test_parse_transcript(self, plugin, sample_jsonl):
        """Should parse transcript correctly."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")

        assert transcript is not None
        assert transcript.source_format == "claude_code_jsonl"
        assert transcript.session_id == "test-session"
        assert transcript.session_cwd == "/test/project"
        assert len(transcript.entries) == 3
        assert len(transcript.prompts) == 1
        assert transcript.prompts[0].text == "Create a hello world function"

    def test_parse_transcript_wrong_format(self, plugin, sample_jsonl):
        """Should return None for wrong format."""
        result = plugin.agentgit_parse_transcript(sample_jsonl, "other_format")
        assert result is None

    def test_extract_operations(self, plugin, sample_jsonl):
        """Should extract file operations."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        assert len(operations) == 2

        # First operation: Write
        assert operations[0].operation_type == OperationType.WRITE
        assert operations[0].file_path == "/test/project/hello.py"
        assert "Hello, World!" in operations[0].content
        assert operations[0].tool_id == "toolu_001"

        # Second operation: Edit
        assert operations[1].operation_type == OperationType.EDIT
        assert operations[1].file_path == "/test/project/hello.py"
        assert operations[1].old_string == "def hello():"
        assert operations[1].new_string == "def hello(name='World'):"
        assert operations[1].tool_id == "toolu_002"

    def test_extract_operations_wrong_format(self, plugin):
        """Should return empty list for wrong format."""
        transcript = Transcript(source_format="other_format")
        operations = plugin.agentgit_extract_operations(transcript)
        assert operations == []

    def test_enrich_operation(self, plugin, sample_jsonl):
        """Should enrich operations with prompt context."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        enriched = plugin.agentgit_enrich_operation(operations[0], transcript)

        assert enriched.prompt is not None
        assert enriched.prompt.text == "Create a hello world function"

    def test_extract_deleted_paths_simple(self, plugin):
        """Should extract paths from simple rm commands."""
        paths = plugin._extract_deleted_paths("rm file.txt")
        assert paths == ["file.txt"]

    def test_extract_deleted_paths_with_flags(self, plugin):
        """Should extract paths from rm with flags."""
        paths = plugin._extract_deleted_paths("rm -rf /path/to/dir")
        assert paths == ["/path/to/dir"]

    def test_extract_deleted_paths_multiple(self, plugin):
        """Should extract multiple paths."""
        paths = plugin._extract_deleted_paths("rm file1.txt file2.txt")
        assert paths == ["file1.txt", "file2.txt"]

    def test_extract_deleted_paths_quoted(self, plugin):
        """Should handle quoted paths."""
        paths = plugin._extract_deleted_paths('rm "path with spaces/file.txt"')
        assert paths == ["path with spaces/file.txt"]

    def test_extract_text_from_string_content(self, plugin):
        """Should extract text from string content."""
        text = plugin._extract_text_from_content("Hello world")
        assert text == "Hello world"

    def test_extract_text_from_array_content(self, plugin):
        """Should extract text from array content."""
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        text = plugin._extract_text_from_content(content)
        assert text == "First part Second part"

    def test_extract_text_ignores_non_text_blocks(self, plugin):
        """Should ignore non-text blocks."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "123"},
            {"type": "text", "text": "World"},
        ]
        text = plugin._extract_text_from_content(content)
        assert text == "Hello World"


class TestClaudeCodeDiscovery:
    """Tests for Claude Code transcript discovery."""

    def test_discover_returns_empty_if_no_claude_dir(self, plugin, tmp_path, monkeypatch):
        """Should return empty if ~/.claude doesn't exist."""
        # Point home to a temp directory without .claude
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(tmp_path / "some-project")
        assert result == []

    def test_discover_returns_empty_if_no_matching_project(self, plugin, tmp_path, monkeypatch):
        """Should return empty if no matching project directory."""
        # Create .claude/projects but no matching project
        claude_dir = tmp_path / ".claude" / "projects"
        claude_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(tmp_path / "some-project")
        assert result == []

    def test_discover_finds_transcripts(self, plugin, tmp_path, monkeypatch):
        """Should find transcript files in matching project directory."""
        # Create project directory
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        # Create .claude/projects with matching directory
        encoded_path = str(project_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create transcript files
        (claude_project_dir / "session1.jsonl").write_text('{"type": "user"}\n')
        (claude_project_dir / "session2.jsonl").write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(project_path)
        assert len(result) == 2
        assert all(p.suffix == ".jsonl" for p in result)

    def test_discover_excludes_agent_files(self, plugin, tmp_path, monkeypatch):
        """Should exclude agent-*.jsonl files."""
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        encoded_path = str(project_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create regular and agent transcript files
        (claude_project_dir / "session.jsonl").write_text('{"type": "user"}\n')
        (claude_project_dir / "agent-abc123.jsonl").write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(project_path)
        assert len(result) == 1
        assert result[0].name == "session.jsonl"

    def test_discover_sorts_by_mtime(self, plugin, tmp_path, monkeypatch):
        """Should sort results by modification time, most recent first."""
        import time

        project_path = tmp_path / "myproject"
        project_path.mkdir()

        encoded_path = str(project_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create files with different modification times
        old_file = claude_project_dir / "old.jsonl"
        old_file.write_text('{"type": "user"}\n')

        time.sleep(0.01)  # Ensure different mtime

        new_file = claude_project_dir / "new.jsonl"
        new_file.write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(project_path)
        assert len(result) == 2
        assert result[0].name == "new.jsonl"  # Most recent first
        assert result[1].name == "old.jsonl"
