"""Tests for agentgit.formats.claude_code module."""

import json
import tempfile
from pathlib import Path

import pytest

from agentgit.core import OperationType, Transcript
from agentgit.formats.claude_code import ClaudeCodePlugin, get_last_timestamp_from_jsonl


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

    def test_detect_format_with_file_history_snapshot_prefix(self, plugin, tmp_path):
        """Should detect Claude Code format when file-history-snapshot is first entry.

        Newer Claude Code sessions start with metadata entries like file-history-snapshot
        before the actual user/assistant entries. The format detection should skip these
        metadata entries and look for actual conversation content.
        """
        content = [
            {"type": "file-history-snapshot", "data": {}},
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": {"content": "Hello"}},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z", "message": {"content": "Hi"}},
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = plugin.agentgit_detect_format(jsonl_path)
        assert result == "claude_code_jsonl"

    def test_detect_format_with_summary_prefix(self, plugin, tmp_path):
        """Should detect Claude Code format when summary is first entry."""
        content = [
            {"type": "summary", "summary": "Previous conversation summary"},
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": {"content": "Continue"}},
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = plugin.agentgit_detect_format(jsonl_path)
        assert result == "claude_code_jsonl"

    def test_detect_format_rejects_pure_api_call_session(self, plugin, tmp_path):
        """Should reject sessions that are purely API calls (no interactive content).

        API call sessions from the llm enhancer typically have:
        1. queue-operation: dequeue as first entry
        2. A single-shot prompt/response pattern
        3. Content indicating it's a name generation request
        """
        content = [
            {"type": "queue-operation", "operation": "dequeue", "timestamp": "2025-01-01T10:00:00.000Z"},
            {"type": "user", "timestamp": "2025-01-01T10:00:01.000Z",
             "message": {"content": "Generate concise, descriptive names for these coding sessions."}},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:02.000Z",
             "message": {"content": [{"type": "text", "text": "[\"session-name\"]"}]}},
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = plugin.agentgit_detect_format(jsonl_path)
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

    def test_extract_operations_skips_empty_file_path(self, plugin, tmp_path):
        """Should skip operations with empty or missing file_path."""
        content = [
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "",  # Empty path
                                "content": "should be skipped",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "Write",
                            "input": {
                                # Missing file_path
                                "content": "should also be skipped",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_003",
                            "name": "Write",
                            "input": {
                                "file_path": "/valid/path.py",
                                "content": "valid",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        # Should only have the valid operation
        assert len(operations) == 1
        assert operations[0].file_path == "/valid/path.py"

    def test_enrich_operation(self, plugin, sample_jsonl):
        """Should enrich operations with prompt context."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        enriched = plugin.agentgit_enrich_operation(operations[0], transcript)

        assert enriched.prompt is not None
        assert enriched.prompt.text == "Create a hello world function"

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


class TestClaudeCodeProjectName:
    """Tests for Claude Code project name extraction from cwd."""

    def test_get_project_name_from_cwd(self, plugin, tmp_path, monkeypatch):
        """Should extract project name from cwd field in transcript."""
        # Create .claude/projects directory
        claude_project_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        claude_project_dir.mkdir(parents=True)

        # Create transcript with cwd field
        transcript = claude_project_dir / "session.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "cwd": "/tmp/myproject"}) + "\n"
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_project_name(transcript)
        # Should return directory name from cwd
        assert result == "myproject"

    def test_get_project_name_returns_none_for_non_claude_transcript(self, plugin, tmp_path):
        """Should return None for transcripts not in ~/.claude/projects/."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text('{"type": "user", "cwd": "/tmp/test"}\n')

        result = plugin.agentgit_get_project_name(transcript)
        assert result is None

    def test_get_project_name_returns_none_without_cwd(self, plugin, tmp_path, monkeypatch):
        """Should return None if transcript has no cwd field."""
        claude_project_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        claude_project_dir.mkdir(parents=True)

        transcript = claude_project_dir / "session.jsonl"
        transcript.write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_project_name(transcript)
        assert result is None


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


class TestBuildPromptResponses:
    """Tests for agentgit_build_prompt_responses hook."""

    def test_concatenates_consecutive_prompts_without_operations(self, plugin, tmp_path):
        """Should concatenate prompts when no operations between them."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "First part of the request"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {"content": [{"type": "text", "text": "Let me think..."}]},
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "Second part - please also add tests"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "I'll create the file."},
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/hello.py",
                                "content": "def hello(): pass",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Should have only 1 prompt response with concatenated text
        assert len(prompt_responses) == 1
        assert "First part of the request" in prompt_responses[0].prompt.text
        assert "Second part - please also add tests" in prompt_responses[0].prompt.text
        # Should use the first prompt's timestamp
        assert prompt_responses[0].prompt.timestamp == "2025-01-01T10:00:00.000Z"

    def test_separate_prompts_when_operations_between(self, plugin, tmp_path):
        """Should keep prompts separate when there are operations between them."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "First request"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/file1.py",
                                "content": "# file 1",
                            },
                        },
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "Second request"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/file2.py",
                                "content": "# file 2",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Should have 2 separate prompt responses
        assert len(prompt_responses) == 2
        assert prompt_responses[0].prompt.text == "First request"
        assert prompt_responses[1].prompt.text == "Second request"

    def test_concatenates_multiple_consecutive_prompts(self, plugin, tmp_path):
        """Should concatenate more than two consecutive prompts."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Part one"},
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {"content": "Part two"},
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "Part three"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/hello.py",
                                "content": "# code",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Should have only 1 prompt response with all three parts
        assert len(prompt_responses) == 1
        assert "Part one" in prompt_responses[0].prompt.text
        assert "Part two" in prompt_responses[0].prompt.text
        assert "Part three" in prompt_responses[0].prompt.text

    def test_adds_context_for_short_prompt(self, plugin, tmp_path):
        """Should add assistant context when user gives short response like 'yes'.

        When there are no operations between prompts, they get concatenated.
        The short follow-up prompt gets context added before concatenation.
        """
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Help me refactor this code"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "I can help refactor this. Should I also add unit tests?",
                        }
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "yes"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/code.py",
                                "content": "# refactored",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Since no operations between prompts, they are concatenated into one
        # The short "yes" prompt gets context added
        assert len(prompt_responses) == 1
        prompt_text = prompt_responses[0].prompt.text
        assert "Help me refactor this code" in prompt_text
        assert "[Assistant context:" in prompt_text
        assert "Should I also add unit tests?" in prompt_text
        assert "yes" in prompt_text

    def test_adds_context_for_numbered_selection(self, plugin, tmp_path):
        """Should add context when user selects numbered items like 'let's do 2, 3, 4'.

        When there are no operations between prompts, they get concatenated.
        The numbered selection gets context added before concatenation.
        """
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Review the code quality"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": """Here are my recommendations:
1. Split the large function
2. Remove duplicate imports
3. Add proper field instead of reusing
4. Consolidate blob reading
5. Extract repeated patterns""",
                        }
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "let's do 2, 3, 4"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Edit",
                            "input": {
                                "file_path": "/test/code.py",
                                "old_string": "import re",
                                "new_string": "",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Since no operations between prompts, they are concatenated into one
        # The "let's do 2, 3, 4" gets context with the numbered list
        assert len(prompt_responses) == 1
        prompt_text = prompt_responses[0].prompt.text
        assert "Review the code quality" in prompt_text
        assert "[Assistant context:" in prompt_text
        assert "1. Split the large function" in prompt_text
        assert "2. Remove duplicate imports" in prompt_text
        assert "let's do 2, 3, 4" in prompt_text

    def test_no_context_for_detailed_prompt(self, plugin, tmp_path):
        """Should NOT add context when user gives a detailed, standalone prompt."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Start a new project"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "What kind of project?"},
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {"file_path": "/test/init.py", "content": ""},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {
                    "content": "Please create a Python web server using FastAPI with endpoints for user authentication including login, logout, and password reset functionality"
                },
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/server.py",
                                "content": "# server",
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

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # The detailed prompt should NOT have context added
        assert len(prompt_responses) == 2
        second_prompt = prompt_responses[1].prompt.text
        assert "[Assistant context:" not in second_prompt
        assert "FastAPI" in second_prompt


class TestPromptNeedsContext:
    """Tests for _prompt_needs_context helper."""

    def test_short_prompts_need_context(self, plugin):
        """Short prompts should need context."""
        assert plugin._prompt_needs_context("yes") is True
        assert plugin._prompt_needs_context("no") is True
        assert plugin._prompt_needs_context("ok do it") is True
        assert plugin._prompt_needs_context("sure, go ahead") is True

    def test_numbered_selections_need_context(self, plugin):
        """Prompts selecting numbered items need context."""
        assert plugin._prompt_needs_context("let's do 2, 3, 4") is True
        assert plugin._prompt_needs_context("do 1 and 3") is True
        assert plugin._prompt_needs_context("items 2 & 5") is True
        assert plugin._prompt_needs_context("just 1, 2") is True

    def test_referential_prompts_need_context(self, plugin):
        """Prompts that reference something need context."""
        assert plugin._prompt_needs_context("that sounds good") is True
        assert plugin._prompt_needs_context("this is what I want") is True
        assert plugin._prompt_needs_context("the first option") is True

    def test_detailed_prompts_dont_need_context(self, plugin):
        """Detailed standalone prompts don't need context."""
        assert (
            plugin._prompt_needs_context(
                "Create a new Python class called UserManager with methods for CRUD operations"
            )
            is False
        )
        assert (
            plugin._prompt_needs_context(
                "Please refactor the authentication module to use JWT tokens instead of sessions"
            )
            is False
        )


class TestIsContinuationPrompt:
    """Tests for _is_continuation_prompt method."""

    def test_detects_continuation_prompt(self, plugin):
        """Should detect session continuation messages."""
        continuation_text = """This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:

Analysis:
Let me analyze the conversation...

Summary:
1. User asked to implement feature X
2. We were working on file Y"""

        assert plugin._is_continuation_prompt(continuation_text) is True

    def test_detects_partial_continuation_markers(self, plugin):
        """Should detect any continuation marker."""
        assert plugin._is_continuation_prompt(
            "This session is being continued from a previous conversation"
        ) is True
        assert plugin._is_continuation_prompt(
            "Some text about conversation that ran out of context here"
        ) is True
        assert plugin._is_continuation_prompt(
            "The conversation is summarized below:"
        ) is True

    def test_normal_prompts_not_continuation(self, plugin):
        """Normal user prompts should not be detected as continuation."""
        assert plugin._is_continuation_prompt("Add a new feature") is False
        assert plugin._is_continuation_prompt("Fix the bug in auth.py") is False
        assert plugin._is_continuation_prompt("yes") is False
        assert plugin._is_continuation_prompt(
            "Please continue working on the previous task"
        ) is False


class TestGetLastTimestamp:
    """Tests for get_last_timestamp_from_jsonl helper function."""

    def test_extracts_last_timestamp(self, tmp_path):
        """Should extract the timestamp from the last entry."""
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": "first"},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z", "message": "second"},
            {"type": "user", "timestamp": "2025-01-01T10:00:10.000Z", "message": "third"},
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should return the last timestamp (2025-01-01T10:00:10)
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:10.000+00:00").timestamp()
        assert result == expected

    def test_skips_entries_without_timestamp(self, tmp_path):
        """Should skip entries that don't have a timestamp field."""
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": "first"},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z", "message": "second"},
            {"type": "other", "message": "no timestamp"},  # No timestamp
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should return the second timestamp since last entry has none
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:05.000+00:00").timestamp()
        assert result == expected

    def test_handles_empty_file(self, tmp_path):
        """Should return None for empty file."""
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        result = get_last_timestamp_from_jsonl(jsonl_path)
        assert result is None

    def test_handles_truncated_first_line(self, tmp_path):
        """Should handle files larger than 8KB by skipping truncated first line."""
        # Create a file with many entries to exceed 8KB
        content = []
        for i in range(200):
            # Use i as seconds (0-59 valid), cycle through minutes
            minute = i // 60
            second = i % 60
            content.append({
                "type": "user",
                "timestamp": f"2025-01-01T10:{minute:02d}:{second:02d}.000Z",
                "message": f"message {i}" * 100,  # Make it large
            })

        jsonl_path = tmp_path / "large.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should still extract the last timestamp correctly
        from datetime import datetime
        # Last entry was i=199 -> minute=3, second=19
        expected = datetime.fromisoformat("2025-01-01T10:03:19.000+00:00").timestamp()
        assert result == expected

    def test_handles_invalid_json_lines(self, tmp_path):
        """Should skip invalid JSON lines and continue searching."""
        jsonl_path = tmp_path / "mixed.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"type": "user", "timestamp": "2025-01-01T10:00:00.000Z"}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z"}) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should return the last valid timestamp
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:05.000+00:00").timestamp()
        assert result == expected

    def test_returns_none_if_no_timestamps(self, tmp_path):
        """Should return None if file has no entries with timestamps."""
        content = [
            {"type": "other", "message": "no timestamp"},
            {"type": "other", "data": "also no timestamp"},
        ]

        jsonl_path = tmp_path / "no_timestamps.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)
        assert result is None


class TestClaudeCodeGetLastTimestamp:
    """Tests for agentgit_get_last_timestamp hook."""

    def test_returns_timestamp_for_claude_code_transcript(self, plugin, tmp_path, monkeypatch):
        """Should return last timestamp for Claude Code transcripts."""
        # Create .claude/projects directory
        claude_project_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        claude_project_dir.mkdir(parents=True)

        # Create transcript with timestamps
        transcript = claude_project_dir / "session.jsonl"
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z"},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:10.000Z"},
        ]
        with open(transcript, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_last_timestamp(transcript)

        # Should return the last timestamp
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:10.000+00:00").timestamp()
        assert result == expected

    def test_returns_none_for_non_claude_transcript(self, plugin, tmp_path):
        """Should return None for transcripts not in ~/.claude/projects/."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text('{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z"}\n')

        result = plugin.agentgit_get_last_timestamp(transcript)
        assert result is None


class TestClaudeCodeAuthorInfo:
    """Tests for agentgit_get_author_info hook."""

    def test_returns_author_info_for_claude_code_transcript(self, plugin, sample_jsonl):
        """Should return Claude author info for Claude Code transcripts."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")

        result = plugin.agentgit_get_author_info(transcript)

        assert result is not None
        assert result["name"] == "Claude"
        assert result["email"] == "claude@anthropic.com"

    def test_returns_none_for_non_claude_transcript(self, plugin):
        """Should return None for non-Claude Code transcripts."""
        transcript = Transcript(source_format="other_format")

        result = plugin.agentgit_get_author_info(transcript)

        assert result is None


class TestTaskToolGrouping:
    """Tests for grouping operations by Task tool calls."""

    def test_task_tool_creates_scene_grouping(self, plugin, tmp_path):
        """Task tool calls should create scenes that group subsequent operations.

        When a Task tool is called, all file operations from that subagent's work
        should be grouped into a single scene, rather than individual commits.
        """
        # Create transcript with Task tool call followed by operations
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Research the codebase structure"},
                "sessionId": "test-session",
                "cwd": "/test/project",
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "I'll use the Task tool to explore the codebase."},
                        {
                            "type": "tool_use",
                            "id": "task_001",
                            "name": "Task",
                            "input": {
                                "description": "Explore codebase",
                                "prompt": "Search for main entry points",
                                "subagent_type": "Explore",
                            },
                        },
                    ]
                },
            },
            # Tool result from Task
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "task_001",
                            "content": "Found main.py as the entry point",
                        }
                    ]
                },
            },
            # Assistant reads files as part of the task's work
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "Based on the exploration, let me create a summary file."},
                        {
                            "type": "tool_use",
                            "id": "write_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/project/ARCHITECTURE.md",
                                "content": "# Architecture\n\nMain entry: main.py",
                            },
                        },
                    ]
                },
            },
            # Tool result
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:20.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "write_001",
                            "content": "File written successfully",
                        }
                    ]
                },
            },
            # Another write as part of the same task
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:25.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "Also creating a README."},
                        {
                            "type": "tool_use",
                            "id": "write_002",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/project/README.md",
                                "content": "# Project\n\nSee ARCHITECTURE.md for details.",
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

        # Parse the transcript
        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")

        # Extract operations and assign to transcript (required for scene building)
        operations = plugin.agentgit_extract_operations(transcript)
        transcript.operations = operations  # Must set this for agentgit_build_scenes to work
        assert len(operations) == 2, "Should have 2 write operations"

        # Build scenes - this should group operations by Task tool boundaries
        scenes = plugin.agentgit_build_scenes(transcript)

        # The key assertion: Task tool should create scene grouping
        # All operations after the Task call should be grouped into one scene
        assert len(scenes) > 0, "Should create at least one scene for Task tool work"

        # Find the scene with operations
        scenes_with_ops = [s for s in scenes if s.operations]
        assert len(scenes_with_ops) == 1, (
            "All operations from Task subagent work should be in a single scene, "
            f"but got {len(scenes_with_ops)} scenes with operations"
        )

        # The scene should have both operations grouped together
        scene = scenes_with_ops[0]
        assert len(scene.operations) == 2, (
            f"Scene should contain both operations (got {len(scene.operations)})"
        )

        # Scene should have the task description
        assert scene.task_description is not None, "Scene should have task_description set"
        assert "Explore" in scene.task_description or "codebase" in scene.task_description.lower()


class TestBuildScenesHookIntegration:
    """Tests for agentgit_build_scenes hook integration with plugin manager.

    Verifies that the hook correctly returns scenes when multiple plugins are
    registered, handling the firstresult=True semantics properly.
    """

    def test_build_scenes_returns_scenes_via_plugin_manager(self, tmp_path):
        """Plugin manager hook should return scenes for claude_code_jsonl format.

        This tests the integration between the plugin manager and plugins.
        With firstresult=True, plugins must return None (not []) when they
        don't handle a format, otherwise an empty list from one plugin blocks
        other plugins from providing results.
        """
        from agentgit.plugins import get_configured_plugin_manager

        # Create a simple transcript with file operations
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Create a test file"},
                "sessionId": "test-session",
                "cwd": "/test/project",
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "Creating test file."},
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/project/test.py",
                                "content": "# test file",
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

        pm = get_configured_plugin_manager()

        # Detect format and parse
        format = pm.hook.agentgit_detect_format(path=jsonl_path)
        assert format == "claude_code_jsonl"

        transcript = pm.hook.agentgit_parse_transcript(path=jsonl_path, format=format)

        # Extract and assign operations
        operations = []
        for ops in pm.hook.agentgit_extract_operations(transcript=transcript):
            operations.extend(ops)
        transcript.operations = operations
        assert len(operations) == 1, "Should have 1 write operation"

        # Build scenes via plugin manager hook
        scenes = pm.hook.agentgit_build_scenes(transcript=transcript)

        # Key assertion: scenes should be built even with multiple plugins registered
        assert len(scenes) == 1, (
            f"Expected 1 scene from plugin manager hook, got {len(scenes)}. "
            "This may indicate a plugin returning [] instead of None for unhandled formats."
        )
