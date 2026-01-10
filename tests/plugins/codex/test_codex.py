"""Tests for agentgit.formats.codex module."""

import json
from pathlib import Path

import pytest

from agentgit.core import OperationType, Transcript
from agentgit.formats.codex import CodexPlugin


@pytest.fixture
def plugin():
    """Create a CodexPlugin instance."""
    return CodexPlugin()


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample Codex JSONL file with apply_patch operations."""
    content = [
        # Thread started
        {
            "type": "thread.started",
            "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a53",
            "timestamp": "2025-01-01T10:00:00.000Z",
        },
        # User message
        {
            "type": "message",
            "role": "user",
            "timestamp": "2025-01-01T10:00:01.000Z",
            "content": [
                {"type": "input_text", "text": "Create a hello world function"}
            ],
        },
        # Turn started
        {"type": "turn.started", "timestamp": "2025-01-01T10:00:02.000Z"},
        # Reasoning item
        {
            "type": "item.completed",
            "timestamp": "2025-01-01T10:00:03.000Z",
            "item": {
                "id": "item_0",
                "type": "reasoning",
                "text": "I'll create a simple hello world function in Python.",
            },
        },
        # Function call with apply_patch to create a file
        {
            "type": "function_call",
            "call_id": "call_001",
            "name": "shell",
            "timestamp": "2025-01-01T10:00:04.000Z",
            "arguments": json.dumps({
                "cmd": [
                    "apply_patch",
                    "*** Begin Patch\n*** Add File: hello.py\ndef hello():\n    return 'Hello, World!'\n*** End Patch",
                ]
            }),
        },
        # Function call output
        {
            "type": "function_call_output",
            "call_id": "call_001",
            "timestamp": "2025-01-01T10:00:05.000Z",
            "output": json.dumps({"output": "Patch applied successfully", "metadata": {"exit_code": 0}}),
        },
        # Agent message
        {
            "type": "item.completed",
            "timestamp": "2025-01-01T10:00:06.000Z",
            "item": {
                "id": "item_1",
                "type": "agent_message",
                "text": "I've created the hello.py file with a hello world function.",
            },
        },
        # Turn completed
        {
            "type": "turn.completed",
            "timestamp": "2025-01-01T10:00:07.000Z",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
        # Second user message
        {
            "type": "message",
            "role": "user",
            "timestamp": "2025-01-01T10:00:10.000Z",
            "content": [
                {"type": "input_text", "text": "Add a name parameter"}
            ],
        },
        {"type": "turn.started", "timestamp": "2025-01-01T10:00:11.000Z"},
        # Function call with apply_patch to update the file
        {
            "type": "function_call",
            "call_id": "call_002",
            "name": "shell",
            "timestamp": "2025-01-01T10:00:12.000Z",
            "arguments": json.dumps({
                "cmd": [
                    "apply_patch",
                    "*** Begin Patch\n*** Update File: hello.py\n@@ -1,2 +1,2 @@\n-def hello():\n+def hello(name='World'):\n     return 'Hello, World!'\n*** End Patch",
                ]
            }),
        },
        {
            "type": "function_call_output",
            "call_id": "call_002",
            "timestamp": "2025-01-01T10:00:13.000Z",
            "output": json.dumps({"output": "Patch applied successfully", "metadata": {"exit_code": 0}}),
        },
        {
            "type": "turn.completed",
            "timestamp": "2025-01-01T10:00:14.000Z",
            "usage": {"input_tokens": 150, "output_tokens": 60},
        },
    ]

    jsonl_path = tmp_path / "rollout-2025-01-01T10-00-00.jsonl"
    with open(jsonl_path, "w") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


@pytest.fixture
def sample_jsonl_with_delete(tmp_path):
    """Create a sample Codex JSONL file with a delete operation."""
    content = [
        {"type": "thread.started", "thread_id": "test-thread", "timestamp": "2025-01-01T10:00:00.000Z"},
        {
            "type": "message",
            "role": "user",
            "timestamp": "2025-01-01T10:00:01.000Z",
            "content": [{"type": "input_text", "text": "Delete the old file"}],
        },
        {"type": "turn.started", "timestamp": "2025-01-01T10:00:02.000Z"},
        {
            "type": "function_call",
            "call_id": "call_del",
            "name": "shell",
            "timestamp": "2025-01-01T10:00:03.000Z",
            "arguments": json.dumps({
                "cmd": [
                    "apply_patch",
                    "*** Begin Patch\n*** Delete File: old_file.py\n*** End Patch",
                ]
            }),
        },
        {
            "type": "function_call_output",
            "call_id": "call_del",
            "timestamp": "2025-01-01T10:00:04.000Z",
            "output": json.dumps({"output": "File deleted", "metadata": {"exit_code": 0}}),
        },
        {"type": "turn.completed", "timestamp": "2025-01-01T10:00:05.000Z"},
    ]

    jsonl_path = tmp_path / "rollout-delete.jsonl"
    with open(jsonl_path, "w") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


class TestCodexPlugin:
    """Tests for CodexPlugin."""

    def test_plugin_info(self, plugin):
        """Should return plugin identification info."""
        info = plugin.agentgit_get_plugin_info()
        assert info["name"] == "codex"
        assert "Codex" in info["description"]

    def test_get_session_id_from_path_codex_format(self, plugin, tmp_path):
        """Should extract UUID from Codex filename format."""
        codex_path = tmp_path / "rollout-2026-01-06T19-18-12-019b9608-6a5d-7212-beb9-b97edb538adf.jsonl"
        codex_path.touch()
        result = plugin.agentgit_get_session_id_from_path(codex_path)
        assert result == "019b9608-6a5d-7212-beb9-b97edb538adf"

    def test_get_session_id_from_path_uuid_only(self, plugin, tmp_path):
        """Should extract UUID from simple UUID filename."""
        uuid_path = tmp_path / "f3d29046-1b89-42ae-9da0-96cda9dd8c90.jsonl"
        uuid_path.touch()
        result = plugin.agentgit_get_session_id_from_path(uuid_path)
        assert result == "f3d29046-1b89-42ae-9da0-96cda9dd8c90"

    def test_get_session_id_from_path_no_uuid(self, plugin, tmp_path):
        """Should return None for paths without UUID."""
        plain_path = tmp_path / "session.jsonl"
        plain_path.touch()
        result = plugin.agentgit_get_session_id_from_path(plain_path)
        assert result is None

    def test_detect_format_jsonl(self, plugin, sample_jsonl):
        """Should detect Codex JSONL format."""
        result = plugin.agentgit_detect_format(sample_jsonl)
        assert result == "codex_jsonl"

    def test_detect_format_non_jsonl(self, plugin, tmp_path):
        """Should return None for non-JSONL files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("just some text")
        result = plugin.agentgit_detect_format(txt_file)
        assert result is None

    def test_detect_format_claude_code_jsonl(self, plugin, tmp_path):
        """Should return None for Claude Code JSONL files."""
        # Claude Code uses type: "user", not type: "message" with role: "user"
        jsonl_file = tmp_path / "session.jsonl"
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00Z", "message": {}},
        ]
        with open(jsonl_file, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")
        result = plugin.agentgit_detect_format(jsonl_file)
        assert result is None

    def test_detect_format_invalid_jsonl(self, plugin, tmp_path):
        """Should return None for invalid JSONL."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text("not valid json\n")
        result = plugin.agentgit_detect_format(jsonl_file)
        assert result is None

    def test_parse_transcript(self, plugin, sample_jsonl):
        """Should parse transcript correctly."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "codex_jsonl")

        assert transcript is not None
        assert transcript.source_format == "codex_jsonl"
        assert transcript.session_id == "0199a213-81c0-7800-8aa1-bbab2a035a53"
        assert len(transcript.prompts) == 2
        assert transcript.prompts[0].text == "Create a hello world function"
        assert transcript.prompts[1].text == "Add a name parameter"

    def test_parse_transcript_wrong_format(self, plugin, sample_jsonl):
        """Should return None for wrong format."""
        result = plugin.agentgit_parse_transcript(sample_jsonl, "other_format")
        assert result is None

    def test_parse_transcript_skips_state_records(self, plugin, tmp_path):
        """Should skip state records when parsing."""
        content = [
            {"type": "thread.started", "thread_id": "test"},
            {"record_type": "state", "data": "some state"},  # Should be skipped
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        ]

        jsonl_path = tmp_path / "rollout-test.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "codex_jsonl")
        assert transcript is not None
        assert len(transcript.prompts) == 1

    def test_extract_operations_add_file(self, plugin, sample_jsonl):
        """Should extract Add File operations from apply_patch."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "codex_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        assert len(operations) >= 1

        # First operation should be WRITE (Add File)
        add_op = [op for op in operations if op.operation_type == OperationType.WRITE]
        assert len(add_op) == 1
        assert add_op[0].file_path == "hello.py"
        assert "def hello():" in add_op[0].content
        assert add_op[0].tool_id == "call_001"

    def test_extract_operations_update_file(self, plugin, sample_jsonl):
        """Should extract Update File operations from apply_patch."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "codex_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        # Second operation should be EDIT (Update File)
        edit_ops = [op for op in operations if op.operation_type == OperationType.EDIT]
        assert len(edit_ops) == 1
        assert edit_ops[0].file_path == "hello.py"
        assert edit_ops[0].tool_id == "call_002"

    def test_extract_operations_delete_file(self, plugin, sample_jsonl_with_delete):
        """Should extract Delete File operations from apply_patch."""
        transcript = plugin.agentgit_parse_transcript(
            sample_jsonl_with_delete, "codex_jsonl"
        )
        operations = plugin.agentgit_extract_operations(transcript)

        assert len(operations) == 1
        assert operations[0].operation_type == OperationType.DELETE
        assert operations[0].file_path == "old_file.py"
        assert operations[0].tool_id == "call_del"

    def test_extract_operations_wrong_format(self, plugin):
        """Should return empty list for wrong format."""
        transcript = Transcript(source_format="other_format")
        operations = plugin.agentgit_extract_operations(transcript)
        assert operations == []

    def test_extract_operations_shell_rm_command(self, plugin, tmp_path):
        """Should extract delete operations from rm shell commands."""
        content = [
            {"type": "thread.started", "thread_id": "test"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Delete files"}]},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "cmd_1",
                    "type": "command_execution",
                    "command": "rm -rf /tmp/old_dir",
                    "status": "completed",
                    "exit_code": 0,
                },
            },
            {"type": "turn.completed"},
        ]

        jsonl_path = tmp_path / "rollout-rm.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "codex_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        assert len(operations) == 1
        assert operations[0].operation_type == OperationType.DELETE
        assert operations[0].file_path == "/tmp/old_dir"
        assert operations[0].recursive is True

    def test_enrich_operation_with_prompt(self, plugin, sample_jsonl):
        """Should enrich operations with prompt context."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "codex_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        enriched = plugin.agentgit_enrich_operation(operations[0], transcript)

        assert enriched.prompt is not None
        assert enriched.prompt.text == "Create a hello world function"

    def test_enrich_operation_with_reasoning(self, plugin, sample_jsonl):
        """Should enrich operations with reasoning context."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "codex_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        enriched = plugin.agentgit_enrich_operation(operations[0], transcript)

        # Should have assistant context with reasoning
        assert enriched.assistant_context is not None
        assert "hello world" in enriched.assistant_context.thinking.lower()

    def test_enrich_operation_wrong_format(self, plugin):
        """Should return operation unchanged for wrong format."""
        from agentgit.core import FileOperation

        op = FileOperation(
            file_path="test.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T10:00:00Z",
        )
        transcript = Transcript(source_format="other_format")

        result = plugin.agentgit_enrich_operation(op, transcript)
        assert result.prompt is None


class TestCodexPatchParsing:
    """Tests for parsing apply_patch content."""

    def test_parse_add_file_patch(self, plugin):
        """Should parse Add File patches correctly."""
        patch = """*** Begin Patch
*** Add File: src/utils.py
def utility():
    pass
*** End Patch"""

        ops = plugin._parse_patch(patch, "call_123", "2025-01-01T10:00:00Z")

        assert len(ops) == 1
        assert ops[0].operation_type == OperationType.WRITE
        assert ops[0].file_path == "src/utils.py"
        assert "def utility():" in ops[0].content

    def test_parse_update_file_patch(self, plugin):
        """Should parse Update File patches correctly."""
        patch = """*** Begin Patch
*** Update File: src/utils.py
@@ -1,2 +1,3 @@
 def utility():
-    pass
+    return True
*** End Patch"""

        ops = plugin._parse_patch(patch, "call_123", "2025-01-01T10:00:00Z")

        assert len(ops) == 1
        assert ops[0].operation_type == OperationType.EDIT
        assert ops[0].file_path == "src/utils.py"

    def test_parse_delete_file_patch(self, plugin):
        """Should parse Delete File patches correctly."""
        patch = """*** Begin Patch
*** Delete File: old_module.py
*** End Patch"""

        ops = plugin._parse_patch(patch, "call_123", "2025-01-01T10:00:00Z")

        assert len(ops) == 1
        assert ops[0].operation_type == OperationType.DELETE
        assert ops[0].file_path == "old_module.py"

    def test_parse_multiple_operations_patch(self, plugin):
        """Should parse patches with multiple file operations."""
        patch = """*** Begin Patch
*** Add File: new_file.py
print("new")
*** Update File: existing.py
@@ -1 +1 @@
-old
+new
*** Delete File: deprecated.py
*** End Patch"""

        ops = plugin._parse_patch(patch, "call_123", "2025-01-01T10:00:00Z")

        assert len(ops) == 3
        assert ops[0].operation_type == OperationType.WRITE
        assert ops[0].file_path == "new_file.py"
        assert ops[1].operation_type == OperationType.EDIT
        assert ops[1].file_path == "existing.py"
        assert ops[2].operation_type == OperationType.DELETE
        assert ops[2].file_path == "deprecated.py"


class TestCodexProjectName:
    """Tests for Codex project name decoding."""

    def test_get_project_name_returns_none_for_non_codex_transcript(
        self, plugin, tmp_path
    ):
        """Should return None for transcripts not in ~/.codex/sessions/."""
        transcript = tmp_path / "rollout.jsonl"
        transcript.write_text('{"type": "thread.started"}\n')

        result = plugin.agentgit_get_project_name(transcript)
        assert result is None

    def test_get_project_name_from_session_meta(self, plugin, tmp_path, monkeypatch):
        """Should extract project name from session_meta cwd."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "15"
        sessions_dir.mkdir(parents=True)

        # Create session with session_meta containing cwd
        content = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "test-session",
                    "cwd": "/home/user/myproject",
                },
            },
            {"type": "thread.started", "thread_id": "test"},
        ]

        transcript = sessions_dir / "rollout-2025-01-15T10-30-00-abc123.jsonl"
        with open(transcript, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_project_name(transcript)
        assert result == "myproject"

    def test_get_project_name_returns_none_without_session_meta(
        self, plugin, tmp_path, monkeypatch
    ):
        """Should return None for sessions without session_meta cwd."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "15"
        sessions_dir.mkdir(parents=True)

        transcript = sessions_dir / "rollout-2025-01-15T10-30-00-abc123.jsonl"
        transcript.write_text('{"type": "thread.started"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_project_name(transcript)
        assert result is None


class TestCodexDiscovery:
    """Tests for Codex transcript discovery."""

    def test_discover_returns_empty_if_no_codex_dir(self, plugin, tmp_path, monkeypatch):
        """Should return empty if ~/.codex doesn't exist."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(tmp_path / "some-project")
        assert result == []

    def test_discover_returns_empty_if_no_sessions(self, plugin, tmp_path, monkeypatch):
        """Should return empty if no sessions directory."""
        (tmp_path / ".codex").mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(tmp_path / "some-project")
        assert result == []

    def test_discover_finds_transcripts(self, plugin, tmp_path, monkeypatch):
        """Should find all transcript files when project_path is None."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "15"
        sessions_dir.mkdir(parents=True)

        # Create rollout files
        (sessions_dir / "rollout-2025-01-15T10-00-00-abc.jsonl").write_text(
            '{"type": "thread.started"}\n'
        )
        (sessions_dir / "rollout-2025-01-15T11-00-00-def.jsonl").write_text(
            '{"type": "thread.started"}\n'
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # With project_path=None, should find all transcripts
        result = plugin.agentgit_discover_transcripts(project_path=None)
        assert len(result) == 2
        assert all(p.suffix == ".jsonl" for p in result)
        assert all("rollout-" in p.name for p in result)

    def test_discover_sorts_by_mtime(self, plugin, tmp_path, monkeypatch):
        """Should sort results by modification time, most recent first."""
        import time

        sessions_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "15"
        sessions_dir.mkdir(parents=True)

        old_file = sessions_dir / "rollout-old.jsonl"
        old_file.write_text('{"type": "thread.started"}\n')

        time.sleep(0.01)  # Ensure different mtime

        new_file = sessions_dir / "rollout-new.jsonl"
        new_file.write_text('{"type": "thread.started"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # With project_path=None, should find all transcripts
        result = plugin.agentgit_discover_transcripts(project_path=None)
        assert len(result) == 2
        assert result[0].name == "rollout-new.jsonl"  # Most recent first
        assert result[1].name == "rollout-old.jsonl"

    def test_discover_filters_by_project(self, plugin, tmp_path, monkeypatch):
        """Should filter transcripts by project path based on cwd in session_meta."""
        import json

        sessions_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "15"
        sessions_dir.mkdir(parents=True)

        # Create a project with .git directory
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        # Create rollout file with matching cwd
        matching_transcript = sessions_dir / "rollout-matching.jsonl"
        matching_transcript.write_text(
            json.dumps({"type": "session_meta", "payload": {"cwd": str(project_dir)}})
            + "\n"
        )

        # Create rollout file with different cwd
        other_transcript = sessions_dir / "rollout-other.jsonl"
        other_transcript.write_text(
            json.dumps({"type": "session_meta", "payload": {"cwd": "/other/project"}})
            + "\n"
        )

        # Create rollout file with no cwd (should not match)
        no_cwd_transcript = sessions_dir / "rollout-nocwd.jsonl"
        no_cwd_transcript.write_text('{"type": "thread.started"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # When filtering by project, should only find the matching transcript
        result = plugin.agentgit_discover_transcripts(project_path=project_dir)
        assert len(result) == 1
        assert result[0].name == "rollout-matching.jsonl"


class TestCodexEnvironmentContext:
    """Tests for extracting cwd from environment context."""

    def test_extract_cwd_from_environment_context(self, plugin, tmp_path):
        """Should extract cwd from environment_context in user messages."""
        content = [
            {"type": "thread.started", "thread_id": "test"},
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<environment_context>\n<cwd>/home/user/myproject</cwd>\n</environment_context>",
                    },
                    {"type": "input_text", "text": "Hello"},
                ],
            },
        ]

        jsonl_path = tmp_path / "rollout-env.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "codex_jsonl")

        assert transcript is not None
        assert transcript.session_cwd == "/home/user/myproject"
        # The environment_context text should not be included as a prompt
        assert len(transcript.prompts) == 1
        assert transcript.prompts[0].text == "Hello"
