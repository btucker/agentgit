"""Tests for agentgit.cli module."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentgit.cli import main


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file."""
    content = [
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Create a hello world function"},
            "sessionId": "test-session",
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
                            "file_path": "/test/hello.py",
                            "content": "def hello():\n    return 'Hello!'",
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


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self, runner):
        """Should show help message."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Process agent transcripts" in result.output

    def test_version(self, runner):
        """Should show version."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_process_command(self, runner, sample_jsonl, tmp_path):
        """Should process transcript to git repo."""
        output_dir = tmp_path / "output"
        result = runner.invoke(main, ["process", str(sample_jsonl), "-o", str(output_dir)])

        assert result.exit_code == 0
        assert "Created git repository" in result.output
        assert (output_dir / ".git").exists()

    def test_process_creates_default_output(self, runner, sample_jsonl, tmp_path, monkeypatch):
        """Should use ~/.agentgit/projects/ if no output specified."""
        # Mock home directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = runner.invoke(main, ["process", str(sample_jsonl)])
        assert result.exit_code == 0
        assert "Created git repository" in result.output
        # Should be in ~/.agentgit/projects/
        assert ".agentgit" in result.output

    def test_process_with_author(self, runner, sample_jsonl, tmp_path):
        """Should use custom author."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                "process",
                str(sample_jsonl),
                "-o",
                str(output_dir),
                "--author",
                "Custom Author",
                "--email",
                "custom@example.com",
            ],
        )
        assert result.exit_code == 0

    def test_process_nonexistent_file(self, runner):
        """Should error on nonexistent file."""
        result = runner.invoke(main, ["process", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0

    def test_default_command(self, runner, sample_jsonl, tmp_path):
        """Should use process as default command."""
        output_dir = tmp_path / "output"
        # Run without specifying "process" - should work as default
        result = runner.invoke(main, [str(sample_jsonl), "-o", str(output_dir)])

        assert result.exit_code == 0
        assert "Created git repository" in result.output

    def test_agents_command(self, runner):
        """Should list supported agent formats."""
        result = runner.invoke(main, ["agents"])
        assert result.exit_code == 0
        assert "claude_code" in result.output

    def test_discover_command(self, runner, tmp_path, monkeypatch):
        """Should list discovered sessions in table format."""
        from git import Repo

        # Create a git repo
        repo_path = tmp_path / "myproject"
        repo_path.mkdir()
        Repo.init(repo_path)

        # Create .claude/projects with matching directory
        encoded_path = str(repo_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create transcript file
        (claude_project_dir / "session.jsonl").write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(repo_path)

        # Use --list flag to avoid interactive selection (discover is an alias for sessions)
        result = runner.invoke(main, ["sessions", "--list"])
        assert result.exit_code == 0
        assert "1 session" in result.output
        # Agent column shows Claude Code
        assert "Claude Code" in result.output

    def test_discover_no_transcripts(self, runner, tmp_path, monkeypatch):
        """Should show message when no transcripts found."""
        from git import Repo

        repo_path = tmp_path / "myproject"
        repo_path.mkdir()
        Repo.init(repo_path)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(repo_path)

        result = runner.invoke(main, ["discover", "--list"])
        assert result.exit_code == 0
        assert "No transcripts found" in result.output


class TestPathTranslation:
    """Tests for path translation in git passthrough."""

    def test_translate_paths_single_match(self, tmp_path):
        """Should translate path when single match found."""
        from agentgit.cli import translate_paths_for_agentgit_repo

        # Create a file in the "agentgit repo"
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "cli.py").write_text("# code")

        # Create a file in the "local project"
        local_dir = tmp_path / "project" / "src" / "agentgit"
        local_dir.mkdir(parents=True)
        local_file = local_dir / "cli.py"
        local_file.write_text("# code")

        args = ["log", str(local_file)]
        result = translate_paths_for_agentgit_repo(args, repo_path)

        assert result == ["log", "cli.py"]

    def test_translate_paths_no_match(self, tmp_path):
        """Should keep original path when no match found."""
        from agentgit.cli import translate_paths_for_agentgit_repo

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Local file that doesn't exist in repo
        local_file = tmp_path / "nonexistent.py"
        local_file.write_text("# code")

        args = ["log", str(local_file)]
        result = translate_paths_for_agentgit_repo(args, repo_path)

        assert result == ["log", str(local_file)]

    def test_translate_paths_preserves_options(self, tmp_path):
        """Should preserve options unchanged."""
        from agentgit.cli import translate_paths_for_agentgit_repo

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        args = ["log", "--oneline", "-10"]
        result = translate_paths_for_agentgit_repo(args, repo_path)

        assert result == ["log", "--oneline", "-10"]

    def test_translate_paths_multiple_matches_uses_best(self, tmp_path):
        """Should use best match when multiple files have same name."""
        from agentgit.cli import translate_paths_for_agentgit_repo

        # Create repo with two files named cli.py
        repo_path = tmp_path / "repo"
        (repo_path / "agentgit").mkdir(parents=True)
        (repo_path / "other").mkdir(parents=True)
        (repo_path / "agentgit" / "cli.py").write_text("# code")
        (repo_path / "other" / "cli.py").write_text("# code")

        # Create local file with matching path structure
        local_dir = tmp_path / "project" / "src" / "agentgit"
        local_dir.mkdir(parents=True)
        local_file = local_dir / "cli.py"
        local_file.write_text("# code")

        args = ["log", str(local_file)]
        result = translate_paths_for_agentgit_repo(args, repo_path)

        # Should match agentgit/cli.py because path suffix matches better
        assert result == ["log", "agentgit/cli.py"]


class TestWebCommand:
    """Tests for the web command."""

    def test_web_help(self, runner):
        """Should show help for web command."""
        result = runner.invoke(main, ["web", "--help"])
        assert result.exit_code == 0
        assert "Claude Code web session" in result.output
        assert "--token" in result.output
        assert "--org-uuid" in result.output

    def test_web_requires_credentials(self, runner, tmp_path, monkeypatch):
        """Should error when credentials not available."""
        # Mock non-macOS platform and no config file
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = runner.invoke(main, ["web"])
        assert result.exit_code != 0
        assert "access token" in result.output.lower()

    def test_web_no_sessions(self, runner, monkeypatch):
        """Should error when no sessions found."""
        from unittest.mock import patch

        with patch(
            "agentgit.web_sessions.resolve_credentials", return_value=("token", "org")
        ), patch("agentgit.web_sessions.fetch_sessions", return_value=[]):
            result = runner.invoke(main, ["web"])
            assert result.exit_code != 0
            assert "No web sessions found" in result.output

    def test_web_with_session_id(self, runner, tmp_path, monkeypatch):
        """Should process specific session when ID provided."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        mock_session_data = {
            "id": "test-session-123",
            "project_path": str(tmp_path / "project"),
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00Z",
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05Z",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Creating file..."},
                            {
                                "type": "tool_use",
                                "id": "toolu_001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/test/hello.py",
                                    "content": "print('hello')",
                                },
                            },
                        ]
                    },
                },
            ],
        }

        with patch(
            "agentgit.web_sessions.resolve_credentials", return_value=("token", "org")
        ), patch(
            "agentgit.web_sessions.fetch_session_data", return_value=mock_session_data
        ):
            output_dir = tmp_path / "output"
            result = runner.invoke(
                main, ["web", "test-session-123", "-o", str(output_dir)]
            )

            assert result.exit_code == 0
            assert "Created git repository" in result.output

    def test_web_session_picker_display(self, runner, monkeypatch):
        """Should display sessions for interactive selection."""
        from unittest.mock import patch

        from agentgit.web_sessions import WebSession

        sessions = [
            WebSession(
                id="session-1",
                title="First Session",
                created_at="2025-01-01T00:00:00Z",
                project_path="/path/to/project",
            ),
            WebSession(
                id="session-2",
                title="Second Session",
                created_at="2025-01-02T00:00:00Z",
            ),
        ]

        with patch(
            "agentgit.web_sessions.resolve_credentials", return_value=("token", "org")
        ), patch("agentgit.web_sessions.fetch_sessions", return_value=sessions):
            # Abort the prompt to exit cleanly
            result = runner.invoke(main, ["web"], input="\x03")  # Ctrl+C

            assert "Found 2 session(s)" in result.output
            assert "First Session" in result.output
            assert "session-1" in result.output
            assert "/path/to/project" in result.output
