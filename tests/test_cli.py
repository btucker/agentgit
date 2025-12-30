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

    def test_prompts_command(self, runner, sample_jsonl):
        """Should list prompts from transcript."""
        result = runner.invoke(main, ["prompts", str(sample_jsonl)])
        assert result.exit_code == 0
        assert "Found 1 prompts" in result.output
        assert "Create a hello world function" in result.output

    def test_operations_command(self, runner, sample_jsonl):
        """Should list operations from transcript."""
        result = runner.invoke(main, ["operations", str(sample_jsonl)])
        assert result.exit_code == 0
        assert "Found 1 operations" in result.output
        assert "WRITE" in result.output
        assert "/test/hello.py" in result.output

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

    def test_prompts_nonexistent_file(self, runner):
        """Should error on nonexistent file."""
        result = runner.invoke(main, ["prompts", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0

    def test_operations_nonexistent_file(self, runner):
        """Should error on nonexistent file."""
        result = runner.invoke(main, ["operations", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0

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

    def test_types_command(self, runner):
        """Should list available plugin types."""
        result = runner.invoke(main, ["types"])
        assert result.exit_code == 0
        assert "claude_code" in result.output

    def test_discover_command(self, runner, tmp_path, monkeypatch):
        """Should list discovered transcripts."""
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

        result = runner.invoke(main, ["discover"])
        assert result.exit_code == 0
        assert "Found 1 transcript" in result.output
        assert "session.jsonl" in result.output

    def test_discover_no_transcripts(self, runner, tmp_path, monkeypatch):
        """Should show message when no transcripts found."""
        from git import Repo

        repo_path = tmp_path / "myproject"
        repo_path.mkdir()
        Repo.init(repo_path)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(repo_path)

        result = runner.invoke(main, ["discover"])
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
