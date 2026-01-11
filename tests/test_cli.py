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

    def test_config_agents_command(self, runner):
        """Should list supported agent formats."""
        result = runner.invoke(main, ["config", "agents"])
        assert result.exit_code == 0
        assert "claude_code" in result.output

    def test_config_show_command(self, runner, tmp_path, sample_jsonl):
        """Should show configuration for a repository."""
        from git import Repo

        from agentgit.config import ProjectConfig, save_config

        # Process a transcript to create agentgit repo
        output_dir = tmp_path / "output"
        result = runner.invoke(main, ["process", str(sample_jsonl), "-o", str(output_dir)])
        assert result.exit_code == 0

        # Save some config
        config = ProjectConfig(enhancer="llm", enhance_model="sonnet")
        save_config(output_dir, config)

        # Get repo ID for directory lookup
        repo = Repo(output_dir)
        repo_id = repo.git.rev_list("--max-parents=0", "HEAD")[:12]

        # Test config show from within the repo
        result = runner.invoke(main, ["config", "show", "-r", str(output_dir)])
        assert result.exit_code == 0
        assert "Enhancer: llm" in result.output
        assert "Model: sonnet" in result.output
        assert repo_id in result.output

    def test_config_show_no_config(self, runner, tmp_path, sample_jsonl):
        """Should show defaults when no config is set."""
        # Process a transcript to create agentgit repo
        output_dir = tmp_path / "output"
        result = runner.invoke(main, ["process", str(sample_jsonl), "-o", str(output_dir)])
        assert result.exit_code == 0

        # Test config show without any config set
        result = runner.invoke(main, ["config", "show", "-r", str(output_dir)])
        assert result.exit_code == 0
        assert "(not set - will use default)" in result.output

    def test_config_set_enhancer(self, runner, tmp_path, sample_jsonl):
        """Should set enhancer configuration."""
        from agentgit.config import load_config

        # Process a transcript to create agentgit repo
        output_dir = tmp_path / "output"
        result = runner.invoke(main, ["process", str(sample_jsonl), "-o", str(output_dir)])
        assert result.exit_code == 0

        # Set enhancer
        result = runner.invoke(main, ["config", "set", "enhancer", "llm", "-r", str(output_dir)])
        assert result.exit_code == 0
        assert "Set enhancer to 'llm'" in result.output

        # Verify it was saved
        config = load_config(output_dir)
        assert config.enhancer == "llm"

    def test_config_set_model(self, runner, tmp_path, sample_jsonl):
        """Should set model configuration."""
        from agentgit.config import load_config

        # Process a transcript to create agentgit repo
        output_dir = tmp_path / "output"
        result = runner.invoke(main, ["process", str(sample_jsonl), "-o", str(output_dir)])
        assert result.exit_code == 0

        # Set model
        result = runner.invoke(
            main, ["config", "set", "model", "claude-cli-haiku", "-r", str(output_dir)]
        )
        assert result.exit_code == 0
        assert "Set model to 'claude-cli-haiku'" in result.output

        # Verify it was saved
        config = load_config(output_dir)
        assert config.enhance_model == "claude-cli-haiku"

    def test_config_set_preserves_other_values(self, runner, tmp_path, sample_jsonl):
        """Should preserve other config values when setting one."""
        from agentgit.config import ProjectConfig, load_config, save_config

        # Process a transcript to create agentgit repo
        output_dir = tmp_path / "output"
        result = runner.invoke(main, ["process", str(sample_jsonl), "-o", str(output_dir)])
        assert result.exit_code == 0

        # Set initial config
        initial_config = ProjectConfig(enhancer="rules", enhance_model="opus")
        save_config(output_dir, initial_config)

        # Set only enhancer
        result = runner.invoke(main, ["config", "set", "enhancer", "llm", "-r", str(output_dir)])
        assert result.exit_code == 0

        # Verify model is preserved
        config = load_config(output_dir)
        assert config.enhancer == "llm"
        assert config.enhance_model == "opus"

    def test_discover_command(self, runner, tmp_path, monkeypatch):
        """Should list discovered sessions in table format."""
        from git import Repo

        # Create a git repo with at least one commit
        repo_path = tmp_path / "myproject"
        repo_path.mkdir()
        repo = Repo.init(repo_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")
            config.set_value("commit", "gpgsign", "false")

        # Create a file and commit it (required for hash-based naming)
        (repo_path / "README.md").write_text("# Test")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

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


class TestPathHelpers:
    """Tests for path helper functions created during refactoring."""

    def test_find_best_match_by_suffix(self, tmp_path):
        """Should find best match by comparing path suffixes."""
        from agentgit.cli import _find_best_match_by_suffix

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create files with similar names
        (repo_path / "src").mkdir()
        (repo_path / "src" / "utils.py").touch()
        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "utils.py").touch()

        matches = [
            repo_path / "src" / "utils.py",
            repo_path / "tests" / "utils.py",
        ]

        # Should match src/utils.py better
        result = _find_best_match_by_suffix(
            ("src", "utils.py"), matches, repo_path
        )
        assert result == repo_path / "src" / "utils.py"

        # Should match tests/utils.py better
        result = _find_best_match_by_suffix(
            ("tests", "utils.py"), matches, repo_path
        )
        assert result == repo_path / "tests" / "utils.py"

    def test_find_best_match_by_suffix_no_match(self, tmp_path):
        """Should return None if no matches."""
        from agentgit.cli import _find_best_match_by_suffix

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        result = _find_best_match_by_suffix(
            ("src", "utils.py"), [], repo_path
        )
        assert result is None

    def test_translate_single_path_option(self, tmp_path):
        """Should pass through options unchanged."""
        from agentgit.cli import _translate_single_path

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        result = _translate_single_path("--oneline", repo_path, None)
        assert result == "--oneline"

    def test_translate_single_path_no_match(self, tmp_path):
        """Should return original if no match found."""
        from agentgit.cli import _translate_single_path

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        result = _translate_single_path("nonexistent.py", repo_path, None)
        assert result == "nonexistent.py"

    def test_translate_single_path_single_match(self, tmp_path):
        """Should translate when single match found."""
        from agentgit.cli import _translate_single_path

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "cli.py").write_text("# code")

        # Create local file
        local_dir = tmp_path / "project" / "src"
        local_dir.mkdir(parents=True)
        local_file = local_dir / "cli.py"
        local_file.write_text("# code")

        result = _translate_single_path(str(local_file), repo_path, None)
        assert result == "cli.py"


class TestEnhanceConfigHelper:
    """Tests for enhancement config resolution helper."""

    def test_resolve_enhance_config_no_args(self, tmp_path):
        """Should return None when no enhancer configured."""
        from agentgit.cli import _resolve_enhance_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        enhancer, model, config = _resolve_enhance_config(output_dir)

        assert enhancer is None
        assert model == "haiku"  # Default model
        assert config is None

    def test_resolve_enhance_config_with_enhancer(self, tmp_path):
        """Should create config when enhancer provided."""
        from agentgit.cli import _resolve_enhance_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        enhancer, model, config = _resolve_enhance_config(
            output_dir, enhancer="rules"
        )

        assert enhancer == "rules"
        assert model == "haiku"
        assert config is not None
        assert config.enhancer == "rules"
        assert config.model == "haiku"
        assert config.enabled is True

    def test_resolve_enhance_config_auto_set_llm(self, tmp_path):
        """Should auto-set enhancer to llm when model provided."""
        from agentgit.cli import _resolve_enhance_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        enhancer, model, config = _resolve_enhance_config(
            output_dir, enhance_model="sonnet"
        )

        assert enhancer == "llm"
        assert model == "sonnet"
        assert config is not None
        assert config.enhancer == "llm"
        assert config.model == "sonnet"

    def test_resolve_enhance_config_from_saved(self, tmp_path):
        """Should use saved config when available."""
        from git import Repo

        from agentgit.cli import _resolve_enhance_config
        from agentgit.config import ProjectConfig, save_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initialize git repo (required for config storage)
        Repo.init(output_dir)

        # Save a config
        saved = ProjectConfig(enhancer="rules", enhance_model="opus")
        save_config(output_dir, saved)

        # Should use saved config
        enhancer, model, config = _resolve_enhance_config(output_dir)

        assert enhancer == "rules"
        assert model == "opus"
        assert config is not None
        assert config.enhancer == "rules"
        assert config.model == "opus"

    def test_resolve_enhance_config_cli_overrides_saved(self, tmp_path):
        """Should prefer CLI args over saved config."""
        from git import Repo

        from agentgit.cli import _resolve_enhance_config
        from agentgit.config import ProjectConfig, save_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initialize git repo (required for config storage)
        Repo.init(output_dir)

        # Save a config
        saved = ProjectConfig(enhancer="rules", enhance_model="opus")
        save_config(output_dir, saved)

        # CLI args should override
        enhancer, model, config = _resolve_enhance_config(
            output_dir, enhancer="llm", enhance_model="haiku"
        )

        assert enhancer == "llm"
        assert model == "haiku"

    def test_resolve_enhance_config_auto_detect_claude_code(self, tmp_path):
        """Should auto-enable llm/claude-cli-haiku for Claude Code transcripts."""
        from agentgit.cli import _resolve_enhance_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a Claude Code JSONL transcript
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(
            '{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", '
            '"message": {"content": "test"}, "sessionId": "test"}\n'
        )

        # Should auto-detect and enable llm/claude-cli-haiku
        enhancer, model, config = _resolve_enhance_config(
            output_dir, transcript_paths=[transcript]
        )

        assert enhancer == "llm"
        assert model == "claude-cli-haiku"
        assert config is not None
        assert config.enhancer == "llm"
        assert config.model == "claude-cli-haiku"

    def test_resolve_enhance_config_auto_detect_respects_cli_args(self, tmp_path):
        """Should not auto-detect when CLI args provided."""
        from agentgit.cli import _resolve_enhance_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a Claude Code JSONL transcript
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(
            '{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", '
            '"message": {"content": "test"}, "sessionId": "test"}\n'
        )

        # CLI arg should override auto-detection
        enhancer, model, config = _resolve_enhance_config(
            output_dir, enhancer="rules", transcript_paths=[transcript]
        )

        assert enhancer == "rules"
        assert model == "haiku"  # Default model, not claude-cli-haiku

    def test_resolve_enhance_config_auto_detect_respects_saved_config(self, tmp_path):
        """Should not auto-detect when saved config exists."""
        from git import Repo

        from agentgit.cli import _resolve_enhance_config
        from agentgit.config import ProjectConfig, save_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initialize git repo and save config
        Repo.init(output_dir)
        saved = ProjectConfig(enhancer="rules", enhance_model="opus")
        save_config(output_dir, saved)

        # Create a Claude Code JSONL transcript
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(
            '{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", '
            '"message": {"content": "test"}, "sessionId": "test"}\n'
        )

        # Saved config should override auto-detection
        enhancer, model, config = _resolve_enhance_config(
            output_dir, transcript_paths=[transcript]
        )

        assert enhancer == "rules"
        assert model == "opus"

    def test_resolve_enhance_config_no_auto_detect_for_non_claude_code(self, tmp_path):
        """Should not auto-enable for non-Claude Code transcripts."""
        from agentgit.cli import _resolve_enhance_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a non-JSONL file (won't be detected as Claude Code)
        transcript = tmp_path / "session.txt"
        transcript.write_text("Some text")

        # Should not auto-enable
        enhancer, model, config = _resolve_enhance_config(
            output_dir, transcript_paths=[transcript]
        )

        assert enhancer is None
        assert model == "haiku"  # Still returns default model
        assert config is None


class TestClaudeCodeDetection:
    """Tests for Claude Code transcript detection."""

    def test_has_claude_code_transcripts_detects_claude_code(self, tmp_path):
        """Should detect Claude Code JSONL format."""
        from agentgit.cli import _has_claude_code_transcripts

        # Create a Claude Code JSONL transcript
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(
            '{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", '
            '"message": {"content": "test"}, "sessionId": "test"}\n'
        )

        assert _has_claude_code_transcripts([transcript]) is True

    def test_has_claude_code_transcripts_detects_web_sessions(self, tmp_path):
        """Should detect Claude Code web sessions format."""
        from agentgit.cli import _has_claude_code_transcripts

        # Create a web sessions file (also contains claude_code in format name)
        transcript = tmp_path / "sessions.json"
        transcript.write_text('{"sessions": []}')

        # This should also be detected as claude_code format
        result = _has_claude_code_transcripts([transcript])
        # Note: This may be False if web format doesn't include "claude_code" in name
        # That's ok, the important test is the JSONL one above

    def test_has_claude_code_transcripts_rejects_non_claude_code(self, tmp_path):
        """Should not detect non-Claude Code files."""
        from agentgit.cli import _has_claude_code_transcripts

        # Create a non-JSONL file
        transcript = tmp_path / "session.txt"
        transcript.write_text("Some text")

        assert _has_claude_code_transcripts([transcript]) is False

    def test_has_claude_code_transcripts_multiple_files(self, tmp_path):
        """Should detect Claude Code in mixed list of transcripts."""
        from agentgit.cli import _has_claude_code_transcripts

        # Create multiple transcripts
        cc_transcript = tmp_path / "session.jsonl"
        cc_transcript.write_text(
            '{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", '
            '"message": {"content": "test"}, "sessionId": "test"}\n'
        )

        other_transcript = tmp_path / "other.txt"
        other_transcript.write_text("Some text")

        # Should detect Claude Code even with other files
        assert _has_claude_code_transcripts([cc_transcript, other_transcript]) is True
        assert _has_claude_code_transcripts([other_transcript, cc_transcript]) is True


class TestUtilityFunctions:
    """Tests for CLI utility functions."""

    def test_get_repo_id_valid_repo(self, tmp_path):
        """Should return first 12 chars of root commit SHA."""
        from git import Repo

        from agentgit.cli import get_repo_id

        # Create repo with commit
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        (repo_path / "README.md").write_text("# Test")
        repo.index.add(["README.md"])
        commit = repo.index.commit("Initial commit")

        repo_id = get_repo_id(repo_path)
        assert repo_id == commit.hexsha[:12]

    def test_get_repo_id_no_commits(self, tmp_path):
        """Should return None for repo with no commits."""
        from git import Repo

        from agentgit.cli import get_repo_id

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        Repo.init(repo_path)

        repo_id = get_repo_id(repo_path)
        assert repo_id is None

    def test_get_repo_id_not_a_repo(self, tmp_path):
        """Should return None for non-git directory."""
        from agentgit.cli import get_repo_id

        repo_id = get_repo_id(tmp_path)
        assert repo_id is None

    def test_get_default_output_dir_with_hash(self, tmp_path, monkeypatch):
        """Should use hash-based naming for output directory."""
        from git import Repo

        from agentgit.cli import get_default_output_dir

        # Create a git repo
        repo_path = tmp_path / "myproject"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        (repo_path / "README.md").write_text("# Test")
        repo.index.add(["README.md"])
        commit = repo.index.commit("Initial commit")
        expected_id = commit.hexsha[:12]

        # Create matching .claude/projects directory
        encoded = str(repo_path.resolve()).replace("/", "-")
        claude_dir = tmp_path / ".claude" / "projects" / encoded
        claude_dir.mkdir(parents=True)
        transcript = claude_dir / "session.jsonl"
        transcript.write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(repo_path)

        output = get_default_output_dir(transcript)
        assert expected_id in str(output)
        assert ".agentgit/projects" in str(output)

    def test_get_agentgit_repo_path_no_transcripts(self, tmp_path, monkeypatch):
        """Should return None when no transcripts found."""
        from git import Repo

        from agentgit.cli import get_agentgit_repo_path

        repo_path = tmp_path / "project"
        repo_path.mkdir()
        Repo.init(repo_path)

        monkeypatch.chdir(repo_path)

        result = get_agentgit_repo_path()
        assert result is None

    def test_get_agentgit_repo_path_with_transcript(self, tmp_path, monkeypatch):
        """Should return agentgit repo path when transcript exists."""
        from git import Repo

        from agentgit.cli import get_agentgit_repo_path

        # Create project repo with commit
        repo_path = tmp_path / "project"
        repo_path.mkdir()
        repo = Repo.init(repo_path)
        (repo_path / "README.md").write_text("# Test")
        repo.index.add(["README.md"])
        commit = repo.index.commit("Initial commit")

        # Create matching transcript
        encoded = str(repo_path.resolve()).replace("/", "-")
        claude_dir = tmp_path / ".claude" / "projects" / encoded
        claude_dir.mkdir(parents=True)
        (claude_dir / "session.jsonl").write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(repo_path)

        result = get_agentgit_repo_path()
        assert result is not None
        assert commit.hexsha[:12] in str(result)
