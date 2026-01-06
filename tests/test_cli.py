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

    def test_discover_command(self, runner, tmp_path, monkeypatch):
        """Should list discovered sessions in table format."""
        from git import Repo

        # Create a git repo with at least one commit
        repo_path = tmp_path / "myproject"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

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


class TestBlameHelpers:
    """Tests for blame helper functions."""

    def test_hash_line_basic(self):
        """Should hash line content consistently."""
        from agentgit.cmd.blame import hash_line as _hash_line

        line1 = "def foo():\n"
        line2 = "def foo():\n"
        line3 = "def bar():\n"

        # Same content should produce same hash
        assert _hash_line(line1) == _hash_line(line2)

        # Different content should produce different hash
        assert _hash_line(line1) != _hash_line(line3)

    def test_hash_line_strips_trailing_newline(self):
        """Should treat lines with/without trailing newline as identical."""
        from agentgit.cmd.blame import hash_line as _hash_line

        line_with_newline = "def foo():\n"
        line_without_newline = "def foo():"

        # Should produce same hash (newline is stripped)
        assert _hash_line(line_with_newline) == _hash_line(line_without_newline)

    def test_hash_line_preserves_leading_whitespace(self):
        """Should preserve leading whitespace in hash."""
        from agentgit.cmd.blame import hash_line as _hash_line

        line1 = "def foo():"
        line2 = "    def foo():"

        # Leading whitespace matters
        assert _hash_line(line1) != _hash_line(line2)

    def test_hash_line_returns_16_chars(self):
        """Should return 16 character hash."""
        from agentgit.cmd.blame import hash_line as _hash_line

        line = "def foo():"
        hash_val = _hash_line(line)

        assert len(hash_val) == 16
        assert all(c in '0123456789abcdef' for c in hash_val)

    def test_normalize_session_path_documents_prefix(self):
        """Should normalize Documents/projects/agentgit/ prefix."""
        from agentgit.cmd.blame import normalize_session_path as _normalize_session_path

        path = "Documents/projects/agentgit/src/agentgit/cli.py"
        expected = "src/agentgit/cli.py"

        assert _normalize_session_path(path) == expected

    def test_normalize_session_path_users_prefix(self):
        """Should normalize Users/... prefix."""
        from agentgit.cmd.blame import normalize_session_path as _normalize_session_path

        path = "Users/btucker/Documents/projects/agentgit/src/agentgit/cli.py"
        expected = "src/agentgit/cli.py"

        assert _normalize_session_path(path) == expected

    def test_normalize_session_path_no_match(self):
        """Should return path unchanged if no prefix matches."""
        from agentgit.cmd.blame import normalize_session_path as _normalize_session_path

        path = "some/other/path/file.py"

        assert _normalize_session_path(path) == path

    def test_normalize_session_path_empty(self):
        """Should handle empty path."""
        from agentgit.cmd.blame import normalize_session_path as _normalize_session_path

        assert _normalize_session_path("") == ""

    def test_build_line_grep_index(self, tmp_path):
        """Should build searchable index of lines from session branches."""
        from git import Repo

        from agentgit.cmd.blame import (
            build_line_grep_index as _build_line_grep_index,
            hash_line as _hash_line,
        )

        # Create an agentgit repo with a session branch
        repo_path = tmp_path / "agentgit_repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        # Create initial commit on main
        (repo_path / "README.md").write_text("# Test\n")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Create a session branch with a file
        repo.git.checkout("-b", "session/claude_code/test-session")
        test_file = repo_path / "src" / "agentgit" / "cli.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def foo():\n    pass\n")
        repo.index.add([str(test_file.relative_to(repo_path))])
        commit = repo.index.commit(
            "Write src/agentgit/cli.py\n\n"
            "Context: Adding a simple function for testing\n\n"
            "Operation: Write\n"
            "Tool-Id: toolu_test123"
        )

        # Build the index
        _build_line_grep_index(repo, repo_path)

        # Verify the index branch exists
        assert "agentgit-index" in [ref.name.split("/")[-1] for ref in repo.refs]

        # Checkout the index branch to verify content
        repo.git.checkout("agentgit-index")
        index_file = repo_path / ".agentgit" / "lines" / "src_agentgit_cli.py"
        assert index_file.exists()

        content = index_file.read_text()
        lines = content.strip().split("\n")

        # Should have entries for both lines
        assert len(lines) == 2

        # Verify first line entry (sliding window: "" + "def foo():" + "    pass")
        line1_hash = _hash_line("def foo():\n", "", "    pass\n")
        assert any(line.startswith(f"{line1_hash}|") for line in lines)

        # Verify the entry contains session name and commit
        matching_line = [line for line in lines if line.startswith(f"{line1_hash}|")][0]
        parts = matching_line.split("|")
        assert parts[1] == "session/claude_code/test-session"
        assert parts[2] == commit.hexsha[:7]
        assert "Adding a simple function" in parts[3]

    def test_build_line_grep_index_multiple_sessions(self, tmp_path):
        """Should index lines from multiple session branches."""
        from git import Repo

        from agentgit.cmd.blame import (
            build_line_grep_index as _build_line_grep_index,
            hash_line as _hash_line,
        )

        repo_path = tmp_path / "agentgit_repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        # Initial commit
        (repo_path / "README.md").write_text("# Test\n")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # First session
        repo.git.checkout("-b", "session/claude_code/session-1")
        test_file = repo_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("# First session\n")
        repo.index.add([str(test_file.relative_to(repo_path))])
        commit1 = repo.index.commit(
            "Write src/test.py\n\n"
            "Context: First\n\n"
            "Operation: Write\n"
            "Tool-Id: toolu_session1"
        )

        # Second session
        repo.git.checkout("main")
        repo.git.checkout("-b", "session/claude_code/session-2")
        # Recreate the file and directory if needed
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# Second session\n")
        repo.index.add([str(test_file.relative_to(repo_path))])
        commit2 = repo.index.commit(
            "Write src/test.py\n\n"
            "Context: Second\n\n"
            "Operation: Write\n"
            "Tool-Id: toolu_session2"
        )

        # Build index
        _build_line_grep_index(repo, repo_path)

        # Verify both sessions are indexed
        repo.git.checkout("agentgit-index")
        index_file = repo_path / ".agentgit" / "lines" / "src_test.py"
        content = index_file.read_text()

        # Both files only have one line, so prev="" and next=""
        line1_hash = _hash_line("# First session\n", "", "")
        line2_hash = _hash_line("# Second session\n", "", "")

        assert f"{line1_hash}|session/claude_code/session-1" in content
        assert f"{line2_hash}|session/claude_code/session-2" in content

    def test_lookup_line_with_grep_found(self, tmp_path):
        """Should find line in grep index."""
        from git import Repo

        from agentgit.cmd.blame import (
            build_line_grep_index as _build_line_grep_index,
            hash_line as _hash_line,
            lookup_line_with_grep as _lookup_line_with_grep,
        )

        # Create agentgit repo with indexed session
        repo_path = tmp_path / "agentgit_repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        (repo_path / "README.md").write_text("# Test\n")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Create session with known line
        repo.git.checkout("-b", "session/claude_code/test-session")
        test_file = repo_path / "src" / "agentgit" / "cli.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def hello():\n    return 'world'\n")
        repo.index.add([str(test_file.relative_to(repo_path))])
        commit = repo.index.commit(
            "Write src/agentgit/cli.py\n\n"
            "Context: Simple greeting function\n\n"
            "Operation: Write\n"
            "Tool-Id: toolu_hello123"
        )

        # Build index
        _build_line_grep_index(repo, repo_path)

        # Look up the line (sliding window: "" + "def hello():" + "    return 'world'")
        line_hash = _hash_line("def hello():\n", "", "    return 'world'\n")
        result = _lookup_line_with_grep(repo, "src/agentgit/cli.py", line_hash)

        assert result is not None
        assert result["session"] == "session/claude_code/test-session"
        assert result["commit"] == commit.hexsha[:7]
        assert "Simple greeting function" in result["context"]

    def test_lookup_line_with_grep_not_found(self, tmp_path):
        """Should return None when line not in index."""
        from git import Repo

        from agentgit.cmd.blame import (
            build_line_grep_index as _build_line_grep_index,
            hash_line as _hash_line,
            lookup_line_with_grep as _lookup_line_with_grep,
        )

        repo_path = tmp_path / "agentgit_repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        (repo_path / "README.md").write_text("# Test\n")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Build empty index
        _build_line_grep_index(repo, repo_path)

        # Look up non-existent line
        line_hash = _hash_line("def nonexistent():")
        result = _lookup_line_with_grep(repo, "src/test.py", line_hash)

        assert result is None

    def test_lookup_line_with_grep_no_index(self, tmp_path):
        """Should return None when index doesn't exist."""
        from git import Repo

        from agentgit.cmd.blame import (
            hash_line as _hash_line,
            lookup_line_with_grep as _lookup_line_with_grep,
        )

        repo_path = tmp_path / "agentgit_repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        (repo_path / "README.md").write_text("# Test\n")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Look up without building index
        line_hash = _hash_line("def hello():")
        result = _lookup_line_with_grep(repo, "src/test.py", line_hash)

        assert result is None


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
        assert config.enhancer == "llm"
        assert config.model == "haiku"


class TestUtilityFunctions:
    """Tests for CLI utility functions."""

    def test_encode_path_as_name(self):
        """Should encode path by replacing slashes with dashes."""
        from pathlib import Path

        from agentgit.cli import encode_path_as_name

        path = Path("/Users/name/project")
        result = encode_path_as_name(path)
        assert result.startswith("-Users-name-project")

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
