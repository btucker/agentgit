"""Tests for agentgit.url_resolver module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentgit.url_resolver import (
    URLResolverError,
    fetch_url,
    find_transcripts_in_repo,
    is_git_url,
    is_url,
    resolve_file_uri,
    resolve_transcript_source,
)


class TestIsUrl:
    """Tests for is_url function."""

    def test_http_url(self):
        """Should recognize http:// URLs."""
        assert is_url("http://example.com/file.jsonl")
        assert is_url("HTTP://EXAMPLE.COM/FILE.jsonl")

    def test_https_url(self):
        """Should recognize https:// URLs."""
        assert is_url("https://example.com/file.jsonl")
        assert is_url("https://raw.githubusercontent.com/user/repo/main/session.jsonl")

    def test_file_uri(self):
        """Should recognize file:// URIs."""
        assert is_url("file:///path/to/file.jsonl")
        assert is_url("file://localhost/path/to/file.jsonl")

    def test_git_ssh_url(self):
        """Should recognize git SSH URLs."""
        assert is_url("git@github.com:user/repo.git")
        assert is_url("ssh://git@github.com/user/repo.git")

    def test_git_protocol_url(self):
        """Should recognize git:// URLs."""
        assert is_url("git://github.com/user/repo.git")

    def test_https_git_url(self):
        """Should recognize HTTPS git URLs."""
        assert is_url("https://github.com/user/repo.git")

    def test_local_path_not_url(self):
        """Should not recognize local paths as URLs."""
        assert not is_url("./session.jsonl")
        assert not is_url("/path/to/session.jsonl")
        assert not is_url("session.jsonl")
        assert not is_url("../parent/session.jsonl")

    def test_relative_path_with_dots(self):
        """Should not recognize relative paths as URLs."""
        assert not is_url("path/to/file.jsonl")


class TestIsGitUrl:
    """Tests for is_git_url function."""

    def test_git_ssh_url(self):
        """Should recognize git SSH URLs."""
        assert is_git_url("git@github.com:user/repo.git")

    def test_ssh_protocol(self):
        """Should recognize ssh:// protocol."""
        assert is_git_url("ssh://git@github.com/user/repo.git")

    def test_git_protocol(self):
        """Should recognize git:// protocol."""
        assert is_git_url("git://github.com/user/repo.git")

    def test_https_with_git_suffix(self):
        """Should recognize HTTPS URLs with .git suffix."""
        assert is_git_url("https://github.com/user/repo.git")

    def test_github_repo_url(self):
        """Should recognize GitHub repo URLs without .git suffix."""
        assert is_git_url("https://github.com/user/repo")
        assert is_git_url("https://github.com/user/repo/")

    def test_gitlab_repo_url(self):
        """Should recognize GitLab repo URLs."""
        assert is_git_url("https://gitlab.com/user/repo")

    def test_bitbucket_repo_url(self):
        """Should recognize Bitbucket repo URLs."""
        assert is_git_url("https://bitbucket.org/user/repo")

    def test_raw_file_url_not_git(self):
        """Should not recognize raw file URLs as git repos."""
        assert not is_git_url("https://raw.githubusercontent.com/user/repo/main/file.jsonl")
        assert not is_git_url("https://github.com/user/repo/blob/main/file.jsonl")
        assert not is_git_url("https://github.com/user/repo/raw/main/file.jsonl")

    def test_gitlab_file_url_not_git(self):
        """Should not recognize GitLab file URLs as git repos."""
        assert not is_git_url("https://gitlab.com/user/repo/-/blob/main/file.jsonl")
        assert not is_git_url("https://gitlab.com/user/repo/-/raw/main/file.jsonl")

    def test_generic_https_not_git(self):
        """Should not recognize generic HTTPS URLs as git repos."""
        assert not is_git_url("https://example.com/file.jsonl")
        assert not is_git_url("https://api.example.com/transcripts/123")


class TestFetchUrl:
    """Tests for fetch_url function."""

    def test_successful_fetch(self):
        """Should fetch content from URL."""
        mock_response = MagicMock()
        mock_response.content = b'{"type": "user"}\n'

        with patch("agentgit.url_resolver.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            content = fetch_url("https://example.com/file.jsonl")
            assert content == b'{"type": "user"}\n'

    def test_timeout_error(self):
        """Should raise URLResolverError on timeout."""
        with patch("agentgit.url_resolver.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value = mock_client

            with pytest.raises(URLResolverError) as exc_info:
                fetch_url("https://example.com/file.jsonl")
            assert "Timeout" in str(exc_info.value)

    def test_http_error(self):
        """Should raise URLResolverError on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch("agentgit.url_resolver.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(URLResolverError) as exc_info:
                fetch_url("https://example.com/file.jsonl")
            assert "404" in str(exc_info.value)


class TestResolveFileUri:
    """Tests for resolve_file_uri function."""

    def test_absolute_path(self, tmp_path):
        """Should resolve file:// URI with absolute path."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"type": "user"}\n')

        path = resolve_file_uri(f"file://{test_file}")
        assert path == test_file

    def test_triple_slash(self, tmp_path):
        """Should resolve file:/// URI."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"type": "user"}\n')

        path = resolve_file_uri(f"file:///{str(test_file).lstrip('/')}")
        assert path.exists()

    def test_nonexistent_file(self):
        """Should raise URLResolverError for nonexistent file."""
        with pytest.raises(URLResolverError) as exc_info:
            resolve_file_uri("file:///nonexistent/path/file.jsonl")
        assert "not found" in str(exc_info.value).lower()


class TestResolveTranscriptSource:
    """Tests for resolve_transcript_source function."""

    def test_local_path(self, tmp_path):
        """Should resolve local path without fetching."""
        test_file = tmp_path / "session.jsonl"
        test_file.write_text('{"type": "user"}\n')

        result = resolve_transcript_source(str(test_file))

        assert result.path == test_file
        assert result.source_type == "local"
        assert not result.is_temporary
        assert result.original_url is None

    def test_local_path_not_found(self, tmp_path):
        """Should raise FileNotFoundError for nonexistent local path."""
        with pytest.raises(FileNotFoundError):
            resolve_transcript_source(str(tmp_path / "nonexistent.jsonl"))

    def test_http_url(self, tmp_path):
        """Should fetch and save HTTP URL to temp file."""
        content = b'{"type": "user"}\n{"type": "assistant"}\n'

        with patch("agentgit.url_resolver.fetch_url", return_value=content):
            result = resolve_transcript_source("https://example.com/session.jsonl")

            assert result.source_type == "url"
            assert result.is_temporary
            assert result.original_url == "https://example.com/session.jsonl"
            assert result.path.exists()
            assert result.path.read_bytes() == content

            # Clean up
            result.cleanup()
            assert not result.path.exists()

    def test_file_uri(self, tmp_path):
        """Should resolve file:// URI."""
        test_file = tmp_path / "session.jsonl"
        test_file.write_text('{"type": "user"}\n')

        result = resolve_transcript_source(f"file://{test_file}")

        assert result.path == test_file
        assert result.source_type == "local"
        assert not result.is_temporary

    def test_git_url(self, tmp_path):
        """Should clone git repository URL."""
        mock_repo_path = tmp_path / "cloned" / "repo"
        mock_repo_path.mkdir(parents=True)
        (mock_repo_path / "session.jsonl").write_text('{"type": "user"}\n')

        with patch(
            "agentgit.url_resolver.clone_git_repo",
            return_value=(mock_repo_path, tmp_path / "cloned"),
        ):
            result = resolve_transcript_source("https://github.com/user/repo.git")

            assert result.source_type == "git_repo"
            assert result.is_temporary
            assert result.original_url == "https://github.com/user/repo.git"
            assert result.path == mock_repo_path

    def test_cleanup_removes_temp_file(self, tmp_path):
        """Should clean up temporary files."""
        content = b'{"type": "user"}\n'

        with patch("agentgit.url_resolver.fetch_url", return_value=content):
            result = resolve_transcript_source("https://example.com/session.jsonl")
            temp_path = result.path

            assert temp_path.exists()
            result.cleanup()
            assert not temp_path.exists()


class TestFindTranscriptsInRepo:
    """Tests for find_transcripts_in_repo function."""

    def test_finds_jsonl_files(self, tmp_path):
        """Should find JSONL files in repository."""
        (tmp_path / "session.jsonl").write_text('{"type": "user"}\n')
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "other.jsonl").write_text('{"type": "user"}\n')

        transcripts = find_transcripts_in_repo(tmp_path)

        assert len(transcripts) == 2
        assert any(t.name == "session.jsonl" for t in transcripts)
        assert any(t.name == "other.jsonl" for t in transcripts)

    def test_excludes_hidden_dirs(self, tmp_path):
        """Should exclude files in hidden directories."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.jsonl").write_text('{"type": "user"}\n')
        (tmp_path / "session.jsonl").write_text('{"type": "user"}\n')

        transcripts = find_transcripts_in_repo(tmp_path)

        assert len(transcripts) == 1
        assert transcripts[0].name == "session.jsonl"

    def test_excludes_node_modules(self, tmp_path):
        """Should exclude files in node_modules."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "package.jsonl").write_text('{"type": "user"}\n')
        (tmp_path / "session.jsonl").write_text('{"type": "user"}\n')

        transcripts = find_transcripts_in_repo(tmp_path)

        assert len(transcripts) == 1
        assert transcripts[0].name == "session.jsonl"

    def test_empty_repo(self, tmp_path):
        """Should return empty list for repo without JSONL files."""
        (tmp_path / "README.md").write_text("# Readme")

        transcripts = find_transcripts_in_repo(tmp_path)

        assert transcripts == []


class TestIsAgentgitRepo:
    """Tests for is_agentgit_repo function."""

    def test_agentgit_repo_with_trailers(self, tmp_path):
        """Should detect agentgit repo by commit trailers."""
        from git import Repo as GitRepo

        from agentgit.url_resolver import is_agentgit_repo

        repo = GitRepo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        # Create a file and commit with agentgit trailers
        (tmp_path / "test.py").write_text("x = 1")
        repo.index.add(["test.py"])
        repo.index.commit("Add test file\n\nTool-Id: toolu_001\nPrompt-Id: abc123")

        assert is_agentgit_repo(tmp_path)

    def test_regular_repo_not_agentgit(self, tmp_path):
        """Should not detect regular repos as agentgit."""
        from git import Repo as GitRepo

        from agentgit.url_resolver import is_agentgit_repo

        repo = GitRepo.init(tmp_path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test")
            config.set_value("user", "email", "test@test.com")

        (tmp_path / "test.py").write_text("x = 1")
        repo.index.add(["test.py"])
        repo.index.commit("Add test file")

        assert not is_agentgit_repo(tmp_path)

    def test_non_repo_directory(self, tmp_path):
        """Should return False for non-git directory."""
        from agentgit.url_resolver import is_agentgit_repo

        (tmp_path / "test.py").write_text("x = 1")

        assert not is_agentgit_repo(tmp_path)


class TestGitConfigDefaults:
    """Tests for git config author defaults."""

    def test_get_git_config_user(self):
        """Should get user from git config."""
        from agentgit.cli import get_git_config_user

        # This will get the actual git config - just verify it returns tuple
        name, email = get_git_config_user()
        assert name is None or isinstance(name, str)
        assert email is None or isinstance(email, str)

    def test_get_default_author_with_git_config(self):
        """Should use git config values when available."""
        from unittest.mock import patch

        from agentgit.cli import get_default_author

        with patch("agentgit.cli.get_git_config_user", return_value=("Test User", "test@example.com")):
            name, email = get_default_author()
            assert name == "Test User"
            assert email == "test@example.com"

    def test_get_default_author_fallback(self):
        """Should fall back to Agent when git config not available."""
        from unittest.mock import patch

        from agentgit.cli import get_default_author

        with patch("agentgit.cli.get_git_config_user", return_value=(None, None)):
            name, email = get_default_author()
            assert name == "Agent"
            assert email == "agent@local"


class TestCliIntegration:
    """Integration tests for CLI with URL support."""

    def test_process_with_url_help(self):
        """Should show URL examples in help."""
        from click.testing import CliRunner

        from agentgit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "http://" in result.output.lower() or "https://" in result.output.lower()
        assert "URL" in result.output

    def test_process_url_transcript(self, tmp_path):
        """Should process transcript from URL."""
        from click.testing import CliRunner

        from agentgit.cli import main

        # Create sample JSONL content
        content = json.dumps({
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Create hello.py"},
            "sessionId": "test",
        }) + "\n" + json.dumps({
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:05.000Z",
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
        }) + "\n"

        with patch("agentgit.url_resolver.fetch_url", return_value=content.encode()):
            runner = CliRunner()
            output_dir = tmp_path / "output"
            result = runner.invoke(
                main,
                ["process", "https://example.com/session.jsonl", "-o", str(output_dir)],
            )

            assert result.exit_code == 0, result.output
            assert "Fetched from: https://example.com/session.jsonl" in result.output
            assert "Created git repository" in result.output
            assert (output_dir / ".git").exists()

    def test_process_url_watch_mode_rejected(self, tmp_path):
        """Should reject watch mode for URLs."""
        from click.testing import CliRunner

        from agentgit.cli import main

        content = b'{"type": "user"}\n'

        with patch("agentgit.url_resolver.fetch_url", return_value=content):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["process", "https://example.com/session.jsonl", "--watch"],
            )

            assert result.exit_code != 0
            assert "not supported for URLs" in result.output

    def test_process_nonexistent_file_error(self):
        """Should show error for nonexistent local file."""
        from click.testing import CliRunner

        from agentgit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["process", "/nonexistent/file.jsonl"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_default_command_with_url(self, tmp_path):
        """Should use process as default command for URLs."""
        from click.testing import CliRunner

        from agentgit.cli import main

        content = json.dumps({
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Test"},
            "sessionId": "test",
        }) + "\n" + json.dumps({
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:05.000Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Write",
                        "input": {"file_path": "/test/f.py", "content": "x=1"},
                    },
                ]
            },
        }) + "\n"

        with patch("agentgit.url_resolver.fetch_url", return_value=content.encode()):
            runner = CliRunner()
            output_dir = tmp_path / "output"
            # Run without "process" command - should still work
            result = runner.invoke(
                main,
                ["https://example.com/session.jsonl", "-o", str(output_dir)],
            )

            assert result.exit_code == 0, result.output
            assert "Created git repository" in result.output
