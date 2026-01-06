"""Tests for agentgit.utils module."""

import pytest

from agentgit.utils import extract_deleted_paths


class TestExtractDeletedPaths:
    """Tests for extract_deleted_paths function."""

    def test_simple_rm(self):
        """Should extract paths from simple rm commands."""
        paths = extract_deleted_paths("rm file.txt")
        assert paths == ["file.txt"]

    def test_with_flags(self):
        """Should extract paths from rm with flags."""
        paths = extract_deleted_paths("rm -rf /path/to/dir")
        assert paths == ["/path/to/dir"]

    def test_multiple_paths(self):
        """Should extract multiple paths."""
        paths = extract_deleted_paths("rm file1.txt file2.txt")
        assert paths == ["file1.txt", "file2.txt"]

    def test_quoted_paths(self):
        """Should handle quoted paths."""
        paths = extract_deleted_paths('rm "path with spaces/file.txt"')
        assert paths == ["path with spaces/file.txt"]

    def test_single_quoted_paths(self):
        """Should handle single-quoted paths."""
        paths = extract_deleted_paths("rm 'path with spaces/file.txt'")
        assert paths == ["path with spaces/file.txt"]

    def test_non_rm_command(self):
        """Should return empty for non-rm commands."""
        paths = extract_deleted_paths("ls -la")
        assert paths == []

    def test_rm_with_multiple_flags(self):
        """Should handle rm with multiple flags."""
        paths = extract_deleted_paths("rm -r -f /path/to/dir")
        assert paths == ["/path/to/dir"]


class TestNormalizeGitUrl:
    """Tests for normalize_git_url function."""

    def test_normalizes_https_url(self):
        """Should normalize HTTPS URLs."""
        from agentgit.utils import normalize_git_url

        url = "https://github.com/user/repo.git"
        result = normalize_git_url(url)
        assert result == "github.com/user/repo"

    def test_converts_git_to_https(self):
        """Should convert git@ URLs to normalized format."""
        from agentgit.utils import normalize_git_url

        url = "git@github.com:user/repo.git"
        result = normalize_git_url(url)
        assert result == "github.com/user/repo"

    def test_handles_url_without_git_suffix(self):
        """Should handle URLs without .git suffix."""
        from agentgit.utils import normalize_git_url

        url = "https://github.com/user/repo"
        result = normalize_git_url(url)
        assert result == "github.com/user/repo"


class TestGetGitRemotes:
    """Tests for get_git_remotes function."""

    def test_get_git_remotes_no_remotes(self, tmp_path):
        """Should return empty list for repo with no remotes."""
        from git import Repo

        from agentgit.utils import get_git_remotes

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        Repo.init(repo_path)

        remotes = get_git_remotes(repo_path)
        assert remotes == []

    def test_get_git_remotes_with_remote(self, tmp_path):
        """Should return list of remote URLs."""
        from git import Repo

        from agentgit.utils import get_git_remotes

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)
        repo.create_remote("origin", "https://github.com/user/repo.git")

        remotes = get_git_remotes(repo_path)
        assert len(remotes) == 1
        assert "github.com/user/repo" in remotes[0]

    def test_get_git_remotes_non_repo(self, tmp_path):
        """Should return empty list for non-git directory."""
        from agentgit.utils import get_git_remotes

        remotes = get_git_remotes(tmp_path)
        assert remotes == []
