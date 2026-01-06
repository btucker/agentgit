"""Tests for agentgit.web_sessions module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentgit.formats.claude_code_web import (
    API_BASE_URL,
    DiscoveredWebSession,
    WebSession,
    WebSessionError,
    discover_web_sessions,
    fetch_session_data,
    fetch_sessions,
    find_matching_local_project,
    get_api_headers,
    get_org_uuid_from_config,
    resolve_credentials,
    session_to_jsonl_entries,
    try_resolve_credentials,
)
from agentgit.utils import get_git_remotes, normalize_git_url


class TestNormalizeGitUrl:
    """Tests for normalize_git_url function."""

    def test_https_url_basic(self):
        """Should normalize HTTPS URLs correctly."""
        url = "https://github.com/user/repo"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_https_url_with_git_suffix(self):
        """Should remove .git suffix from HTTPS URLs."""
        url = "https://github.com/user/repo.git"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_http_url(self):
        """Should normalize HTTP URLs correctly."""
        url = "http://github.com/user/repo"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_https_url_with_trailing_slash(self):
        """Should remove trailing slashes."""
        url = "https://github.com/user/repo/"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_https_url_with_trailing_slash_and_git(self):
        """Should remove trailing slash and .git suffix."""
        url = "https://github.com/user/repo.git/"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_ssh_url_with_user(self):
        """Should normalize SSH URLs with user@ prefix."""
        url = "git@github.com:user/repo.git"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_ssh_url_without_user(self):
        """Should normalize SSH URLs without user@ prefix."""
        url = "github.com:user/repo.git"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_ssh_url_without_git_suffix(self):
        """Should normalize SSH URLs without .git suffix."""
        url = "git@github.com:user/repo"
        assert normalize_git_url(url) == "github.com/user/repo"

    def test_ssh_url_with_different_user(self):
        """Should handle SSH URLs with different usernames."""
        url = "myuser@gitlab.com:user/repo.git"
        assert normalize_git_url(url) == "gitlab.com/user/repo"

    def test_ssh_url_with_nested_path(self):
        """Should handle SSH URLs with nested paths."""
        url = "git@github.com:org/team/repo.git"
        assert normalize_git_url(url) == "github.com/org/team/repo"

    def test_https_url_with_nested_path(self):
        """Should handle HTTPS URLs with nested paths."""
        url = "https://github.com/org/team/repo"
        assert normalize_git_url(url) == "github.com/org/team/repo"

    def test_different_hosts(self):
        """Should work with different Git hosting services."""
        urls = [
            ("https://gitlab.com/user/repo", "gitlab.com/user/repo"),
            ("git@gitlab.com:user/repo.git", "gitlab.com/user/repo"),
            ("https://bitbucket.org/user/repo", "bitbucket.org/user/repo"),
            ("git@bitbucket.org:user/repo.git", "bitbucket.org/user/repo"),
        ]
        for url, expected in urls:
            assert normalize_git_url(url) == expected

    def test_ssh_vs_https_equivalence(self):
        """SSH and HTTPS URLs for same repo should normalize to same value."""
        ssh_url = "git@github.com:btucker/agentgit.git"
        https_url = "https://github.com/btucker/agentgit"

        assert normalize_git_url(ssh_url) == normalize_git_url(https_url)
        assert normalize_git_url(ssh_url) == "github.com/btucker/agentgit"

    def test_unknown_format_returns_as_is(self):
        """Should return URL as-is if format not recognized."""
        url = "/local/path/to/repo"
        assert normalize_git_url(url) == "/local/path/to/repo"


class TestGetGitRemotes:
    """Tests for get_git_remotes function."""

    def test_returns_empty_list_for_non_git_directory(self, tmp_path):
        """Should return empty list for non-git directories."""
        project = tmp_path / "not-a-git-repo"
        project.mkdir()

        remotes = get_git_remotes(project)
        assert remotes == []

    def test_returns_remotes_for_git_repository(self, tmp_path):
        """Should return list of remote URLs for a git repository."""
        import subprocess

        # Create a git repo with remotes
        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        remotes = get_git_remotes(repo)
        assert len(remotes) == 1
        assert "git@github.com:user/repo.git" in remotes

    def test_returns_multiple_remotes(self, tmp_path):
        """Should return all unique remote URLs."""
        import subprocess

        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "remote", "add", "upstream", "https://github.com/upstream/repo"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        remotes = get_git_remotes(repo)
        assert len(remotes) == 2
        assert "git@github.com:user/repo.git" in remotes
        assert "https://github.com/upstream/repo" in remotes

    def test_deduplicates_same_remote(self, tmp_path):
        """Should not duplicate the same remote URL."""
        import subprocess

        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # git remote -v shows each remote twice (fetch and push)
        remotes = get_git_remotes(repo)
        assert remotes.count("git@github.com:user/repo.git") == 1

    def test_handles_subprocess_errors_gracefully(self, tmp_path):
        """Should return empty list if git command fails."""
        # Use a path that will cause git to fail
        nonexistent = tmp_path / "does-not-exist"

        remotes = get_git_remotes(nonexistent)
        assert remotes == []


class TestWebSession:
    """Tests for WebSession dataclass."""

    def test_matches_project_true(self, tmp_path):
        """Should return True when paths match."""
        project = tmp_path / "myproject"
        project.mkdir()

        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            project_path=str(project),
        )

        assert session.matches_project(project) is True

    def test_matches_project_false(self, tmp_path):
        """Should return False when paths don't match."""
        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()

        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            project_path=str(project1),
        )

        assert session.matches_project(project2) is False

    def test_matches_project_no_path(self, tmp_path):
        """Should return False when session has no project_path."""
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
        )

        assert session.matches_project(tmp_path) is False

    def test_matches_project_by_git_url_ssh(self, tmp_path):
        """Should match project by git URL (SSH format)."""
        import subprocess

        # Create a git repo with a remote
        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # Session with HTTPS URL for same repo
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            git_url="https://github.com/user/repo",
        )

        assert session.matches_project(repo) is True

    def test_matches_project_by_git_url_https(self, tmp_path):
        """Should match project by git URL (HTTPS format)."""
        import subprocess

        # Create a git repo with HTTPS remote
        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/user/repo.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # Session with SSH URL for same repo
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            git_url="git@github.com:user/repo.git",
        )

        assert session.matches_project(repo) is True

    def test_matches_project_checks_all_remotes(self, tmp_path):
        """Should check all git remotes for a match."""
        import subprocess

        # Create a git repo with multiple remotes
        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo1.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "remote", "add", "upstream", "git@github.com:user/repo2.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # Session matches the upstream remote
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            git_url="https://github.com/user/repo2",
        )

        assert session.matches_project(repo) is True

    def test_matches_project_git_url_no_match(self, tmp_path):
        """Should return False when git URLs don't match."""
        import subprocess

        # Create a git repo with a remote
        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo1.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # Session with different repo URL
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            git_url="https://github.com/user/repo2",
        )

        assert session.matches_project(repo) is False

    def test_matches_project_git_url_fallback_to_path(self, tmp_path):
        """Should fall back to path matching if git URL doesn't match."""
        import subprocess

        # Create a git repo with a remote
        repo = tmp_path / "test-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo1.git"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # Session with different git URL but matching path
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            git_url="https://github.com/user/repo2",
            project_path=str(repo),
        )

        # Should match by path even though git URL doesn't match
        assert session.matches_project(repo) is True

    def test_matches_project_git_url_and_path_fallback(self, tmp_path):
        """Should try git URL first but fall back to path if git URL doesn't match."""
        import subprocess

        # Create two repos
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()

        # repo1 has the matching git URL
        subprocess.run(["git", "init"], cwd=repo1, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/myrepo.git"],
            cwd=repo1,
            check=True,
            capture_output=True,
        )

        # Session has git_url matching repo1 but project_path pointing to repo2
        session = WebSession(
            id="test-123",
            title="Test Session",
            created_at="2025-01-01T00:00:00Z",
            git_url="https://github.com/user/myrepo",
            project_path=str(repo2),
        )

        # Should match repo1 by git URL
        assert session.matches_project(repo1) is True
        # Should also match repo2 by path (fallback when git URL doesn't match)
        assert session.matches_project(repo2) is True


class TestGetApiHeaders:
    """Tests for get_api_headers function."""

    def test_returns_correct_headers(self):
        """Should return proper API headers."""
        headers = get_api_headers("test-token", "test-org-uuid")

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["x-organization-uuid"] == "test-org-uuid"
        assert headers["Content-Type"] == "application/json"
        assert "anthropic-version" in headers


class TestGetOrgUuidFromConfig:
    """Tests for get_org_uuid_from_config function."""

    def test_returns_uuid_when_present(self, tmp_path, monkeypatch):
        """Should return org UUID from ~/.claude.json."""
        config = {"oauthAccount": {"organizationUuid": "org-123"}}
        config_path = tmp_path / ".claude.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = get_org_uuid_from_config()
        assert result == "org-123"

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        """Should return None when config file doesn't exist."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = get_org_uuid_from_config()
        assert result is None

    def test_returns_none_when_uuid_missing(self, tmp_path, monkeypatch):
        """Should return None when UUID not in config."""
        config = {"someOtherKey": "value"}
        config_path = tmp_path / ".claude.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = get_org_uuid_from_config()
        assert result is None


class TestResolveCredentials:
    """Tests for resolve_credentials function."""

    def test_uses_provided_credentials(self):
        """Should use explicitly provided token and org_uuid."""
        token, org_uuid = resolve_credentials("my-token", "my-org")
        assert token == "my-token"
        assert org_uuid == "my-org"

    def test_raises_when_token_not_available(self, tmp_path, monkeypatch):
        """Should raise when token can't be retrieved."""
        # Mock non-macOS platform
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(WebSessionError) as exc_info:
            resolve_credentials(None, "org-123")

        assert "access token" in str(exc_info.value).lower()

    def test_raises_when_org_uuid_not_available(self, tmp_path, monkeypatch):
        """Should raise when org_uuid can't be retrieved."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(WebSessionError) as exc_info:
            resolve_credentials("token-123", None)

        assert "organization UUID" in str(exc_info.value)


class TestFetchSessions:
    """Tests for fetch_sessions function."""

    def test_fetches_sessions_successfully(self):
        """Should return list of WebSession objects."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "session-1",
                    "title": "First Session",
                    "created_at": "2025-01-01T00:00:00Z",
                    "project_path": "/path/to/project",
                },
                {
                    "id": "session-2",
                    "title": "Second Session",
                    "created_at": "2025-01-02T00:00:00Z",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            sessions = fetch_sessions("token", "org")

            mock_get.assert_called_once()
            assert len(sessions) == 2
            assert sessions[0].id == "session-1"
            assert sessions[0].title == "First Session"
            assert sessions[0].project_path == "/path/to/project"
            assert sessions[1].project_path is None

    def test_extracts_git_url_from_session_context(self):
        """Should extract git URL from session_context.sources."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "session-1",
                    "title": "Test Session",
                    "created_at": "2025-01-01T00:00:00Z",
                    "session_context": {
                        "sources": [
                            {
                                "type": "git_repository",
                                "url": "https://github.com/user/repo",
                            }
                        ]
                    },
                },
                {
                    "id": "session-2",
                    "title": "No Git Session",
                    "created_at": "2025-01-02T00:00:00Z",
                    "session_context": {
                        "sources": []
                    },
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            sessions = fetch_sessions("token", "org")

            assert len(sessions) == 2
            assert sessions[0].git_url == "https://github.com/user/repo"
            assert sessions[1].git_url is None

    def test_extracts_git_url_from_multiple_sources(self):
        """Should extract git_repository URL from multiple sources."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "session-1",
                    "title": "Test Session",
                    "created_at": "2025-01-01T00:00:00Z",
                    "session_context": {
                        "sources": [
                            {"type": "local_file", "path": "/some/file"},
                            {"type": "git_repository", "url": "git@github.com:user/repo.git"},
                            {"type": "other", "value": "something"},
                        ]
                    },
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            sessions = fetch_sessions("token", "org")

            assert sessions[0].git_url == "git@github.com:user/repo.git"

    def test_handles_missing_session_context(self):
        """Should handle sessions without session_context."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "session-1",
                    "title": "Test Session",
                    "created_at": "2025-01-01T00:00:00Z",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            sessions = fetch_sessions("token", "org")

            assert sessions[0].git_url is None

    def test_raises_on_http_error(self):
        """Should raise WebSessionError on HTTP failure."""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection failed")

            with pytest.raises(WebSessionError) as exc_info:
                fetch_sessions("token", "org")

            assert "Failed to fetch sessions" in str(exc_info.value)


class TestFetchSessionData:
    """Tests for fetch_session_data function."""

    def test_fetches_session_data_successfully(self):
        """Should return session data dictionary."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "session-1",
            "loglines": [{"type": "user", "message": {"content": "Hello"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            data = fetch_session_data("token", "org", "session-1")

            mock_get.assert_called_once()
            call_url = mock_get.call_args[0][0]
            assert "session-1" in call_url
            assert data["id"] == "session-1"

    def test_raises_on_http_error(self):
        """Should raise WebSessionError on HTTP failure."""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("Not found")

            with pytest.raises(WebSessionError) as exc_info:
                fetch_session_data("token", "org", "session-1")

            assert "Failed to fetch session session-1" in str(exc_info.value)


class TestFindMatchingLocalProject:
    """Tests for find_matching_local_project function."""

    def test_returns_none_when_no_project_path(self):
        """Should return None when session has no project_path."""
        session = WebSession(
            id="test-123",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
        )

        result = find_matching_local_project(session)
        assert result is None

    def test_returns_path_when_project_exists(self, tmp_path):
        """Should return project path when it exists locally."""
        project = tmp_path / "myproject"
        project.mkdir()

        session = WebSession(
            id="test-123",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            project_path=str(project),
        )

        result = find_matching_local_project(session)
        assert result == project

    def test_returns_none_when_project_doesnt_exist(self):
        """Should return None when project path doesn't exist."""
        session = WebSession(
            id="test-123",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            project_path="/nonexistent/path",
        )

        result = find_matching_local_project(session)
        assert result is None


class TestSessionToJsonlEntries:
    """Tests for session_to_jsonl_entries function."""

    def test_converts_loglines_to_entries(self):
        """Should convert API response to JSONL format."""
        session_data = {
            "id": "session-123",
            "project_path": "/path/to/project",
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
                        "content": [{"type": "text", "text": "Hi there!"}]
                    },
                },
            ],
        }

        entries = session_to_jsonl_entries(session_data)

        assert len(entries) == 2
        assert entries[0]["type"] == "user"
        assert entries[0]["sessionId"] == "session-123"
        assert entries[0]["cwd"] == "/path/to/project"
        assert entries[1]["type"] == "assistant"

    def test_filters_non_user_assistant_entries(self):
        """Should only include user and assistant entries."""
        session_data = {
            "id": "session-123",
            "loglines": [
                {"type": "user", "timestamp": "2025-01-01T10:00:00Z", "message": {}},
                {"type": "system", "timestamp": "2025-01-01T10:00:01Z", "message": {}},
                {"type": "assistant", "timestamp": "2025-01-01T10:00:02Z", "message": {}},
            ],
        }

        entries = session_to_jsonl_entries(session_data)

        assert len(entries) == 2
        assert entries[0]["type"] == "user"
        assert entries[1]["type"] == "assistant"

    def test_preserves_additional_fields(self):
        """Should preserve toolUseResult and other fields."""
        session_data = {
            "id": "session-123",
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00Z",
                    "message": {},
                    "toolUseResult": {"originalFile": "content"},
                    "isCompactSummary": True,
                    "isMeta": False,
                },
            ],
        }

        entries = session_to_jsonl_entries(session_data)

        assert entries[0]["toolUseResult"] == {"originalFile": "content"}
        assert entries[0]["isCompactSummary"] is True
        assert entries[0]["isMeta"] is False

    def test_handles_empty_loglines(self):
        """Should return empty list for empty loglines."""
        session_data = {"id": "session-123", "loglines": []}

        entries = session_to_jsonl_entries(session_data)

        assert entries == []

    def test_handles_missing_loglines(self):
        """Should return empty list when loglines missing."""
        session_data = {"id": "session-123"}

        entries = session_to_jsonl_entries(session_data)

        assert entries == []


class TestDiscoveredWebSession:
    """Tests for DiscoveredWebSession dataclass."""

    def test_created_at_formatted(self):
        """Should format ISO timestamp correctly."""
        session = DiscoveredWebSession(
            session_id="test-123",
            title="Test Session",
            created_at="2025-01-15T10:30:00Z",
        )

        assert session.created_at_formatted == "2025-01-15 10:30:00"

    def test_created_at_formatted_handles_invalid(self):
        """Should return raw value for invalid timestamps."""
        session = DiscoveredWebSession(
            session_id="test-123",
            title="Test Session",
            created_at="invalid-date",
        )

        assert session.created_at_formatted == "invalid-date"

    def test_display_name_uses_title(self):
        """Should use title as display name."""
        session = DiscoveredWebSession(
            session_id="test-123",
            title="My Session",
            created_at="2025-01-15T10:30:00Z",
        )

        assert session.display_name == "My Session"

    def test_display_name_truncates_long_title(self):
        """Should truncate long titles."""
        long_title = "A" * 60
        session = DiscoveredWebSession(
            session_id="test-123",
            title=long_title,
            created_at="2025-01-15T10:30:00Z",
        )

        assert len(session.display_name) == 50
        assert session.display_name.endswith("...")

    def test_display_name_fallback(self):
        """Should fall back to session ID if no title."""
        session = DiscoveredWebSession(
            session_id="abc12345xyz",
            title="",
            created_at="2025-01-15T10:30:00Z",
        )

        assert session.display_name == "session-abc12345"

    def test_matches_project_true(self, tmp_path):
        """Should return True when paths match."""
        project = tmp_path / "myproject"
        project.mkdir()

        session = DiscoveredWebSession(
            session_id="test-123",
            title="Test",
            created_at="2025-01-15T10:30:00Z",
            project_path=str(project),
        )

        assert session.matches_project(project) is True

    def test_matches_project_false(self, tmp_path):
        """Should return False when paths don't match."""
        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()

        session = DiscoveredWebSession(
            session_id="test-123",
            title="Test",
            created_at="2025-01-15T10:30:00Z",
            project_path=str(project1),
        )

        assert session.matches_project(project2) is False

    def test_matches_project_no_path(self, tmp_path):
        """Should return False when session has no project_path."""
        session = DiscoveredWebSession(
            session_id="test-123",
            title="Test",
            created_at="2025-01-15T10:30:00Z",
        )

        assert session.matches_project(tmp_path) is False


class TestTryResolveCredentials:
    """Tests for try_resolve_credentials function."""

    def test_returns_none_on_failure(self, tmp_path, monkeypatch):
        """Should return None when credentials not available."""
        # Mock both platform and home directory to prevent reading actual credentials
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = try_resolve_credentials()

        assert result is None

    def test_returns_credentials_when_available(self, monkeypatch):
        """Should return credentials when available."""
        with patch(
            "agentgit.formats.claude_code_web.resolve_credentials",
            return_value=("token-123", "org-456"),
        ):
            result = try_resolve_credentials()

            assert result == ("token-123", "org-456")


class TestDiscoverWebSessions:
    """Tests for discover_web_sessions function."""

    def test_returns_empty_when_no_credentials(self, tmp_path, monkeypatch):
        """Should return empty list when credentials not available."""
        # Mock both platform and home directory to prevent reading actual credentials
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = discover_web_sessions()

        assert result == []

    def test_returns_empty_on_api_error(self):
        """Should return empty list when API fails."""
        with patch(
            "agentgit.formats.claude_code_web.try_resolve_credentials",
            return_value=("token", "org"),
        ):
            with patch(
                "agentgit.formats.claude_code_web.fetch_sessions",
                side_effect=WebSessionError("API Error"),
            ):
                result = discover_web_sessions()

                assert result == []

    def test_returns_sessions(self):
        """Should return discovered web sessions."""
        mock_sessions = [
            WebSession(
                id="session-1",
                title="First Session",
                created_at="2025-01-15T10:00:00Z",
                project_path="/path/to/project",
            ),
            WebSession(
                id="session-2",
                title="Second Session",
                created_at="2025-01-15T11:00:00Z",
            ),
        ]

        with patch(
            "agentgit.formats.claude_code_web.try_resolve_credentials",
            return_value=("token", "org"),
        ):
            with patch(
                "agentgit.formats.claude_code_web.fetch_sessions",
                return_value=mock_sessions,
            ):
                result = discover_web_sessions()

                assert len(result) == 2
                # Should be sorted by created_at, newest first
                assert result[0].session_id == "session-2"
                assert result[1].session_id == "session-1"
                assert result[1].project_name == "project"

    def test_filters_by_project(self, tmp_path):
        """Should filter sessions by project path."""
        project = tmp_path / "myproject"
        project.mkdir()

        mock_sessions = [
            WebSession(
                id="session-1",
                title="Matching",
                created_at="2025-01-15T10:00:00Z",
                project_path=str(project),
            ),
            WebSession(
                id="session-2",
                title="Non-matching",
                created_at="2025-01-15T11:00:00Z",
                project_path="/other/project",
            ),
        ]

        with patch(
            "agentgit.formats.claude_code_web.try_resolve_credentials",
            return_value=("token", "org"),
        ):
            with patch(
                "agentgit.formats.claude_code_web.fetch_sessions",
                return_value=mock_sessions,
            ):
                result = discover_web_sessions(project_path=project)

                assert len(result) == 1
                assert result[0].session_id == "session-1"

    def test_filters_by_git_url(self, tmp_path):
        """Should filter sessions by git URL matching."""
        import subprocess

        # Create a git repo with a remote
        project = tmp_path / "myproject"
        project.mkdir()
        subprocess.run(["git", "init"], cwd=project, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/myrepo.git"],
            cwd=project,
            check=True,
            capture_output=True,
        )

        mock_sessions = [
            WebSession(
                id="session-1",
                title="Matching by git URL",
                created_at="2025-01-15T10:00:00Z",
                git_url="https://github.com/user/myrepo",  # Matches via git URL
            ),
            WebSession(
                id="session-2",
                title="Non-matching",
                created_at="2025-01-15T11:00:00Z",
                git_url="https://github.com/user/otherrepo",
            ),
            WebSession(
                id="session-3",
                title="No git URL",
                created_at="2025-01-15T12:00:00Z",
            ),
        ]

        with patch(
            "agentgit.formats.claude_code_web.try_resolve_credentials",
            return_value=("token", "org"),
        ):
            with patch(
                "agentgit.formats.claude_code_web.fetch_sessions",
                return_value=mock_sessions,
            ):
                result = discover_web_sessions(project_path=project)

                assert len(result) == 1
                assert result[0].session_id == "session-1"
                assert result[0].title == "Matching by git URL"

    def test_filters_by_any_git_remote(self, tmp_path):
        """Should match if session git URL matches any local remote."""
        import subprocess

        # Create a git repo with multiple remotes
        project = tmp_path / "myproject"
        project.mkdir()
        subprocess.run(["git", "init"], cwd=project, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo1.git"],
            cwd=project,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "remote", "add", "upstream", "git@github.com:user/repo2.git"],
            cwd=project,
            check=True,
            capture_output=True,
        )

        mock_sessions = [
            WebSession(
                id="session-1",
                title="Matches origin",
                created_at="2025-01-15T10:00:00Z",
                git_url="https://github.com/user/repo1",
            ),
            WebSession(
                id="session-2",
                title="Matches upstream",
                created_at="2025-01-15T11:00:00Z",
                git_url="https://github.com/user/repo2",
            ),
            WebSession(
                id="session-3",
                title="Matches neither",
                created_at="2025-01-15T12:00:00Z",
                git_url="https://github.com/user/repo3",
            ),
        ]

        with patch(
            "agentgit.formats.claude_code_web.try_resolve_credentials",
            return_value=("token", "org"),
        ):
            with patch(
                "agentgit.formats.claude_code_web.fetch_sessions",
                return_value=mock_sessions,
            ):
                result = discover_web_sessions(project_path=project)

                assert len(result) == 2
                session_ids = [s.session_id for s in result]
                # Should be sorted by created_at (newest first)
                assert "session-2" in session_ids
                assert "session-1" in session_ids
                assert "session-3" not in session_ids

    def test_uses_provided_credentials(self):
        """Should use explicitly provided credentials."""
        mock_sessions = [
            WebSession(
                id="session-1",
                title="Test",
                created_at="2025-01-15T10:00:00Z",
            ),
        ]

        with patch(
            "agentgit.formats.claude_code_web.fetch_sessions",
            return_value=mock_sessions,
        ) as mock_fetch:
            result = discover_web_sessions(
                token="my-token",
                org_uuid="my-org",
            )

            assert len(result) == 1
            mock_fetch.assert_called_once_with("my-token", "my-org")
