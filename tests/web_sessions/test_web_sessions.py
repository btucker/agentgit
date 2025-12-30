"""Tests for agentgit.web_sessions module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentgit.web_sessions import (
    API_BASE_URL,
    WebSession,
    WebSessionError,
    fetch_session_data,
    fetch_sessions,
    find_matching_local_project,
    get_api_headers,
    get_org_uuid_from_config,
    resolve_credentials,
    session_to_jsonl_entries,
)


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
