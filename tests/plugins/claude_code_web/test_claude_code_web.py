"""Tests for agentgit.formats.claude_code_web plugin."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentgit.formats.claude_code_web import (
    WEB_SESSION_CACHE_DIR,
    ClaudeCodeWebPlugin,
    DiscoveredWebSession,
    WebSession,
    WebSessionError,
    cache_session_metadata,
    cache_web_session,
    clear_web_session_cache,
    get_cached_session_path,
    get_session_metadata_path,
    is_session_cached,
    load_session_metadata,
    refresh_web_session,
)


@pytest.fixture
def plugin():
    """Create a ClaudeCodeWebPlugin instance."""
    return ClaudeCodeWebPlugin()


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Use a temporary directory for the web session cache."""
    cache_dir = tmp_path / "web-sessions"
    monkeypatch.setattr(
        "agentgit.formats.claude_code_web.WEB_SESSION_CACHE_DIR",
        cache_dir,
    )
    return cache_dir


class TestPluginInfo:
    """Tests for plugin info hook."""

    def test_returns_plugin_info(self, plugin):
        """Should return correct plugin info."""
        info = plugin.agentgit_get_plugin_info()

        assert info["name"] == "claude_code_web"
        assert "web" in info["description"].lower()


class TestCacheFunctions:
    """Tests for cache helper functions."""

    def test_get_cached_session_path(self, temp_cache_dir):
        """Should return correct cache path for session ID."""
        path = get_cached_session_path("session-123")

        assert path == temp_cache_dir / "session-123.jsonl"

    def test_is_session_cached_false(self, temp_cache_dir):
        """Should return False when session not cached."""
        assert is_session_cached("nonexistent") is False

    def test_is_session_cached_true(self, temp_cache_dir):
        """Should return True when session is cached."""
        temp_cache_dir.mkdir(parents=True)
        (temp_cache_dir / "session-123.jsonl").write_text('{"type": "user"}\n')

        assert is_session_cached("session-123") is True

    def test_get_session_metadata_path(self, temp_cache_dir):
        """Should return correct metadata path."""
        path = get_session_metadata_path("session-123")

        assert path == temp_cache_dir / "session-123.meta.json"


class TestCacheWebSession:
    """Tests for cache_web_session function."""

    def test_caches_session_successfully(self, temp_cache_dir):
        """Should download and cache session."""
        mock_session_data = {
            "id": "session-123",
            "project_path": "/path/to/project",
            "loglines": [
                {"type": "user", "timestamp": "2025-01-15T10:00:00Z", "message": {}},
                {"type": "assistant", "timestamp": "2025-01-15T10:00:05Z", "message": {}},
            ],
        }

        with patch(
            "agentgit.formats.claude_code_web.fetch_session_data",
            return_value=mock_session_data,
        ):
            result = cache_web_session("session-123", "token", "org")

            assert result is not None
            assert result.exists()
            assert result.suffix == ".jsonl"

            # Verify content
            lines = result.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_returns_none_on_error(self, temp_cache_dir):
        """Should return None when fetch fails."""
        with patch(
            "agentgit.formats.claude_code_web.fetch_session_data",
            side_effect=WebSessionError("API Error"),
        ):
            result = cache_web_session("session-123", "token", "org")

            assert result is None

    def test_returns_none_when_no_entries(self, temp_cache_dir):
        """Should return None when session has no entries."""
        mock_session_data = {
            "id": "session-123",
            "loglines": [],
        }

        with patch(
            "agentgit.formats.claude_code_web.fetch_session_data",
            return_value=mock_session_data,
        ):
            result = cache_web_session("session-123", "token", "org")

            assert result is None


class TestSessionMetadata:
    """Tests for session metadata caching."""

    def test_cache_and_load_metadata(self, temp_cache_dir):
        """Should cache and load session metadata."""
        session = DiscoveredWebSession(
            session_id="session-123",
            title="Test Session",
            created_at="2025-01-15T10:00:00Z",
            project_path="/path/to/project",
            project_name="project",
        )

        cache_session_metadata(session)
        loaded = load_session_metadata("session-123")

        assert loaded is not None
        assert loaded.session_id == "session-123"
        assert loaded.title == "Test Session"
        assert loaded.project_name == "project"

    def test_load_nonexistent_metadata(self, temp_cache_dir):
        """Should return None for nonexistent metadata."""
        result = load_session_metadata("nonexistent")

        assert result is None

    def test_load_corrupted_metadata(self, temp_cache_dir):
        """Should return None for corrupted metadata."""
        temp_cache_dir.mkdir(parents=True)
        (temp_cache_dir / "session-123.meta.json").write_text("not json")

        result = load_session_metadata("session-123")

        assert result is None


class TestDiscoverTranscripts:
    """Tests for plugin discovery hook."""

    def test_returns_empty_when_no_credentials(self, plugin, temp_cache_dir, monkeypatch):
        """Should return empty list when credentials not available."""
        monkeypatch.setattr("sys.platform", "linux")

        with patch(
            "agentgit.formats.claude_code_web.discover_web_sessions",
            return_value=[],
        ):
            result = plugin.agentgit_discover_transcripts(project_path=None)

            assert result == []

    def test_returns_cached_sessions(self, plugin, temp_cache_dir):
        """Should return paths to cached sessions."""
        # Create a cached session
        temp_cache_dir.mkdir(parents=True)
        cached_file = temp_cache_dir / "session-123.jsonl"
        cached_file.write_text('{"type": "user"}\n')

        mock_sessions = [
            DiscoveredWebSession(
                session_id="session-123",
                title="Test",
                created_at="2025-01-15T10:00:00Z",
            ),
        ]

        with patch(
            "agentgit.formats.claude_code_web.discover_web_sessions",
            return_value=mock_sessions,
        ):
            with patch(
                "agentgit.formats.claude_code_web.resolve_credentials",
                return_value=("token", "org"),
            ):
                result = plugin.agentgit_discover_transcripts(project_path=None)

                assert len(result) == 1
                assert result[0] == cached_file

    def test_downloads_and_caches_new_sessions(self, plugin, temp_cache_dir):
        """Should download and cache sessions not yet cached."""
        mock_sessions = [
            DiscoveredWebSession(
                session_id="new-session",
                title="New Session",
                created_at="2025-01-15T10:00:00Z",
            ),
        ]

        mock_session_data = {
            "id": "new-session",
            "loglines": [
                {"type": "user", "timestamp": "2025-01-15T10:00:00Z", "message": {}},
            ],
        }

        with patch(
            "agentgit.formats.claude_code_web.discover_web_sessions",
            return_value=mock_sessions,
        ):
            with patch(
                "agentgit.formats.claude_code_web.resolve_credentials",
                return_value=("token", "org"),
            ):
                with patch(
                    "agentgit.formats.claude_code_web.fetch_session_data",
                    return_value=mock_session_data,
                ):
                    result = plugin.agentgit_discover_transcripts(project_path=None)

                    assert len(result) == 1
                    assert result[0].exists()


class TestGetProjectName:
    """Tests for project name hook."""

    def test_returns_project_name_from_metadata(self, plugin, temp_cache_dir):
        """Should return project name from cached metadata."""
        temp_cache_dir.mkdir(parents=True)

        # Create cached session and metadata
        (temp_cache_dir / "session-123.jsonl").write_text('{"type": "user"}\n')
        session = DiscoveredWebSession(
            session_id="session-123",
            title="Test",
            created_at="2025-01-15T10:00:00Z",
            project_name="myproject",
        )
        cache_session_metadata(session)

        result = plugin.agentgit_get_project_name(temp_cache_dir / "session-123.jsonl")

        assert result == "myproject"

    def test_returns_none_for_non_cache_path(self, plugin, tmp_path):
        """Should return None for paths outside cache directory."""
        other_file = tmp_path / "other.jsonl"
        other_file.write_text('{"type": "user"}\n')

        result = plugin.agentgit_get_project_name(other_file)

        assert result is None


class TestGetDisplayName:
    """Tests for display name hook."""

    def test_returns_display_name_with_web_prefix(self, plugin, temp_cache_dir):
        """Should return display name with [Web] prefix."""
        temp_cache_dir.mkdir(parents=True)

        # Create cached session and metadata
        (temp_cache_dir / "session-123.jsonl").write_text('{"type": "user"}\n')
        session = DiscoveredWebSession(
            session_id="session-123",
            title="My Session",
            created_at="2025-01-15T10:00:00Z",
        )
        cache_session_metadata(session)

        result = plugin.agentgit_get_display_name(temp_cache_dir / "session-123.jsonl")

        assert result == "[Web] My Session"

    def test_returns_fallback_name(self, plugin, temp_cache_dir):
        """Should return fallback name when no metadata."""
        temp_cache_dir.mkdir(parents=True)
        (temp_cache_dir / "abc12345.jsonl").write_text('{"type": "user"}\n')

        result = plugin.agentgit_get_display_name(temp_cache_dir / "abc12345.jsonl")

        assert result == "[Web] abc12345..."

    def test_returns_none_for_non_cache_path(self, plugin, tmp_path):
        """Should return None for paths outside cache directory."""
        other_file = tmp_path / "other.jsonl"
        other_file.write_text('{"type": "user"}\n')

        result = plugin.agentgit_get_display_name(other_file)

        assert result is None


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_clear_cache(self, temp_cache_dir):
        """Should clear all cached files."""
        temp_cache_dir.mkdir(parents=True)
        (temp_cache_dir / "session-1.jsonl").write_text('{"type": "user"}\n')
        (temp_cache_dir / "session-2.jsonl").write_text('{"type": "user"}\n')
        (temp_cache_dir / "session-1.meta.json").write_text("{}")

        count = clear_web_session_cache()

        assert count == 3
        assert len(list(temp_cache_dir.iterdir())) == 0

    def test_clear_empty_cache(self, temp_cache_dir):
        """Should handle clearing empty/nonexistent cache."""
        count = clear_web_session_cache()

        assert count == 0

    def test_refresh_session(self, temp_cache_dir):
        """Should re-download session."""
        temp_cache_dir.mkdir(parents=True)

        # Create existing cached file
        cached = temp_cache_dir / "session-123.jsonl"
        cached.write_text('{"old": "content"}\n')

        mock_session_data = {
            "id": "session-123",
            "loglines": [
                {"type": "user", "timestamp": "2025-01-15T10:00:00Z", "message": {}},
            ],
        }

        with patch(
            "agentgit.formats.claude_code_web.resolve_credentials",
            return_value=("token", "org"),
        ):
            with patch(
                "agentgit.formats.claude_code_web.fetch_session_data",
                return_value=mock_session_data,
            ):
                result = refresh_web_session("session-123")

                assert result is not None
                # Verify it was re-downloaded
                content = result.read_text()
                assert "old" not in content
