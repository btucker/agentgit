"""Claude Code Web Sessions plugin for agentgit.

This plugin discovers and caches web sessions from the Claude API,
making them available through the standard plugin discovery interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from agentgit.plugins import hookimpl
from agentgit.web_sessions import (
    DiscoveredWebSession,
    WebSessionError,
    discover_web_sessions,
    fetch_session_data,
    resolve_credentials,
    session_to_jsonl_entries,
)

if TYPE_CHECKING:
    from agentgit.core import Transcript

# Format identifier - same as local Claude Code since format is identical
FORMAT_CLAUDE_CODE_WEB = "claude_code_jsonl"

# Cache directory for downloaded web sessions
WEB_SESSION_CACHE_DIR = Path.home() / ".cache" / "agentgit" / "web-sessions"


def get_cached_session_path(session_id: str) -> Path:
    """Get the cache path for a web session."""
    return WEB_SESSION_CACHE_DIR / f"{session_id}.jsonl"


def is_session_cached(session_id: str) -> bool:
    """Check if a web session is already cached locally."""
    return get_cached_session_path(session_id).exists()


def cache_web_session(
    session_id: str,
    token: str,
    org_uuid: str,
) -> Path | None:
    """Download and cache a web session.

    Args:
        session_id: The session ID to download.
        token: API access token.
        org_uuid: Organization UUID.

    Returns:
        Path to the cached session file, or None if download failed.
    """
    try:
        session_data = fetch_session_data(token, org_uuid, session_id)
        entries = session_to_jsonl_entries(session_data)

        if not entries:
            return None

        # Ensure cache directory exists
        WEB_SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write to cache
        cache_path = get_cached_session_path(session_id)
        with open(cache_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return cache_path

    except WebSessionError:
        return None


def get_session_metadata_path(session_id: str) -> Path:
    """Get the metadata file path for a web session."""
    return WEB_SESSION_CACHE_DIR / f"{session_id}.meta.json"


def cache_session_metadata(session: DiscoveredWebSession) -> None:
    """Cache metadata about a web session for later use."""
    WEB_SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = get_session_metadata_path(session.session_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "project_path": session.project_path,
                "project_name": session.project_name,
            },
            f,
        )


def load_session_metadata(session_id: str) -> DiscoveredWebSession | None:
    """Load cached metadata for a web session."""
    meta_path = get_session_metadata_path(session_id)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return DiscoveredWebSession(
            session_id=data["session_id"],
            title=data["title"],
            created_at=data["created_at"],
            project_path=data.get("project_path"),
            project_name=data.get("project_name"),
        )
    except (json.JSONDecodeError, KeyError, OSError):
        return None


class ClaudeCodeWebPlugin:
    """Plugin for discovering and processing Claude Code web sessions.

    This plugin fetches web sessions from the Claude API and caches them
    locally so they can be processed through the standard plugin pipeline.
    """

    @hookimpl
    def agentgit_get_plugin_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": "claude_code_web",
            "description": "Claude Code web sessions (cloud API)",
        }

    @hookimpl
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Discover Claude Code web sessions.

        Fetches available web sessions from the API and caches them locally.
        Returns paths to the cached session files.

        Note: Currently returns ALL web sessions regardless of project_path,
        since the Claude API doesn't provide project information in the
        sessions list endpoint.

        Args:
            project_path: Path to the project (currently ignored).

        Returns:
            List of paths to cached web session files.
        """
        # Discover all web sessions (API doesn't provide project filtering)
        # TODO: Extract project from session loglines and cache for filtering
        sessions = discover_web_sessions(project_path=None)

        if not sessions:
            return []

        # Try to get credentials for caching
        try:
            token, org_uuid = resolve_credentials()
        except WebSessionError:
            return []

        cached_paths = []
        for session in sessions:
            # Check if already cached
            cache_path = get_cached_session_path(session.session_id)
            if cache_path.exists():
                cached_paths.append(cache_path)
                # Update metadata
                cache_session_metadata(session)
                continue

            # Download and cache the session
            cached = cache_web_session(session.session_id, token, org_uuid)
            if cached:
                cached_paths.append(cached)
                cache_session_metadata(session)

        # Sort by modification time (most recent first)
        cached_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cached_paths

    @hookimpl
    def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
        """Get project name for a cached web session.

        Reads the cached metadata to get the project name.
        """
        # Only handle files in our cache directory
        try:
            transcript_path.resolve().relative_to(WEB_SESSION_CACHE_DIR.resolve())
        except ValueError:
            return None

        # Extract session ID from filename
        session_id = transcript_path.stem
        if session_id.endswith(".meta"):
            return None

        metadata = load_session_metadata(session_id)
        if metadata and metadata.project_name:
            return metadata.project_name

        return None

    @hookimpl
    def agentgit_get_display_name(self, transcript_path: Path) -> str | None:
        """Get display name for a cached web session.

        Returns the session title from cached metadata.
        """
        # Only handle files in our cache directory
        try:
            transcript_path.resolve().relative_to(WEB_SESSION_CACHE_DIR.resolve())
        except ValueError:
            return None

        # Extract session ID from filename
        session_id = transcript_path.stem
        if session_id.endswith(".meta"):
            return None

        metadata = load_session_metadata(session_id)
        if metadata:
            return f"[Web] {metadata.display_name}"

        return f"[Web] {session_id[:8]}..."


def clear_web_session_cache() -> int:
    """Clear all cached web sessions.

    Returns:
        Number of files deleted.
    """
    if not WEB_SESSION_CACHE_DIR.exists():
        return 0

    count = 0
    for path in WEB_SESSION_CACHE_DIR.iterdir():
        if path.is_file():
            path.unlink()
            count += 1

    return count


def refresh_web_session(session_id: str) -> Path | None:
    """Force refresh a cached web session.

    Args:
        session_id: The session ID to refresh.

    Returns:
        Path to the refreshed cache file, or None if refresh failed.
    """
    # Remove existing cache
    cache_path = get_cached_session_path(session_id)
    if cache_path.exists():
        cache_path.unlink()

    # Re-download
    try:
        token, org_uuid = resolve_credentials()
        return cache_web_session(session_id, token, org_uuid)
    except WebSessionError:
        return None
