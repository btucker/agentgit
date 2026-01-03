"""Web session API client for fetching Claude Code sessions from the Claude API.

Based on the approach from simonw/claude-code-transcripts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class WebSession:
    """Represents a Claude Code web session."""

    id: str
    title: str
    created_at: str
    project_path: str | None = None

    def matches_project(self, project_path: Path) -> bool:
        """Check if this session is for the given project path."""
        if not self.project_path:
            return False
        # Normalize both paths for comparison
        session_path = Path(self.project_path).resolve()
        check_path = project_path.resolve()
        return session_path == check_path


class WebSessionError(Exception):
    """Error fetching or processing web sessions."""

    pass


def get_api_headers(token: str, org_uuid: str) -> dict[str, str]:
    """Build API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "x-organization-uuid": org_uuid,
    }


def get_access_token_from_keychain() -> str | None:
    """Get access token from macOS keychain.

    Returns None if not on macOS or if credentials not found.
    """
    if sys.platform != "darwin":
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (json.JSONDecodeError, subprocess.SubprocessError):
        return None


def get_org_uuid_from_config() -> str | None:
    """Get organization UUID from ~/.claude.json."""
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("oauthAccount", {}).get("organizationUuid")
    except (json.JSONDecodeError, OSError):
        return None


def resolve_credentials(
    token: str | None = None, org_uuid: str | None = None
) -> tuple[str, str]:
    """Resolve token and org_uuid from arguments or auto-detect.

    Args:
        token: API access token (optional, will auto-detect on macOS)
        org_uuid: Organization UUID (optional, will read from ~/.claude.json)

    Returns:
        Tuple of (token, org_uuid)

    Raises:
        WebSessionError: If credentials cannot be resolved
    """
    if token is None:
        token = get_access_token_from_keychain()
        if token is None:
            raise WebSessionError(
                "Could not retrieve access token from macOS keychain. "
                "Please provide --token manually."
            )

    if org_uuid is None:
        org_uuid = get_org_uuid_from_config()
        if org_uuid is None:
            raise WebSessionError(
                "Could not find organization UUID in ~/.claude.json. "
                "Please provide --org-uuid manually."
            )

    return token, org_uuid


def fetch_sessions(token: str, org_uuid: str) -> list[WebSession]:
    """Fetch list of sessions from the API.

    Args:
        token: API access token
        org_uuid: Organization UUID

    Returns:
        List of WebSession objects

    Raises:
        WebSessionError: If API request fails
    """
    headers = get_api_headers(token, org_uuid)
    try:
        response = httpx.get(
            f"{API_BASE_URL}/sessions",
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        sessions = []
        for item in data.get("data", []):
            sessions.append(
                WebSession(
                    id=item.get("id", ""),
                    title=item.get("title", "Untitled"),
                    created_at=item.get("created_at", ""),
                    project_path=item.get("project_path"),
                )
            )
        return sessions
    except httpx.HTTPError as e:
        raise WebSessionError(f"Failed to fetch sessions: {e}") from e


def fetch_session_data(token: str, org_uuid: str, session_id: str) -> dict[str, Any]:
    """Fetch a specific session's transcript data from the API.

    Args:
        token: API access token
        org_uuid: Organization UUID
        session_id: Session ID to fetch

    Returns:
        Session data as a dictionary containing the transcript

    Raises:
        WebSessionError: If API request fails
    """
    import logging
    logger = logging.getLogger(__name__)

    headers = get_api_headers(token, org_uuid)
    try:
        response = httpx.get(
            f"{API_BASE_URL}/session_ingress/session/{session_id}",
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        # Debug logging to inspect available fields
        logger.debug(f"Session {session_id} top-level keys: {sorted(data.keys())}")

        # Log git-related fields if present
        git_fields = ["repository_url", "github_url", "git_remote", "remote_url",
                      "repo_url", "git_url", "project_url"]
        found_git_fields = {k: data.get(k) for k in git_fields if k in data}
        if found_git_fields:
            logger.info(f"Found git-related fields in session {session_id}: {found_git_fields}")

        # Check for git info in nested structures
        if "project" in data and isinstance(data["project"], dict):
            logger.debug(f"Session has 'project' dict with keys: {sorted(data['project'].keys())}")
            project_git_fields = {k: data["project"].get(k) for k in git_fields if k in data["project"]}
            if project_git_fields:
                logger.info(f"Found git-related fields in project: {project_git_fields}")

        return data
    except httpx.HTTPError as e:
        raise WebSessionError(f"Failed to fetch session {session_id}: {e}") from e


def find_matching_local_project(session: WebSession) -> Path | None:
    """Find a local project that matches the web session.

    Checks if there's a local Claude Code project directory
    (~/.claude/projects/{encoded-path}) that matches this session.

    Args:
        session: Web session to match

    Returns:
        Path to local project if found, None otherwise
    """
    if not session.project_path:
        return None

    # Check if the project path exists locally
    project_path = Path(session.project_path)
    if not project_path.exists():
        return None

    # Check if there's a Claude Code project directory for it
    claude_projects_dir = Path.home() / ".claude" / "projects"
    if not claude_projects_dir.exists():
        return project_path if project_path.exists() else None

    # Encode the path Claude-style
    encoded_path = str(project_path.resolve()).replace("/", "-")
    project_dir = claude_projects_dir / encoded_path

    if project_dir.exists():
        return project_path

    return project_path if project_path.exists() else None


def session_to_jsonl_entries(session_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert web session API response to JSONL entries.

    The web API returns a different format than local JSONL files.
    This converts to the local format for compatibility with the
    existing claude_code plugin.

    Args:
        session_data: Raw session data from the API

    Returns:
        List of entries in the same format as local JSONL files
    """
    entries = []

    # The API returns loglines with message content
    loglines = session_data.get("loglines", [])
    session_id = session_data.get("id", "")
    cwd = session_data.get("project_path", "")

    for line in loglines:
        entry_type = line.get("type", "")
        if entry_type not in ("user", "assistant"):
            continue

        entry = {
            "type": entry_type,
            "timestamp": line.get("timestamp", ""),
            "sessionId": session_id,
            "cwd": cwd,
            "message": line.get("message", {}),
        }

        # Copy any additional fields
        if "toolUseResult" in line:
            entry["toolUseResult"] = line["toolUseResult"]
        if "isCompactSummary" in line:
            entry["isCompactSummary"] = line["isCompactSummary"]
        if "isMeta" in line:
            entry["isMeta"] = line["isMeta"]

        entries.append(entry)

    return entries


@dataclass
class DiscoveredWebSession:
    """A web session discovered via the API.

    Similar to DiscoveredTranscript but for web sessions that don't have a local path.
    """

    session_id: str
    title: str
    created_at: str
    project_path: Optional[str] = None
    project_name: Optional[str] = None

    @property
    def created_at_formatted(self) -> str:
        """Human-readable creation time."""
        try:
            dt = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            return self.created_at

    @property
    def display_name(self) -> str:
        """Get display name for the session."""
        if self.title:
            # Truncate long titles
            if len(self.title) > 50:
                return self.title[:47] + "..."
            return self.title
        return f"session-{self.session_id[:8]}"

    def matches_project(self, project_path: Path) -> bool:
        """Check if this session is for the given project path."""
        if not self.project_path:
            return False
        try:
            session_path = Path(self.project_path).resolve()
            check_path = project_path.resolve()
            return session_path == check_path
        except (ValueError, OSError):
            return False


def try_resolve_credentials() -> tuple[str, str] | None:
    """Try to resolve credentials, returning None if not available.

    This is a non-raising version of resolve_credentials for optional
    web session discovery.

    Returns:
        Tuple of (token, org_uuid) if available, None otherwise.
    """
    try:
        return resolve_credentials()
    except WebSessionError:
        return None


def discover_web_sessions(
    project_path: Path | None = None,
    token: str | None = None,
    org_uuid: str | None = None,
) -> list[DiscoveredWebSession]:
    """Discover web sessions, optionally filtered by project.

    This function attempts to fetch web sessions from the API. If credentials
    are not available or the request fails, it returns an empty list rather
    than raising an error.

    Args:
        project_path: If provided, only return sessions matching this project.
        token: API access token (optional, will auto-detect if possible)
        org_uuid: Organization UUID (optional, will auto-detect if possible)

    Returns:
        List of DiscoveredWebSession objects, sorted by creation time (newest first).
    """
    # Try to get credentials
    if token is None or org_uuid is None:
        creds = try_resolve_credentials()
        if creds is None:
            return []
        token, org_uuid = creds

    # Fetch sessions from API
    try:
        sessions = fetch_sessions(token, org_uuid)
    except WebSessionError:
        return []

    # Convert to DiscoveredWebSession and optionally filter
    discovered = []
    for session in sessions:
        # Extract project name from path
        project_name = None
        if session.project_path:
            project_name = Path(session.project_path).name

        web_session = DiscoveredWebSession(
            session_id=session.id,
            title=session.title,
            created_at=session.created_at,
            project_path=session.project_path,
            project_name=project_name,
        )

        # Filter by project if specified
        if project_path is not None:
            if not web_session.matches_project(project_path):
                continue

        discovered.append(web_session)

    # Sort by creation time, newest first
    discovered.sort(key=lambda s: s.created_at, reverse=True)
    return discovered
