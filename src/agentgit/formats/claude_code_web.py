"""Claude Code Web Sessions plugin for agentgit.

This plugin discovers and caches web sessions from the Claude API,
making them available through the standard plugin discovery interface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import httpx

from agentgit.plugins import hookimpl
from agentgit.utils import get_git_remotes, normalize_git_url

if TYPE_CHECKING:
    from agentgit.core import Transcript

# API Configuration
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


# Data classes and exceptions
@dataclass
class WebSession:
    """Represents a Claude Code web session."""

    id: str
    title: str
    created_at: str
    project_path: str | None = None
    git_url: str | None = None  # Git repository URL from session_context.sources

    def matches_project(self, project_path: Path) -> bool:
        """Check if this session is for the given project.

        Matches by comparing the session's git URL against all git remotes
        in the local project. Falls back to path comparison if no git info.

        Args:
            project_path: Path to the local project directory

        Returns:
            True if the session matches this project
        """
        # Try git URL matching first (most reliable)
        if self.git_url:
            local_remotes = get_git_remotes(project_path)
            if local_remotes:
                normalized_session = normalize_git_url(self.git_url)
                for remote_url in local_remotes:
                    if normalize_git_url(remote_url) == normalized_session:
                        return True

        # Fall back to path comparison (legacy/local sessions)
        if self.project_path:
            session_path = Path(self.project_path).resolve()
            check_path = project_path.resolve()
            return session_path == check_path

        return False


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


class WebSessionError(Exception):
    """Error fetching or processing web sessions."""

    pass


# Authentication and credential management
def get_api_headers(token: str, org_uuid: str) -> dict[str, str]:
    """Build API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "x-organization-uuid": org_uuid,
    }


def get_access_token_from_keychain() -> str | None:
    """Get access token from Claude Code credentials.

    Checks two locations:
    1. ~/.claude/.credentials.json (current Claude Code format)
    2. macOS keychain (legacy format)

    Returns None if credentials not found.
    """
    # First, try the credentials file (current format)
    creds_file = Path.home() / ".claude" / ".credentials.json"
    if creds_file.exists():
        try:
            with open(creds_file) as f:
                creds = json.load(f)
            token = creds.get("claudeAiOauth", {}).get("accessToken")
            if token:
                return token
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to macOS keychain (legacy format)
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


# API functions
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
            # Extract git URL from session_context.sources
            git_url = None
            session_context = item.get("session_context", {})
            sources = session_context.get("sources", [])
            for source in sources:
                if source.get("type") == "git_repository":
                    git_url = source.get("url")
                    break

            sessions.append(
                WebSession(
                    id=item.get("id", ""),
                    title=item.get("title", "Untitled"),
                    created_at=item.get("created_at", ""),
                    project_path=item.get("project_path"),
                    git_url=git_url,
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
        git_fields = [
            "repository_url",
            "github_url",
            "git_remote",
            "remote_url",
            "repo_url",
            "git_url",
            "project_url",
        ]
        found_git_fields = {k: data.get(k) for k in git_fields if k in data}
        if found_git_fields:
            logger.info(
                f"Found git-related fields in session {session_id}: {found_git_fields}"
            )

        # Check for git info in nested structures
        if "project" in data and isinstance(data["project"], dict):
            logger.debug(
                f"Session has 'project' dict with keys: {sorted(data['project'].keys())}"
            )
            project_git_fields = {
                k: data["project"].get(k) for k in git_fields if k in data["project"]
            }
            if project_git_fields:
                logger.info(f"Found git-related fields in project: {project_git_fields}")

        return data
    except httpx.HTTPError as e:
        raise WebSessionError(f"Failed to fetch session {session_id}: {e}") from e


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

    # Filter by project and convert to DiscoveredWebSession
    discovered = []
    for session in sessions:
        # Filter by project if specified (using WebSession's git-aware matching)
        if project_path is not None:
            if not session.matches_project(project_path):
                continue

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

        discovered.append(web_session)

    # Sort by creation time, newest first
    discovered.sort(key=lambda s: s.created_at, reverse=True)
    return discovered

# Format identifier - different from local to distinguish in UI
FORMAT_CLAUDE_CODE_WEB = "claude_code_web_jsonl"

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

    Inherits from ClaudeCodePlugin since web sessions use the same JSONL format.
    Only overrides hooks for format detection, discovery, and display.
    """

    def __init__(self):
        """Initialize the web plugin with a base claude_code plugin for delegation."""
        from agentgit.formats.claude_code import ClaudeCodePlugin
        self._base_plugin = ClaudeCodePlugin()

    @hookimpl
    def agentgit_get_plugin_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": "claude_code_web",
            "description": "Claude Code web sessions (cloud API)",
        }

    @hookimpl
    def agentgit_detect_format(self, path: Path) -> str | None:
        """Detect if this is a cached web session.

        Returns the format type for cached web sessions so they can be
        distinguished from local sessions in the UI.
        """
        # Only handle files in our cache directory
        try:
            path.resolve().relative_to(WEB_SESSION_CACHE_DIR.resolve())
        except ValueError:
            return None

        # Skip metadata files
        if path.suffix != ".jsonl":
            return None

        return FORMAT_CLAUDE_CODE_WEB

    @hookimpl
    def agentgit_parse_transcript(self, path: Path, format: str) -> "Transcript | None":
        """Parse a cached web session file.

        Delegates to base plugin since web sessions use the same JSONL format.
        """
        if format != FORMAT_CLAUDE_CODE_WEB:
            return None

        transcript = self._base_plugin.agentgit_parse_transcript(path, "claude_code_jsonl")
        if transcript:
            transcript.source_format = FORMAT_CLAUDE_CODE_WEB
            # Web sessions don't have sessionId in entries, use filename
            if not transcript.session_id:
                transcript.session_id = path.stem
        return transcript

    @hookimpl
    def agentgit_extract_operations(self, transcript: "Transcript") -> list:
        """Extract operations - delegates to base plugin."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_WEB:
            return []
        return self._delegate_with_format(
            lambda: self._base_plugin.agentgit_extract_operations(transcript),
            transcript
        )

    @hookimpl
    def agentgit_enrich_operation(self, operation, transcript: "Transcript"):
        """Enrich operations - delegates to base plugin."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_WEB:
            return
        return self._delegate_with_format(
            lambda: self._base_plugin.agentgit_enrich_operation(operation, transcript),
            transcript
        )

    @hookimpl
    def agentgit_build_prompt_responses(self, transcript: "Transcript") -> list:
        """Build prompt responses - delegates to base plugin."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_WEB:
            return []
        return self._delegate_with_format(
            lambda: self._base_plugin.agentgit_build_prompt_responses(transcript),
            transcript
        )

    @hookimpl
    def agentgit_build_conversation_rounds(self, transcript: "Transcript") -> list:
        """Build conversation rounds - delegates to base plugin."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_WEB:
            return []
        return self._delegate_with_format(
            lambda: self._base_plugin.agentgit_build_conversation_rounds(transcript),
            transcript
        )

    def _delegate_with_format(self, func, transcript: "Transcript"):
        """Helper to temporarily change format for delegation."""
        original_format = transcript.source_format
        transcript.source_format = "claude_code_jsonl"
        try:
            return func()
        finally:
            transcript.source_format = original_format

    @hookimpl
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Discover Claude Code web sessions.

        Fetches available web sessions from the API and caches them locally.
        Returns paths to the cached session files.

        Sessions are matched to projects by comparing git remote URLs.

        Args:
            project_path: Path to the project to filter sessions.

        Returns:
            List of paths to cached web session files.
        """
        # Discover web sessions filtered by project (matches via git remotes)
        sessions = discover_web_sessions(project_path=project_path)

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

    @hookimpl
    def agentgit_get_last_timestamp(self, transcript_path: Path) -> float | None:
        """Get the last timestamp from a cached web session.

        Reads the end of the JSONL file to find the last entry's timestamp
        instead of using file modification time.
        """
        # Only handle files in our cache directory
        try:
            transcript_path.resolve().relative_to(WEB_SESSION_CACHE_DIR.resolve())
        except ValueError:
            return None

        # Skip metadata files
        if transcript_path.suffix != ".jsonl":
            return None

        from agentgit.formats.claude_code import get_last_timestamp_from_jsonl
        return get_last_timestamp_from_jsonl(transcript_path)


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
