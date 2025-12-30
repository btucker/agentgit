"""Pluggy hookspecs for agentgit transcript parsing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from typing import Any

    from agentgit.core import DiscoveredWebSession, FileOperation, PromptResponse, Transcript

hookspec = pluggy.HookspecMarker("agentgit")
hookimpl = pluggy.HookimplMarker("agentgit")


class AgentGitSpec:
    """Hook specifications for agentgit plugins."""

    @hookspec
    def agentgit_get_plugin_info(self) -> dict[str, str] | None:
        """Return plugin identification info.

        Returns:
            Dict with 'name' (short identifier like 'claude_code') and
            'description' (human-readable description), or None.
        """

    @hookspec(firstresult=True)
    def agentgit_detect_format(self, path: Path) -> str | None:
        """Detect the format of a transcript file.

        Args:
            path: Path to the transcript file.

        Returns:
            Format identifier string (e.g., "claude_code_jsonl") if detected,
            None if this plugin cannot handle the file.
        """

    @hookspec(firstresult=True)
    def agentgit_parse_transcript(self, path: Path, format: str) -> Transcript | None:
        """Parse a transcript file into structured data.

        Args:
            path: Path to the transcript file.
            format: Format identifier from agentgit_detect_format.

        Returns:
            Parsed Transcript object, or None if this plugin doesn't handle
            the specified format.
        """

    @hookspec
    def agentgit_extract_operations(self, transcript: Transcript) -> list[FileOperation]:
        """Extract file operations from a transcript.

        Called after parsing to extract Write/Edit/Delete operations.
        Multiple plugins can contribute operations.

        Args:
            transcript: The parsed transcript.

        Returns:
            List of FileOperation objects found in the transcript.
        """

    @hookspec
    def agentgit_enrich_operation(
        self,
        operation: FileOperation,
        transcript: Transcript,
    ) -> FileOperation:
        """Enrich a file operation with additional metadata.

        Called for each operation to add prompt context, assistant reasoning, etc.

        Args:
            operation: The file operation to enrich.
            transcript: The full transcript for context lookup.

        Returns:
            The enriched FileOperation (may be same object, modified in place).
        """

    @hookspec
    def agentgit_build_prompt_responses(
        self, transcript: Transcript
    ) -> list[PromptResponse]:
        """Build the prompt-response structure from a transcript.

        Groups operations by assistant message and organizes them under their
        triggering prompts. This structure is used for creating the merge-based
        git history.

        Args:
            transcript: The parsed transcript with operations.

        Returns:
            List of PromptResponse objects containing grouped AssistantTurns.
        """

    @hookspec
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Discover transcript files for a project.

        Called when no transcript is explicitly provided to find relevant
        transcripts based on the project path.

        Args:
            project_path: Path to the project (usually a git repo root).
                If None, returns all transcripts from all projects.

        Returns:
            List of paths to transcript files found for this project.
        """

    @hookspec(firstresult=True)
    def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
        """Get the project name/identifier that a transcript corresponds to.

        Plugins can decode the project name from transcript location.
        For example, Claude Code stores transcripts in:
        ~/.claude/projects/-Users-name-project/session.jsonl
        which corresponds to project name "-Users-name-project".

        Args:
            transcript_path: Path to the transcript file.

        Returns:
            The project name/identifier, or None if this plugin can't determine it.
        """

    @hookspec(firstresult=True)
    def agentgit_get_display_name(self, transcript_path: Path) -> str | None:
        """Get a human-readable display name for a transcript file.

        Plugins should return a concise, meaningful name for the transcript.
        This is used in the discover command's table display.

        Args:
            transcript_path: Path to the transcript file.

        Returns:
            A display name string, or None if this plugin can't provide one.
        """

    # Web session hooks for cloud-based transcript sources

    @hookspec(firstresult=True)
    def agentgit_resolve_web_credentials(
        self,
        token: str | None,
        org_uuid: str | None,
    ) -> tuple[str, str] | None:
        """Resolve credentials for web session access.

        Attempts to auto-detect credentials from system sources (keychain,
        config files) or uses explicitly provided values.

        Args:
            token: Optional API access token (if provided explicitly).
            org_uuid: Optional organization UUID (if provided explicitly).

        Returns:
            Tuple of (token, org_uuid) if credentials can be resolved,
            None if this plugin cannot provide credentials.

        Raises:
            ValueError: If credentials are required but cannot be resolved.
        """

    @hookspec
    def agentgit_discover_web_sessions(
        self,
        project_path: Path | None,
        credentials: tuple[str, str],
    ) -> list[DiscoveredWebSession]:
        """Discover web sessions from a cloud API.

        Similar to agentgit_discover_transcripts, but for remote sessions
        that need to be fetched from an API.

        Args:
            project_path: Path to the project to filter by (optional).
                If None, returns all available web sessions.
            credentials: Tuple of (token, org_uuid) for API authentication.

        Returns:
            List of DiscoveredWebSession objects.
        """

    @hookspec(firstresult=True)
    def agentgit_fetch_web_session(
        self,
        session_id: str,
        credentials: tuple[str, str],
    ) -> list[dict[str, Any]] | None:
        """Fetch a web session and convert to JSONL-compatible entries.

        Args:
            session_id: The session ID to fetch.
            credentials: Tuple of (token, org_uuid) for API authentication.

        Returns:
            List of JSONL entry dicts (same format as local transcripts),
            or None if this plugin cannot handle the session.
        """


def get_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager."""
    pm = pluggy.PluginManager("agentgit")
    pm.add_hookspecs(AgentGitSpec)
    return pm


def register_builtin_plugins(pm: pluggy.PluginManager) -> None:
    """Register the built-in format plugins."""
    from agentgit.formats.claude_code import ClaudeCodePlugin
    from agentgit.formats.codex import CodexPlugin

    pm.register(ClaudeCodePlugin())
    pm.register(CodexPlugin())


_configured_plugin_manager: pluggy.PluginManager | None = None


def get_configured_plugin_manager() -> pluggy.PluginManager:
    """Get a plugin manager with builtin plugins already registered.

    This function caches the plugin manager for efficiency, so repeated
    calls return the same instance.

    Returns:
        A configured PluginManager with builtin plugins registered.
    """
    global _configured_plugin_manager
    if _configured_plugin_manager is None:
        _configured_plugin_manager = get_plugin_manager()
        register_builtin_plugins(_configured_plugin_manager)
    return _configured_plugin_manager
