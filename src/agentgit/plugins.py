"""Pluggy hookspecs for agentgit transcript parsing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt, PromptResponse, Transcript

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

    # AI Enhancement hooks
    @hookspec
    def agentgit_get_ai_enhancer_info(self) -> dict[str, str] | None:
        """Return AI enhancer plugin identification info.

        Returns:
            Dict with 'name' (short identifier like 'claude_code') and
            'description' (human-readable description), or None.
        """

    @hookspec(firstresult=True)
    def agentgit_enhance_operation_summary(
        self,
        operation: FileOperation,
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an enhanced summary for a file operation entry.

        Args:
            operation: The file operation to generate a summary for.
            enhancer: The enhancer type to use (e.g., 'llm', 'rules').
            model: Optional model override (e.g., 'claude-cli-haiku').

        Returns:
            Generated entry summary, or None if generation fails.
        """

    @hookspec(firstresult=True)
    def agentgit_enhance_turn_summary(
        self,
        turn: AssistantTurn,
        prompt: Prompt | None,
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an enhanced summary for an assistant turn entry.

        Args:
            turn: The assistant turn containing grouped operations.
            prompt: Optional user prompt that triggered this turn.
            enhancer: The enhancer type to use (e.g., 'llm', 'rules').
            model: Optional model override.

        Returns:
            Generated entry summary, or None if generation fails.
        """

    @hookspec(firstresult=True)
    def agentgit_enhance_prompt_summary(
        self,
        prompt: Prompt,
        turns: list[AssistantTurn],
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate an enhanced summary for a user prompt entry.

        Args:
            prompt: The user prompt.
            turns: All assistant turns that responded to the prompt.
            enhancer: The enhancer type to use (e.g., 'llm', 'rules').
            model: Optional model override.

        Returns:
            Generated entry summary, or None if generation fails.
        """

    @hookspec(firstresult=True)
    def agentgit_curate_turn_context(
        self,
        turn: AssistantTurn,
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Curate the context/reasoning to include in a turn commit body.

        The enhancer can select and organize the most relevant parts of the
        assistant's thinking to include in the commit message body.

        Args:
            turn: The assistant turn to curate context for.
            enhancer: The enhancer type to use.
            model: Optional model override.

        Returns:
            Curated context string to include in commit body, or None to use default.
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


def register_enhancer_plugins(pm: pluggy.PluginManager) -> None:
    """Register the built-in AI enhancer plugins."""
    from agentgit.enhancers.llm import LLMEnhancerPlugin
    from agentgit.enhancers.rules import RulesEnhancerPlugin

    pm.register(RulesEnhancerPlugin())
    pm.register(LLMEnhancerPlugin())


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
        register_enhancer_plugins(_configured_plugin_manager)
    return _configured_plugin_manager
