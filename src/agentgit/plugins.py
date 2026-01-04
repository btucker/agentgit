"""Pluggy hookspecs for agentgit transcript parsing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pluggy

from agentgit.settings import CONFIG_PATH, get_config, save_config

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

    # Enhancement hooks
    @hookspec
    def agentgit_get_enhancer_info(self) -> dict[str, str] | None:
        """Return enhancer plugin identification info.

        Returns:
            Dict with 'name' (short identifier like 'llm' or 'rules') and
            'description' (human-readable description), or None.
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

    @hookspec(firstresult=True)
    def agentgit_generate_session_name(
        self,
        prompt_responses: list[PromptResponse],
        enhancer: str,
        model: str | None,
    ) -> str | None:
        """Generate a descriptive name for a coding session.

        Used to create meaningful git branch names for sessions.

        Args:
            prompt_responses: All prompts and responses in the session.
            enhancer: The enhancer type to use.
            model: Optional model override.

        Returns:
            A descriptive session name (will be sanitized for git), or None.
        """


def get_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager."""
    pm = pluggy.PluginManager("agentgit")
    pm.add_hookspecs(AgentGitSpec)
    return pm


def register_builtin_plugins(pm: pluggy.PluginManager) -> None:
    """Register the built-in format plugins."""
    from agentgit.formats.claude_code import ClaudeCodePlugin
    from agentgit.formats.claude_code_web import ClaudeCodeWebPlugin
    from agentgit.formats.codex import CodexPlugin

    pm.register(ClaudeCodePlugin())
    pm.register(ClaudeCodeWebPlugin())
    pm.register(CodexPlugin())


def load_external_plugins(pm: pluggy.PluginManager) -> int:
    """Load external plugins from entry points.

    Discovers pip-installed plugins that declare [project.entry-points."agentgit"].

    Returns:
        Number of external plugins loaded.
    """
    # Load from setuptools entry points (pip-installed plugins)
    # We manually load and instantiate to ensure we get instances, not classes
    return load_plugins_from_entry_points(pm)


def load_plugins_from_entry_points(pm: pluggy.PluginManager) -> int:
    """Load plugins from setuptools entry points.

    Handles both class entry points (instantiates them) and
    instance/factory entry points.

    Returns:
        Number of plugins loaded.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        # Python < 3.9 fallback
        from importlib_metadata import entry_points  # type: ignore

    count = 0

    # Get entry points for the 'agentgit' group
    try:
        # Python 3.10+ / importlib_metadata 3.6+
        eps = entry_points(group="agentgit")
    except TypeError:
        # Older API
        eps = entry_points().get("agentgit", [])

    for ep in eps:
        try:
            plugin_obj = ep.load()

            # If it's a class, instantiate it
            if isinstance(plugin_obj, type):
                plugin_obj = plugin_obj()

            # Check if already registered (by name if available)
            if not pm.is_registered(plugin_obj):
                pm.register(plugin_obj, name=ep.name)
                count += 1
        except Exception:
            # Skip plugins that fail to load
            pass

    return count


def get_plugins_config() -> dict[str, Any]:
    """Get the plugins section of the configuration.

    Returns:
        The plugins config dict with "packages" key.
    """
    config = get_config()
    return config.get("plugins", {"packages": []})


def save_plugins_config(plugins_config: dict[str, Any]) -> None:
    """Save the plugins section of the configuration.

    Args:
        plugins_config: The plugins config dict to save.
    """
    config = get_config()
    config["plugins"] = plugins_config
    save_config(config)


def get_pip_command() -> list[str]:
    """Get the pip command to use for installing packages.

    Prefers uv if available, falls back to pip.

    Returns:
        Command list like ["uv", "pip"] or ["pip"].
    """
    if shutil.which("uv"):
        return ["uv", "pip"]
    return ["pip"]


def install_plugin_package(package: str) -> None:
    """Install a plugin package using pip/uv.

    Args:
        package: Package name to install (e.g., "agentgit-aider").

    Raises:
        subprocess.CalledProcessError: If installation fails.
    """
    pip_cmd = get_pip_command()
    subprocess.run(
        [*pip_cmd, "install", package],
        check=True,
        capture_output=True,
        text=True,
    )


def uninstall_plugin_package(package: str) -> None:
    """Uninstall a plugin package using pip/uv.

    Args:
        package: Package name to uninstall.

    Raises:
        subprocess.CalledProcessError: If uninstallation fails.
    """
    pip_cmd = get_pip_command()
    subprocess.run(
        [*pip_cmd, "uninstall", "-y", package],
        check=True,
        capture_output=True,
        text=True,
    )


def add_plugin(package: str) -> tuple[bool, str]:
    """Install and register a plugin package.

    Args:
        package: Package name to install (e.g., "agentgit-aider").

    Returns:
        Tuple of (success, message).
    """
    config = get_plugins_config()

    # Check if already registered
    if package in config.get("packages", []):
        return False, f"Package '{package}' is already registered"

    # Install the package
    try:
        install_plugin_package(package)
    except subprocess.CalledProcessError as e:
        return False, f"Failed to install '{package}': {e.stderr or e.stdout or str(e)}"

    # Register in config
    if "packages" not in config:
        config["packages"] = []
    config["packages"].append(package)
    save_plugins_config(config)

    # Clear cached plugin manager
    global _configured_plugin_manager
    _configured_plugin_manager = None

    return True, f"Installed and registered '{package}'"


def remove_plugin(package: str) -> tuple[bool, str]:
    """Uninstall and unregister a plugin package.

    Args:
        package: Package name to remove.

    Returns:
        Tuple of (success, message).
    """
    config = get_plugins_config()
    packages = config.get("packages", [])

    if package not in packages:
        return False, f"Package '{package}' is not registered"

    # Uninstall the package
    try:
        uninstall_plugin_package(package)
    except subprocess.CalledProcessError as e:
        # Continue with unregistration even if uninstall fails
        # (package might have been manually removed)
        pass

    # Unregister from config
    packages.remove(package)
    config["packages"] = packages
    save_plugins_config(config)

    # Clear cached plugin manager
    global _configured_plugin_manager
    _configured_plugin_manager = None

    return True, f"Removed '{package}'"


def list_configured_plugins() -> list[dict[str, str]]:
    """List all configured plugins with their sources.

    Returns:
        List of dicts with 'name', 'description', 'source', 'package' keys.
    """
    pm = get_configured_plugin_manager()
    config = get_plugins_config()
    installed_packages = set(config.get("packages", []))

    plugins = []
    for info in pm.hook.agentgit_get_plugin_info():
        if info:
            name = info.get("name", "unknown")
            # Determine source based on whether it's a builtin
            if name in ("claude_code", "codex"):
                source = "builtin"
                package = None
            else:
                source = "pip"
                # Try to find matching package name
                package = None
                for pkg in installed_packages:
                    if name.replace("_", "-") in pkg or name in pkg:
                        package = pkg
                        break

            plugins.append({
                "name": name,
                "description": info.get("description", ""),
                "source": source,
                "package": package,
            })

    return plugins


def list_installed_packages() -> list[str]:
    """List packages installed via agentgit agents add.

    Returns:
        List of package names.
    """
    config = get_plugins_config()
    return config.get("packages", [])


def register_enhancer_plugins(pm: pluggy.PluginManager) -> None:
    """Register the built-in AI enhancer plugins."""
    from agentgit.enhancers.llm import LLMEnhancerPlugin
    from agentgit.enhancers.rules import RulesEnhancerPlugin

    pm.register(RulesEnhancerPlugin())
    pm.register(LLMEnhancerPlugin())


_configured_plugin_manager: pluggy.PluginManager | None = None


def get_configured_plugin_manager() -> pluggy.PluginManager:
    """Get a plugin manager with all plugins registered.

    Loads plugins from:
    1. Built-in plugins (claude_code, codex)
    2. Pip-installed plugins (via entry points)
    3. Config file plugins (~/.agentgit/config.yml)

    This function caches the plugin manager for efficiency, so repeated
    calls return the same instance.

    Returns:
        A configured PluginManager with all plugins registered.
    """
    global _configured_plugin_manager
    if _configured_plugin_manager is None:
        _configured_plugin_manager = get_plugin_manager()
        register_builtin_plugins(_configured_plugin_manager)
        register_enhancer_plugins(_configured_plugin_manager)
        load_external_plugins(_configured_plugin_manager)
    return _configured_plugin_manager


def reset_plugin_manager() -> None:
    """Reset the cached plugin manager.

    Call this to force reloading of plugins.
    """
    global _configured_plugin_manager
    _configured_plugin_manager = None
