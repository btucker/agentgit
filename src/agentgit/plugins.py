"""Pluggy hookspecs for agentgit transcript parsing."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pluggy

if TYPE_CHECKING:
    from agentgit.core import FileOperation, PromptResponse, Transcript

hookspec = pluggy.HookspecMarker("agentgit")
hookimpl = pluggy.HookimplMarker("agentgit")

# Config file for external plugins
PLUGINS_CONFIG_PATH = Path.home() / ".config" / "agentgit" / "plugins.json"


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


def load_external_plugins(pm: pluggy.PluginManager) -> int:
    """Load external plugins from entry points and config file.

    Returns:
        Number of external plugins loaded.
    """
    count = 0

    # Load from setuptools entry points (pip-installed plugins)
    # This discovers packages that declare [project.entry-points."agentgit"]
    # We manually load and instantiate to ensure we get instances, not classes
    count += load_plugins_from_entry_points(pm)

    # Load from config file
    count += load_plugins_from_config(pm)

    return count


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


def load_plugins_from_config(pm: pluggy.PluginManager) -> int:
    """Load plugins from the config file.

    Config file format (~/.config/agentgit/plugins.json):
    {
        "plugins": [
            "package.module:PluginClass",
            "/path/to/plugin.py:PluginClass"
        ]
    }

    Returns:
        Number of plugins loaded.
    """
    if not PLUGINS_CONFIG_PATH.exists():
        return 0

    try:
        config = json.loads(PLUGINS_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return 0

    plugins = config.get("plugins", [])
    count = 0

    for plugin_spec in plugins:
        try:
            plugin = load_plugin_from_spec(plugin_spec)
            if plugin and not pm.is_registered(plugin):
                pm.register(plugin)
                count += 1
        except Exception:
            # Skip plugins that fail to load
            pass

    return count


def load_plugin_from_spec(spec: str) -> Any:
    """Load a plugin from a specification string.

    Args:
        spec: Either "package.module:ClassName" or "/path/to/file.py:ClassName"

    Returns:
        Instantiated plugin object.
    """
    if ":" not in spec:
        raise ValueError(f"Invalid plugin spec '{spec}': must contain ':'")

    module_path, class_name = spec.rsplit(":", 1)

    if module_path.endswith(".py"):
        # Load from file path
        import importlib.util

        file_path = Path(module_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Plugin file not found: {file_path}")

        spec_obj = importlib.util.spec_from_file_location(
            file_path.stem, file_path
        )
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        # Load from installed package
        module = importlib.import_module(module_path)

    plugin_class = getattr(module, class_name)
    return plugin_class()


def get_plugins_config() -> dict[str, Any]:
    """Get the current plugins configuration.

    Returns:
        The plugins config dict, or empty dict with "plugins" key if not exists.
    """
    if not PLUGINS_CONFIG_PATH.exists():
        return {"plugins": []}

    try:
        return json.loads(PLUGINS_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"plugins": []}


def save_plugins_config(config: dict[str, Any]) -> None:
    """Save the plugins configuration.

    Args:
        config: The config dict to save.
    """
    PLUGINS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLUGINS_CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def add_plugin_to_config(spec: str) -> bool:
    """Add a plugin specification to the config file.

    Args:
        spec: Plugin spec like "package.module:ClassName" or "/path/to/file.py:ClassName"

    Returns:
        True if added, False if already exists.
    """
    # Validate the spec first
    load_plugin_from_spec(spec)  # Raises if invalid

    config = get_plugins_config()
    if spec in config["plugins"]:
        return False

    config["plugins"].append(spec)
    save_plugins_config(config)

    # Clear the cached plugin manager so it reloads
    global _configured_plugin_manager
    _configured_plugin_manager = None

    return True


def remove_plugin_from_config(name_or_spec: str) -> bool:
    """Remove a plugin from the config file.

    Args:
        name_or_spec: Either the plugin name or full spec.

    Returns:
        True if removed, False if not found.
    """
    config = get_plugins_config()
    original_count = len(config["plugins"])

    # Try exact match first
    if name_or_spec in config["plugins"]:
        config["plugins"].remove(name_or_spec)
    else:
        # Try matching by plugin name (class name portion)
        config["plugins"] = [
            p for p in config["plugins"]
            if not p.endswith(f":{name_or_spec}") and
               not p.rsplit(":", 1)[-1].lower().replace("plugin", "") == name_or_spec.lower()
        ]

    if len(config["plugins"]) < original_count:
        save_plugins_config(config)

        # Clear the cached plugin manager
        global _configured_plugin_manager
        _configured_plugin_manager = None

        return True

    return False


def list_configured_plugins() -> list[dict[str, str]]:
    """List all configured plugins with their sources.

    Returns:
        List of dicts with 'name', 'description', 'source' keys.
    """
    pm = get_configured_plugin_manager()
    plugins = []

    for info in pm.hook.agentgit_get_plugin_info():
        if info:
            plugins.append({
                "name": info.get("name", "unknown"),
                "description": info.get("description", ""),
                "source": "builtin",  # Will be updated below
            })

    # Identify external plugins from config
    config = get_plugins_config()
    for spec in config.get("plugins", []):
        try:
            class_name = spec.rsplit(":", 1)[-1]
            # Find matching plugin and mark as config
            for p in plugins:
                if class_name.lower().replace("plugin", "") in p["name"].lower():
                    p["source"] = "config"
                    break
        except Exception:
            pass

    return plugins


_configured_plugin_manager: pluggy.PluginManager | None = None


def get_configured_plugin_manager() -> pluggy.PluginManager:
    """Get a plugin manager with all plugins registered.

    Loads plugins from:
    1. Built-in plugins (claude_code, codex)
    2. Pip-installed plugins (via entry points)
    3. Config file plugins (~/.config/agentgit/plugins.json)

    This function caches the plugin manager for efficiency, so repeated
    calls return the same instance.

    Returns:
        A configured PluginManager with all plugins registered.
    """
    global _configured_plugin_manager
    if _configured_plugin_manager is None:
        _configured_plugin_manager = get_plugin_manager()
        register_builtin_plugins(_configured_plugin_manager)
        load_external_plugins(_configured_plugin_manager)
    return _configured_plugin_manager


def reset_plugin_manager() -> None:
    """Reset the cached plugin manager.

    Call this to force reloading of plugins.
    """
    global _configured_plugin_manager
    _configured_plugin_manager = None
