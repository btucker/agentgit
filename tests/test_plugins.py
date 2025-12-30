"""Tests for agentgit.plugins module."""

import pytest

from agentgit.plugins import (
    get_configured_plugin_manager,
    get_plugin_manager,
    register_builtin_plugins,
)


class TestPluginManager:
    """Tests for plugin manager."""

    def test_get_plugin_manager_returns_manager(self):
        """Should return a PluginManager instance."""
        pm = get_plugin_manager()
        assert pm is not None
        assert pm.project_name == "agentgit"

    def test_register_builtin_plugins(self):
        """Should register the Claude Code plugin."""
        pm = get_plugin_manager()
        register_builtin_plugins(pm)

        # Check that at least one plugin is registered
        plugins = pm.get_plugins()
        assert len(plugins) > 0

    def test_hookspecs_are_registered(self):
        """Should have hookspecs registered."""
        pm = get_plugin_manager()

        # Check that hooks are available
        assert hasattr(pm.hook, "agentgit_detect_format")
        assert hasattr(pm.hook, "agentgit_parse_transcript")
        assert hasattr(pm.hook, "agentgit_extract_operations")
        assert hasattr(pm.hook, "agentgit_enrich_operation")

    def test_get_configured_plugin_manager(self):
        """Should return a plugin manager with plugins already registered."""
        pm = get_configured_plugin_manager()
        assert pm is not None

        # Should have plugins registered
        plugins = pm.get_plugins()
        assert len(plugins) > 0

        # Should have Claude Code plugin
        has_claude = False
        for info in pm.hook.agentgit_get_plugin_info():
            if info and info.get("name") == "claude_code":
                has_claude = True
                break
        assert has_claude

    def test_get_configured_plugin_manager_is_cached(self):
        """Should return the same instance on repeated calls."""
        pm1 = get_configured_plugin_manager()
        pm2 = get_configured_plugin_manager()
        assert pm1 is pm2
