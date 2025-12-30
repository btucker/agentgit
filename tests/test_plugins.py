"""Tests for agentgit.plugins module."""

import pytest

from agentgit.plugins import get_plugin_manager, register_builtin_plugins


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
