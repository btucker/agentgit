"""Tests for agentgit.plugins module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agentgit.plugins import (
    add_plugin,
    get_configured_plugin_manager,
    get_pip_command,
    get_plugin_manager,
    get_plugins_config,
    install_plugin_package,
    list_configured_plugins,
    list_installed_packages,
    load_plugins_from_entry_points,
    register_builtin_plugins,
    remove_plugin,
    reset_plugin_manager,
    save_plugins_config,
    uninstall_plugin_package,
)
from agentgit.settings import CONFIG_PATH, get_config, save_config


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

    def test_reset_plugin_manager(self):
        """Should allow resetting the cached plugin manager."""
        pm1 = get_configured_plugin_manager()
        reset_plugin_manager()
        pm2 = get_configured_plugin_manager()
        # After reset, should get a new instance
        assert pm1 is not pm2


class TestPluginsConfig:
    """Tests for plugin configuration management."""

    def test_get_config_empty(self, tmp_path, monkeypatch):
        """Should return empty dict when config doesn't exist."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        config = get_config()
        assert config == {}

    def test_get_config_existing(self, tmp_path, monkeypatch):
        """Should return config from existing file."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("plugins:\n  packages:\n    - agentgit-test\n")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        config = get_config()
        assert config == {"plugins": {"packages": ["agentgit-test"]}}

    def test_get_config_invalid_yaml(self, tmp_path, monkeypatch):
        """Should return empty config on invalid YAML."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("invalid: yaml: content:")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        config = get_config()
        assert config == {}

    def test_save_config(self, tmp_path, monkeypatch):
        """Should save config to file."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        save_config({"plugins": {"packages": ["test-pkg"]}})

        assert fake_config.exists()
        content = yaml.safe_load(fake_config.read_text())
        assert content == {"plugins": {"packages": ["test-pkg"]}}

    def test_get_plugins_config_empty(self, tmp_path, monkeypatch):
        """Should return empty packages list when config doesn't exist."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        config = get_plugins_config()
        assert config == {"packages": []}

    def test_get_plugins_config_existing(self, tmp_path, monkeypatch):
        """Should return plugins section from config."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("plugins:\n  packages:\n    - agentgit-test\n")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        config = get_plugins_config()
        assert config == {"packages": ["agentgit-test"]}

    def test_save_plugins_config(self, tmp_path, monkeypatch):
        """Should save plugins section to config."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        save_plugins_config({"packages": ["test-pkg"]})

        assert fake_config.exists()
        content = yaml.safe_load(fake_config.read_text())
        assert content == {"plugins": {"packages": ["test-pkg"]}}


class TestPipCommands:
    """Tests for pip/uv package management."""

    def test_get_pip_command_with_uv(self, monkeypatch):
        """Should prefer uv when available."""
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/uv" if x == "uv" else None)
        assert get_pip_command() == ["uv", "pip"]

    def test_get_pip_command_without_uv(self, monkeypatch):
        """Should fall back to pip when uv not available."""
        monkeypatch.setattr("shutil.which", lambda x: None)
        assert get_pip_command() == ["pip"]

    def test_install_plugin_package(self, monkeypatch):
        """Should run pip install command."""
        mock_run = MagicMock()
        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", lambda x: None)  # Use pip

        install_plugin_package("test-package")

        mock_run.assert_called_once_with(
            ["pip", "install", "test-package"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_uninstall_plugin_package(self, monkeypatch):
        """Should run pip uninstall command."""
        mock_run = MagicMock()
        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", lambda x: None)  # Use pip

        uninstall_plugin_package("test-package")

        mock_run.assert_called_once_with(
            ["pip", "uninstall", "-y", "test-package"],
            check=True,
            capture_output=True,
            text=True,
        )


class TestAddRemovePlugin:
    """Tests for add_plugin and remove_plugin functions."""

    def test_add_plugin_success(self, tmp_path, monkeypatch):
        """Should install package and register it."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)
        monkeypatch.setattr("subprocess.run", MagicMock())
        monkeypatch.setattr("shutil.which", lambda x: None)

        success, message = add_plugin("agentgit-test")

        assert success is True
        assert "Installed" in message
        config = yaml.safe_load(fake_config.read_text())
        assert "agentgit-test" in config["plugins"]["packages"]

    def test_add_plugin_already_registered(self, tmp_path, monkeypatch):
        """Should fail if package already registered."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("plugins:\n  packages:\n    - agentgit-test\n")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        success, message = add_plugin("agentgit-test")

        assert success is False
        assert "already registered" in message

    def test_add_plugin_install_fails(self, tmp_path, monkeypatch):
        """Should return error on install failure."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)
        monkeypatch.setattr("shutil.which", lambda x: None)

        def mock_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "pip", stderr="Package not found")

        monkeypatch.setattr("subprocess.run", mock_run)

        success, message = add_plugin("nonexistent-package")

        assert success is False
        assert "Failed to install" in message

    def test_remove_plugin_success(self, tmp_path, monkeypatch):
        """Should uninstall package and unregister it."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("plugins:\n  packages:\n    - agentgit-test\n")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)
        monkeypatch.setattr("subprocess.run", MagicMock())
        monkeypatch.setattr("shutil.which", lambda x: None)

        success, message = remove_plugin("agentgit-test")

        assert success is True
        assert "Removed" in message
        config = yaml.safe_load(fake_config.read_text())
        assert "agentgit-test" not in config["plugins"]["packages"]

    def test_remove_plugin_not_registered(self, tmp_path, monkeypatch):
        """Should fail if package not registered."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("plugins:\n  packages: []\n")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        success, message = remove_plugin("nonexistent-package")

        assert success is False
        assert "not registered" in message


class TestListPlugins:
    """Tests for plugin listing functions."""

    def test_list_installed_packages(self, tmp_path, monkeypatch):
        """Should return list of installed packages."""
        fake_config = tmp_path / ".agentgit" / "config.yml"
        fake_config.parent.mkdir(parents=True)
        fake_config.write_text("plugins:\n  packages:\n    - pkg1\n    - pkg2\n")
        monkeypatch.setattr("agentgit.settings.CONFIG_PATH", fake_config)

        packages = list_installed_packages()
        assert packages == ["pkg1", "pkg2"]

    def test_list_configured_plugins(self):
        """Should list all configured plugins including builtins."""
        plugins = list_configured_plugins()

        # Should include builtin plugins
        plugin_names = [p["name"] for p in plugins]
        assert "claude_code" in plugin_names

        # Each plugin should have expected keys
        for plugin in plugins:
            assert "name" in plugin
            assert "description" in plugin
            assert "source" in plugin


class TestEntryPoints:
    """Tests for entry point loading."""

    def test_load_plugins_from_entry_points_empty(self, monkeypatch):
        """Should handle empty entry points."""
        pm = get_plugin_manager()

        # Mock empty entry points at the importlib.metadata level
        def mock_entry_points(group=None):
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        count = load_plugins_from_entry_points(pm)
        assert count == 0

    def test_load_plugins_from_entry_points_class(self, monkeypatch):
        """Should instantiate class entry points."""
        pm = get_plugin_manager()

        class MockPlugin:
            pass

        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = MockPlugin

        def mock_entry_points(group=None):
            return [mock_ep]

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        count = load_plugins_from_entry_points(pm)
        assert count == 1

    def test_load_plugins_from_entry_points_handles_errors(self, monkeypatch):
        """Should skip plugins that fail to load."""
        pm = get_plugin_manager()

        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.load.side_effect = ImportError("Module not found")

        def mock_entry_points(group=None):
            return [mock_ep]

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        # Should not raise, just skip the bad plugin
        count = load_plugins_from_entry_points(pm)
        assert count == 0
