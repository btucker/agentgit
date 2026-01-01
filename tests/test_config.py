"""Tests for agentgit config module."""

import pytest
from git import Repo

from agentgit.config import ProjectConfig, load_config, save_config


class TestProjectConfig:
    """Tests for ProjectConfig dataclass."""

    def test_default_config(self):
        """Should have None defaults."""
        config = ProjectConfig()
        assert config.enhancer is None
        assert config.enhance_model is None

    def test_custom_config(self):
        """Should accept custom values."""
        config = ProjectConfig(enhancer="rules", enhance_model="haiku")
        assert config.enhancer == "rules"
        assert config.enhance_model == "haiku"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_nonexistent_dir(self, tmp_path):
        """Should return empty config for nonexistent directory."""
        config = load_config(tmp_path / "nonexistent")
        assert config.enhancer is None
        assert config.enhance_model is None

    def test_load_from_non_git_dir(self, tmp_path):
        """Should return empty config for non-git directory."""
        config = load_config(tmp_path)
        assert config.enhancer is None
        assert config.enhance_model is None

    def test_load_from_git_repo_without_config(self, tmp_path):
        """Should return empty config for git repo without agentgit config."""
        Repo.init(tmp_path)
        config = load_config(tmp_path)
        assert config.enhancer is None
        assert config.enhance_model is None

    def test_load_from_git_repo_with_config(self, tmp_path):
        """Should load config from git repo."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as writer:
            writer.set_value("agentgit", "enhancer", "claude_code")
            writer.set_value("agentgit", "enhanceModel", "sonnet")

        config = load_config(tmp_path)
        assert config.enhancer == "claude_code"
        assert config.enhance_model == "sonnet"

    def test_load_partial_config(self, tmp_path):
        """Should handle partial config."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as writer:
            writer.set_value("agentgit", "enhancer", "rules")

        config = load_config(tmp_path)
        assert config.enhancer == "rules"
        assert config.enhance_model is None


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_to_nonexistent_dir(self, tmp_path):
        """Should not raise for nonexistent directory."""
        config = ProjectConfig(enhancer="rules")
        # Should not raise
        save_config(tmp_path / "nonexistent", config)

    def test_save_to_non_git_dir(self, tmp_path):
        """Should not raise for non-git directory."""
        config = ProjectConfig(enhancer="rules")
        # Should not raise
        save_config(tmp_path, config)

    def test_save_and_load_config(self, tmp_path):
        """Should save config that can be loaded back."""
        Repo.init(tmp_path)
        config = ProjectConfig(enhancer="claude_code", enhance_model="haiku")
        save_config(tmp_path, config)

        loaded = load_config(tmp_path)
        assert loaded.enhancer == "claude_code"
        assert loaded.enhance_model == "haiku"

    def test_save_partial_config(self, tmp_path):
        """Should save partial config."""
        Repo.init(tmp_path)
        config = ProjectConfig(enhancer="rules")
        save_config(tmp_path, config)

        loaded = load_config(tmp_path)
        assert loaded.enhancer == "rules"
        assert loaded.enhance_model is None

    def test_update_existing_config(self, tmp_path):
        """Should update existing config values."""
        repo = Repo.init(tmp_path)
        with repo.config_writer() as writer:
            writer.set_value("agentgit", "enhancer", "rules")
            writer.set_value("agentgit", "enhanceModel", "haiku")

        # Update just the enhancer
        config = ProjectConfig(enhancer="claude_code")
        save_config(tmp_path, config)

        loaded = load_config(tmp_path)
        assert loaded.enhancer == "claude_code"
        # enhanceModel should still be there from before
        assert loaded.enhance_model == "haiku"
