"""Configuration management for agentgit projects using git config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """Configuration for an agentgit project."""

    enhancer: str | None = None
    enhance_model: str | None = None


def load_config(output_dir: Path) -> ProjectConfig:
    """Load project configuration from git config.

    Args:
        output_dir: The agentgit output directory (git repo).

    Returns:
        ProjectConfig with saved settings, or defaults if no config exists.
    """
    from git import Repo
    from git.exc import InvalidGitRepositoryError
    from git.exc import NoSuchPathError

    if not output_dir.exists():
        return ProjectConfig()

    try:
        repo = Repo(output_dir)
        reader = repo.config_reader()

        enhancer = None
        enhance_model = None

        try:
            enhancer = reader.get_value("agentgit", "enhancer")
        except Exception:
            pass

        try:
            enhance_model = reader.get_value("agentgit", "enhanceModel")
        except Exception:
            pass

        return ProjectConfig(enhancer=enhancer, enhance_model=enhance_model)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return ProjectConfig()


def save_config(output_dir: Path, config: ProjectConfig) -> None:
    """Save project configuration to git config.

    Args:
        output_dir: The agentgit output directory (git repo).
        config: The configuration to save.
    """
    from git import Repo
    from git.exc import InvalidGitRepositoryError
    from git.exc import NoSuchPathError

    if not output_dir.exists():
        return

    try:
        repo = Repo(output_dir)
        with repo.config_writer() as writer:
            if config.enhancer is not None:
                writer.set_value("agentgit", "enhancer", config.enhancer)
            if config.enhance_model is not None:
                writer.set_value("agentgit", "enhanceModel", config.enhance_model)
    except (InvalidGitRepositoryError, NoSuchPathError):
        # Repo doesn't exist yet, config will be saved after creation
        pass
