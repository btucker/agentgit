"""Tests for alazygit command."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAlazygit:
    """Tests for the alazygit wrapper command."""

    def test_alazygit_checks_for_lazygit(self, monkeypatch, tmp_path):
        """Test that alazygit checks if lazygit is installed."""
        from agentgit.cli import run_lazygit_passthrough

        # Mock shutil.which to return None (lazygit not found)
        monkeypatch.setattr("shutil.which", lambda x: None)

        # Mock get_agentgit_repo_path
        monkeypatch.setattr(
            "agentgit.cli.get_agentgit_repo_path", lambda: tmp_path / ".agentgit"
        )

        # Should raise ClickException about lazygit not installed
        from click import ClickException

        with pytest.raises(ClickException, match="lazygit is not installed"):
            run_lazygit_passthrough([])

    def test_alazygit_checks_for_repo(self, monkeypatch, tmp_path):
        """Test that alazygit checks for agentgit repo."""
        from agentgit.cli import run_lazygit_passthrough

        # Mock shutil.which to return lazygit path
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lazygit")

        # Mock get_agentgit_repo_path to return non-existent path
        monkeypatch.setattr(
            "agentgit.cli.get_agentgit_repo_path", lambda: tmp_path / "nonexistent"
        )

        # Should raise ClickException about no repo
        from click import ClickException

        with pytest.raises(ClickException, match="No agentgit repository found"):
            run_lazygit_passthrough([])

    def test_alazygit_checks_for_git_dir(self, monkeypatch, tmp_path):
        """Test that alazygit checks for .git directory."""
        from agentgit.cli import run_lazygit_passthrough

        # Create repo directory but no .git directory
        repo_path = tmp_path / ".agentgit" / "projects" / "test"
        repo_path.mkdir(parents=True)

        # Mock shutil.which to return lazygit path
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lazygit")

        # Mock get_agentgit_repo_path
        monkeypatch.setattr("agentgit.cli.get_agentgit_repo_path", lambda: repo_path)

        # Should raise ClickException about missing .git directory
        from click import ClickException

        with pytest.raises(ClickException, match="Git directory not found"):
            run_lazygit_passthrough([])

    def test_alazygit_runs_with_correct_args(self, monkeypatch, tmp_path):
        """Test that alazygit passes correct arguments to lazygit."""
        from agentgit.cli import run_lazygit_passthrough

        # Create a fake repo directory with .git
        repo_path = tmp_path / ".agentgit" / "projects" / "test"
        repo_path.mkdir(parents=True)
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        # Mock shutil.which to return lazygit path
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lazygit")

        # Mock get_agentgit_repo_path
        monkeypatch.setattr("agentgit.cli.get_agentgit_repo_path", lambda: repo_path)

        # Mock subprocess.run
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr("subprocess.run", mock_run)

        # Run with some args
        with pytest.raises(SystemExit) as exc:
            run_lazygit_passthrough(["--version"])

        # Check that subprocess.run was called with correct arguments
        # Should pass the .git directory path, not the repo root
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["lazygit", "-g", str(git_dir), "--version"]
        assert exc.value.code == 0

    def test_alazygit_passes_exit_code(self, monkeypatch, tmp_path):
        """Test that alazygit passes through lazygit's exit code."""
        from agentgit.cli import run_lazygit_passthrough

        # Create a fake repo directory with .git
        repo_path = tmp_path / ".agentgit" / "projects" / "test"
        repo_path.mkdir(parents=True)
        (repo_path / ".git").mkdir()

        # Mock dependencies
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lazygit")
        monkeypatch.setattr("agentgit.cli.get_agentgit_repo_path", lambda: repo_path)

        # Mock subprocess.run to return exit code 42
        mock_run = MagicMock(return_value=MagicMock(returncode=42))
        monkeypatch.setattr("subprocess.run", mock_run)

        # Run and check exit code
        with pytest.raises(SystemExit) as exc:
            run_lazygit_passthrough([])

        assert exc.value.code == 42
