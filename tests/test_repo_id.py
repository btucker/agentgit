"""Tests for repo ID-based naming."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agentgit.cli import get_default_output_dir, get_repo_id


def test_get_repo_id_returns_first_12_chars_of_root_commit(tmp_path):
    """Test that get_repo_id returns first 12 chars of root commit SHA."""
    # Create a git repo with a known root commit
    import subprocess

    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    # Disable GPG signing for test commits
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    test_file = repo_path / "test.txt"
    test_file.write_text("test")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Get the root commit
    result = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    expected_root = result.stdout.strip()[:12]

    # Test get_repo_id
    repo_id = get_repo_id(repo_path)
    assert repo_id == expected_root
    assert len(repo_id) == 12


def test_get_repo_id_returns_none_for_non_git_dir(tmp_path):
    """Test that get_repo_id returns None for non-git directories."""
    non_git_dir = tmp_path / "not-a-repo"
    non_git_dir.mkdir()

    repo_id = get_repo_id(non_git_dir)
    assert repo_id is None


def test_get_default_output_dir_uses_repo_id_for_git_repo(tmp_path, monkeypatch):
    """Test that get_default_output_dir uses repo ID for git repos."""
    import subprocess

    # Create a git repo
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    # Disable GPG signing for test commits
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    test_file = repo_path / "test.txt"
    test_file.write_text("test")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Get expected repo ID
    expected_id = get_repo_id(repo_path)

    # Mock find_git_root to return our test repo
    def mock_find_git_root():
        return repo_path

    monkeypatch.setattr("agentgit.find_git_root", mock_find_git_root)

    # Create a fake transcript path
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.touch()

    # Test get_default_output_dir
    output_dir = get_default_output_dir(transcript_path)

    # Should use repo ID, not path encoding
    assert output_dir == Path.home() / ".agentgit" / "projects" / expected_id
    assert expected_id in str(output_dir)
    assert "test-repo" not in str(output_dir)  # Should not use path encoding


def test_get_default_output_dir_fails_for_non_git(tmp_path):
    """Test that non-git directories raise an error (no fallback)."""
    from click import ClickException

    def mock_find_git_root():
        return None

    with patch("agentgit.find_git_root", mock_find_git_root):
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.touch()

        # Should raise an exception when no git repo found
        with pytest.raises(ClickException, match="Could not determine git repository"):
            get_default_output_dir(transcript_path)
