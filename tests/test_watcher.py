"""Tests for agentgit.watcher module."""

import json
import time
from pathlib import Path

import pytest

from agentgit.watcher import TranscriptWatcher


@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content for testing."""
    return [
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Create hello.py"},
            "sessionId": "test-session",
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:05.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "Creating file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/hello.py",
                            "content": "print('hello')",
                        },
                    },
                ]
            },
        },
    ]


@pytest.fixture
def sample_jsonl(tmp_path, sample_jsonl_content):
    """Create a sample JSONL file."""
    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for line in sample_jsonl_content:
            f.write(json.dumps(line) + "\n")
    return jsonl_path


class TestTranscriptWatcher:
    """Tests for TranscriptWatcher class."""

    def test_initial_build(self, sample_jsonl, tmp_path):
        """Should do initial build when started."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        watcher = TranscriptWatcher(
            transcript_path=sample_jsonl,
            output_dir=output_dir,
        )
        watcher.start()
        watcher.stop()

        # Should have created the git repo
        assert (output_dir / ".git").exists()
        assert (output_dir / "hello.py").exists()

    def test_calls_on_update_callback(self, sample_jsonl, tmp_path):
        """Should call on_update when new commits are added."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        updates = []

        def on_update(count: int) -> None:
            updates.append(count)

        watcher = TranscriptWatcher(
            transcript_path=sample_jsonl,
            output_dir=output_dir,
            on_update=on_update,
        )
        watcher.start()

        # Give it time to process
        time.sleep(0.1)

        # Add a new operation to the transcript
        new_operation = {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:10.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "Creating another file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_002",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/world.py",
                            "content": "print('world')",
                        },
                    },
                ]
            },
        }

        with open(sample_jsonl, "a") as f:
            f.write(json.dumps(new_operation) + "\n")

        # Give it time to detect the change
        time.sleep(0.5)

        watcher.stop()

        # Should have called on_update at least once
        assert len(updates) >= 1

    def test_custom_author(self, sample_jsonl, tmp_path):
        """Should use custom author name and email."""
        from git import Repo

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        watcher = TranscriptWatcher(
            transcript_path=sample_jsonl,
            output_dir=output_dir,
            author_name="Custom Author",
            author_email="custom@example.com",
        )
        watcher.start()
        time.sleep(0.1)
        watcher.stop()

        repo = Repo(output_dir)
        commit = list(repo.iter_commits())[0]
        assert commit.author.name == "Custom Author"
        assert commit.author.email == "custom@example.com"

    def test_debounces_changes(self, sample_jsonl, tmp_path):
        """Should debounce rapid changes."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        watcher = TranscriptWatcher(
            transcript_path=sample_jsonl,
            output_dir=output_dir,
        )
        watcher.start()
        time.sleep(0.1)

        # Manually trigger handle_change multiple times quickly
        watcher._handle_change()
        watcher._handle_change()
        watcher._handle_change()

        watcher.stop()

        # Should still have processed correctly (debounced)
        assert (output_dir / ".git").exists()

    def test_handles_missing_file(self, tmp_path):
        """Should handle gracefully if transcript file disappears."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a temporary file
        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text('{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": {"content": "test"}}')

        watcher = TranscriptWatcher(
            transcript_path=jsonl_path,
            output_dir=output_dir,
        )

        # Start and then delete the file
        watcher.start()
        time.sleep(0.1)
        jsonl_path.unlink()

        # Manually trigger handle_change - should not crash
        watcher._handle_change()

        watcher.stop()

