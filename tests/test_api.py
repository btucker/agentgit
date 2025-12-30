"""Tests for agentgit public API."""

import json
from pathlib import Path

import pytest

import agentgit
from agentgit import (
    FileOperation,
    OperationType,
    Prompt,
    Transcript,
    build_repo,
    parse_transcript,
    transcript_to_repo,
)


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file."""
    content = [
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

    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


class TestPublicAPI:
    """Tests for public API functions."""

    def test_version(self):
        """Should have version defined."""
        assert agentgit.__version__ == "0.1.0"

    def test_parse_transcript(self, sample_jsonl):
        """Should parse transcript and extract operations."""
        transcript = parse_transcript(sample_jsonl)

        assert isinstance(transcript, Transcript)
        assert len(transcript.prompts) == 1
        assert len(transcript.operations) == 1
        assert transcript.operations[0].operation_type == OperationType.WRITE

    def test_parse_transcript_with_string_path(self, sample_jsonl):
        """Should accept string path."""
        transcript = parse_transcript(str(sample_jsonl))
        assert isinstance(transcript, Transcript)

    def test_parse_transcript_invalid_format(self, tmp_path):
        """Should raise error for invalid format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a transcript")

        with pytest.raises(ValueError, match="Could not detect"):
            parse_transcript(txt_file)

    def test_build_repo(self, tmp_path):
        """Should build repo from operations."""
        ops = [
            FileOperation(
                file_path="/project/hello.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="print('hello')",
            )
        ]

        repo, repo_path, mapping = build_repo(ops, output_dir=tmp_path)

        assert repo is not None
        assert repo_path == tmp_path
        assert (tmp_path / ".git").exists()

    def test_transcript_to_repo(self, sample_jsonl, tmp_path):
        """Should parse and build in one call."""
        output_dir = tmp_path / "output"
        repo, repo_path, transcript = transcript_to_repo(sample_jsonl, output_dir=output_dir)

        assert repo is not None
        assert repo_path == output_dir
        assert isinstance(transcript, Transcript)
        assert len(list(repo.iter_commits())) > 0

    def test_transcript_to_repo_with_string_paths(self, sample_jsonl, tmp_path):
        """Should accept string paths."""
        output_dir = tmp_path / "output"
        repo, repo_path, transcript = transcript_to_repo(
            str(sample_jsonl),
            output_dir=output_dir,
        )
        assert repo is not None


class TestExports:
    """Tests for module exports."""

    def test_core_types_exported(self):
        """Core types should be exported."""
        assert hasattr(agentgit, "FileOperation")
        assert hasattr(agentgit, "Prompt")
        assert hasattr(agentgit, "AssistantContext")
        assert hasattr(agentgit, "TranscriptEntry")
        assert hasattr(agentgit, "Transcript")
        assert hasattr(agentgit, "OperationType")
        assert hasattr(agentgit, "SourceCommit")

    def test_plugin_decorators_exported(self):
        """Plugin decorators should be exported."""
        assert hasattr(agentgit, "hookspec")
        assert hasattr(agentgit, "hookimpl")

    def test_functions_exported(self):
        """Main functions should be exported."""
        assert hasattr(agentgit, "parse_transcript")
        assert hasattr(agentgit, "build_repo")
        assert hasattr(agentgit, "transcript_to_repo")
        assert hasattr(agentgit, "format_commit_message")
        assert hasattr(agentgit, "discover_transcripts")
        assert hasattr(agentgit, "find_git_root")


class TestFindGitRoot:
    """Tests for find_git_root function."""

    def test_finds_git_root(self, tmp_path):
        """Should find git root directory."""
        from agentgit import find_git_root
        from git import Repo

        # Create a git repo
        repo_path = tmp_path / "myrepo"
        repo_path.mkdir()
        Repo.init(repo_path)

        # Create a subdirectory
        subdir = repo_path / "src" / "lib"
        subdir.mkdir(parents=True)

        # find_git_root from subdirectory should return repo root
        result = find_git_root(subdir)
        assert result == repo_path

    def test_returns_none_if_not_git_repo(self, tmp_path):
        """Should return None if not in a git repo."""
        from agentgit import find_git_root

        result = find_git_root(tmp_path)
        assert result is None


class TestDiscoverTranscripts:
    """Tests for discover_transcripts function."""

    def test_raises_if_not_in_git_repo(self, tmp_path, monkeypatch):
        """Should raise ValueError if not in git repo and no path given."""
        from agentgit import discover_transcripts

        # Change to non-git directory
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="Not in a git repository"):
            discover_transcripts()

    def test_returns_empty_if_no_transcripts(self, tmp_path):
        """Should return empty list if no transcripts found."""
        from agentgit import discover_transcripts

        result = discover_transcripts(tmp_path)
        assert result == []
