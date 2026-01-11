"""Tests for agentgit.core module."""

import pytest

from agentgit.core import (
    AssistantContext,
    FileOperation,
    OperationType,
    Prompt,
    Transcript,
    TranscriptEntry,
)


class TestPrompt:
    """Tests for Prompt dataclass."""

    def test_prompt_id_is_md5_hash(self):
        """Prompt ID should be MD5 hash of text."""
        prompt = Prompt(text="hello world", timestamp="2025-01-01T00:00:00Z")
        # MD5 of "hello world" is 5eb63bbbe01eeed093cb22bb8f5acdc3
        assert prompt.prompt_id == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_prompt_id_is_stable(self):
        """Same text should always produce same ID."""
        prompt1 = Prompt(text="test prompt", timestamp="2025-01-01T00:00:00Z")
        prompt2 = Prompt(text="test prompt", timestamp="2025-01-02T00:00:00Z")
        assert prompt1.prompt_id == prompt2.prompt_id

    def test_different_text_different_id(self):
        """Different text should produce different ID."""
        prompt1 = Prompt(text="first prompt", timestamp="2025-01-01T00:00:00Z")
        prompt2 = Prompt(text="second prompt", timestamp="2025-01-01T00:00:00Z")
        assert prompt1.prompt_id != prompt2.prompt_id

    def test_short_id_is_first_8_chars(self):
        """Short ID should be first 8 characters of full ID."""
        prompt = Prompt(text="hello world", timestamp="2025-01-01T00:00:00Z")
        assert prompt.short_id == prompt.prompt_id[:8]
        assert len(prompt.short_id) == 8

    def test_prompt_id_cached(self):
        """Prompt ID should be computed once and cached."""
        prompt = Prompt(text="test", timestamp="2025-01-01T00:00:00Z")
        id1 = prompt.prompt_id
        id2 = prompt.prompt_id
        assert id1 is id2


class TestAssistantContext:
    """Tests for AssistantContext dataclass."""

    def test_summary_returns_thinking_if_present(self):
        """Summary should return thinking content if available."""
        context = AssistantContext(thinking="I need to think about this", text="Some text")
        assert context.summary == "I need to think about this"

    def test_summary_returns_text_if_no_thinking(self):
        """Summary should return text if no thinking."""
        context = AssistantContext(text="Just some text")
        assert context.summary == "Just some text"

    def test_summary_returns_empty_if_nothing(self):
        """Summary should return empty string if nothing available."""
        context = AssistantContext()
        assert context.summary == ""


class TestFileOperation:
    """Tests for FileOperation dataclass."""

    def test_filename_extracts_from_path(self):
        """Filename should extract just the file name from full path."""
        op = FileOperation(
            file_path="/path/to/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert op.filename == "file.py"

    def test_filename_handles_no_path(self):
        """Filename should handle files with no directory."""
        op = FileOperation(
            file_path="file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert op.filename == "file.py"

    def test_operation_types(self):
        """All operation types should work."""
        for op_type in OperationType:
            op = FileOperation(
                file_path="/test.py",
                operation_type=op_type,
                timestamp="2025-01-01T00:00:00Z",
            )
            assert op.operation_type == op_type


class TestTranscript:
    """Tests for Transcript dataclass."""

    def test_get_prompt_by_full_id(self):
        """Should find prompt by full ID."""
        prompt = Prompt(text="test prompt", timestamp="2025-01-01T00:00:00Z")
        transcript = Transcript(prompts=[prompt])
        found = transcript.get_prompt_by_id(prompt.prompt_id)
        assert found is prompt

    def test_get_prompt_by_short_id(self):
        """Should find prompt by short ID."""
        prompt = Prompt(text="test prompt", timestamp="2025-01-01T00:00:00Z")
        transcript = Transcript(prompts=[prompt])
        found = transcript.get_prompt_by_id(prompt.short_id)
        assert found is prompt

    def test_get_prompt_not_found(self):
        """Should return None if prompt not found."""
        transcript = Transcript(prompts=[])
        found = transcript.get_prompt_by_id("nonexistent")
        assert found is None


class TestTranscriptEntry:
    """Tests for TranscriptEntry dataclass."""

    def test_entry_creation(self):
        """Should create entry with required fields."""
        entry = TranscriptEntry(
            entry_type="user",
            timestamp="2025-01-01T00:00:00Z",
            message={"content": "Hello"},
        )
        assert entry.entry_type == "user"
        assert entry.timestamp == "2025-01-01T00:00:00Z"
        assert entry.message == {"content": "Hello"}
        assert entry.is_continuation is False
        assert entry.is_meta is False

# TestSourceCommit deleted - SourceCommit class removed
