"""Core dataclasses and interfaces for agentgit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OperationType(str, Enum):
    """Types of file operations."""

    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"


@dataclass
class Prompt:
    """Represents a user prompt in a transcript.

    The prompt_id is computed as md5(text) for stable, content-based identification
    that persists across re-processing of the same transcript.
    """

    text: str
    timestamp: str
    raw_entry: dict[str, Any] = field(default_factory=dict)

    _prompt_id: Optional[str] = field(default=None, repr=False)

    @property
    def prompt_id(self) -> str:
        """MD5 hash of the prompt text for stable identification."""
        if self._prompt_id is None:
            self._prompt_id = hashlib.md5(self.text.encode("utf-8")).hexdigest()
        return self._prompt_id

    @property
    def short_id(self) -> str:
        """First 8 characters of the prompt_id for display."""
        return self.prompt_id[:8]


@dataclass
class AssistantContext:
    """Context from assistant messages preceding a file operation.

    Captures the reasoning/thinking that explains why a change was made.
    """

    thinking: Optional[str] = None
    text: Optional[str] = None
    timestamp: str = ""

    @property
    def summary(self) -> str:
        """Get a summary of the context for commit messages."""
        if self.thinking:
            return self.thinking
        if self.text:
            return self.text
        return ""


@dataclass
class FileOperation:
    """Represents a single file operation extracted from a transcript.

    Contains all metadata needed to create a rich git commit.
    """

    file_path: str
    operation_type: OperationType
    timestamp: str

    tool_id: str = ""

    # Content for operations
    content: Optional[str] = None  # For WRITE operations
    old_string: Optional[str] = None  # For EDIT operations
    new_string: Optional[str] = None  # For EDIT operations
    replace_all: bool = False  # For EDIT: replace all occurrences vs first
    recursive: bool = False  # For DELETE: recursive directory delete
    original_content: Optional[str] = None  # Pre-edit content from tool result

    # Rich metadata for commit messages
    prompt: Optional[Prompt] = None
    assistant_context: Optional[AssistantContext] = None

    # Raw data for extensibility
    raw_tool_use: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Extract just the filename from the path."""
        return self.file_path.rsplit("/", 1)[-1] if "/" in self.file_path else self.file_path


@dataclass
class TranscriptEntry:
    """A single entry from an agent transcript.

    Generic representation that plugins parse into.
    """

    entry_type: str  # "user", "assistant", "system", etc.
    timestamp: str
    message: dict[str, Any]
    raw_entry: dict[str, Any] = field(default_factory=dict)

    is_continuation: bool = False
    is_meta: bool = False  # e.g., skill expansions


@dataclass
class Transcript:
    """A complete parsed transcript with all entries and extracted operations."""

    entries: list[TranscriptEntry] = field(default_factory=list)
    prompts: list[Prompt] = field(default_factory=list)
    operations: list[FileOperation] = field(default_factory=list)

    source_path: Optional[str] = None
    source_format: str = ""

    session_id: Optional[str] = None
    session_cwd: Optional[str] = None

    def get_prompt_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Look up a prompt by its md5-based ID."""
        for prompt in self.prompts:
            if prompt.prompt_id == prompt_id or prompt.short_id == prompt_id:
                return prompt
        return None


@dataclass
class SourceCommit:
    """A commit from the source repository."""

    sha: str
    message: str
    timestamp: str
    author: str
    author_email: str = ""
    files_changed: list[str] = field(default_factory=list)
