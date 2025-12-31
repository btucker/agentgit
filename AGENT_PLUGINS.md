# Agent Plugins

agentgit uses a plugin system to support different AI coding agent transcript formats. This document explains how agent plugins work and how to create your own.

## Overview

Agent plugins convert transcripts from AI coding assistants (like Claude Code, Codex, Aider, etc.) into a standardized format that agentgit can process into git repositories. The plugin system is built on [pluggy](https://pluggy.readthedocs.io/), the same framework used by pytest.

## Built-in Plugins

agentgit ships with plugins for:

- **claude_code** - Claude Code JSONL transcripts (`~/.claude/projects/`)
- **codex** - Codex CLI JSONL transcripts (`~/.codex/sessions/`)

List available plugins:

```bash
agentgit agents
```

## Plugin Architecture

### Hook-based Design

Plugins implement **hooks** - functions that agentgit calls at specific points in the processing pipeline. Each hook has a well-defined purpose:

| Hook | Purpose | Returns |
|------|---------|---------|
| `agentgit_get_plugin_info` | Identify the plugin | `dict` with name/description |
| `agentgit_detect_format` | Auto-detect if a file matches this format | Format string or `None` |
| `agentgit_parse_transcript` | Parse file into structured data | `Transcript` object |
| `agentgit_extract_operations` | Extract file operations (Write/Edit/Delete) | `list[FileOperation]` |
| `agentgit_enrich_operation` | Add context (prompt, reasoning) to operations | `FileOperation` |
| `agentgit_build_prompt_responses` | Group operations by prompt/response | `list[PromptResponse]` |
| `agentgit_discover_transcripts` | Find transcripts for a project | `list[Path]` |
| `agentgit_get_project_name` | Extract project identifier from path | `str` or `None` |
| `agentgit_get_display_name` | Human-readable name for UI | `str` or `None` |

### Data Flow

```
Transcript File
      │
      ▼
┌─────────────────┐
│ detect_format   │──▶ "my_agent_jsonl"
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ parse_transcript│──▶ Transcript(entries, prompts, ...)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│extract_operations──▶ [FileOperation, FileOperation, ...]
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ enrich_operation│──▶ FileOperation with prompt + context
└─────────────────┘
      │
      ▼
┌─────────────────┐
│build_prompt_    │
│  responses      │──▶ [PromptResponse(prompt, turns), ...]
└─────────────────┘
      │
      ▼
    Git Repository
```

## Creating an Agent Plugin

### Step 1: Create the Plugin Class

```python
# my_agent_plugin.py
from pathlib import Path
from agentgit.plugins import hookimpl
from agentgit.core import (
    Transcript,
    TranscriptEntry,
    Prompt,
    FileOperation,
    OperationType,
    AssistantContext,
    AssistantTurn,
    PromptResponse,
)

FORMAT_MY_AGENT = "my_agent_jsonl"

class MyAgentPlugin:
    """Plugin for parsing MyAgent transcripts."""

    @hookimpl
    def agentgit_get_plugin_info(self) -> dict[str, str]:
        """Return plugin identification."""
        return {
            "name": "my_agent",
            "description": "MyAgent JSONL transcripts",
        }

    @hookimpl
    def agentgit_detect_format(self, path: Path) -> str | None:
        """Detect if this file is a MyAgent transcript."""
        if path.suffix != ".jsonl":
            return None

        # Check for format-specific markers
        try:
            with open(path) as f:
                for line in f:
                    import json
                    obj = json.loads(line)
                    # Look for MyAgent-specific fields
                    if "my_agent_version" in obj:
                        return FORMAT_MY_AGENT
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_parse_transcript(
        self, path: Path, format: str
    ) -> Transcript | None:
        """Parse MyAgent transcript into structured data."""
        if format != FORMAT_MY_AGENT:
            return None

        entries = []
        prompts = []

        import json
        with open(path) as f:
            for line in f:
                obj = json.loads(line)

                # Parse entries based on your format
                entry_type = obj.get("role")  # e.g., "user" or "assistant"
                timestamp = obj.get("timestamp", "")

                entry = TranscriptEntry(
                    entry_type=entry_type,
                    timestamp=timestamp,
                    message=obj.get("message", {}),
                    raw_entry=obj,
                )
                entries.append(entry)

                # Extract user prompts
                if entry_type == "user":
                    prompts.append(Prompt(
                        text=obj.get("content", ""),
                        timestamp=timestamp,
                        raw_entry=obj,
                    ))

        return Transcript(
            entries=entries,
            prompts=prompts,
            source_path=str(path),
            source_format=FORMAT_MY_AGENT,
        )

    @hookimpl
    def agentgit_extract_operations(
        self, transcript: Transcript
    ) -> list[FileOperation]:
        """Extract file operations from transcript."""
        if transcript.source_format != FORMAT_MY_AGENT:
            return []

        operations = []

        for entry in transcript.entries:
            if entry.entry_type != "assistant":
                continue

            # Parse tool calls from your format
            for tool_call in entry.message.get("tool_calls", []):
                if tool_call.get("name") == "write_file":
                    operations.append(FileOperation(
                        file_path=tool_call["args"]["path"],
                        operation_type=OperationType.WRITE,
                        timestamp=entry.timestamp,
                        tool_id=tool_call.get("id", ""),
                        content=tool_call["args"]["content"],
                        raw_tool_use=tool_call,
                    ))
                elif tool_call.get("name") == "edit_file":
                    operations.append(FileOperation(
                        file_path=tool_call["args"]["path"],
                        operation_type=OperationType.EDIT,
                        timestamp=entry.timestamp,
                        tool_id=tool_call.get("id", ""),
                        old_string=tool_call["args"]["old"],
                        new_string=tool_call["args"]["new"],
                        raw_tool_use=tool_call,
                    ))

        return sorted(operations, key=lambda op: op.timestamp)

    @hookimpl
    def agentgit_enrich_operation(
        self, operation: FileOperation, transcript: Transcript
    ) -> FileOperation:
        """Add prompt and context to operations."""
        if transcript.source_format != FORMAT_MY_AGENT:
            return operation

        # Find the prompt that triggered this operation
        for prompt in transcript.prompts:
            if prompt.timestamp <= operation.timestamp:
                operation.prompt = prompt
            else:
                break

        # Add assistant reasoning/thinking if available
        for entry in transcript.entries:
            if entry.timestamp > operation.timestamp:
                break
            if entry.entry_type == "assistant":
                thinking = entry.message.get("thinking", "")
                if thinking:
                    operation.assistant_context = AssistantContext(
                        thinking=thinking,
                        timestamp=entry.timestamp,
                    )

        return operation

    @hookimpl
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Find MyAgent transcripts for a project."""
        # Implement based on where your agent stores transcripts
        my_agent_dir = Path.home() / ".my_agent" / "sessions"
        if not my_agent_dir.exists():
            return []

        transcripts = list(my_agent_dir.glob("**/*.jsonl"))

        if project_path:
            # Filter to transcripts for this project
            transcripts = [
                t for t in transcripts
                if self._matches_project(t, project_path)
            ]

        return sorted(transcripts, key=lambda p: p.stat().st_mtime, reverse=True)

    def _matches_project(self, transcript: Path, project: Path) -> bool:
        """Check if transcript belongs to project."""
        # Implement your matching logic
        return True
```

### Step 2: Core Data Structures

#### FileOperation

Represents a single file change:

```python
@dataclass
class FileOperation:
    file_path: str                    # Absolute path to file
    operation_type: OperationType     # WRITE, EDIT, or DELETE
    timestamp: str                    # ISO timestamp
    tool_id: str = ""                 # Unique ID for deduplication

    # For WRITE operations:
    content: str | None = None

    # For EDIT operations:
    old_string: str | None = None
    new_string: str | None = None
    replace_all: bool = False
    original_content: str | None = None  # Full file before edit

    # For DELETE operations:
    recursive: bool = False

    # Enrichment data (added by enrich_operation):
    prompt: Prompt | None = None
    assistant_context: AssistantContext | None = None
    raw_tool_use: dict = field(default_factory=dict)
```

#### Prompt

User request that triggered operations:

```python
@dataclass
class Prompt:
    text: str           # The user's prompt text
    timestamp: str      # When the prompt was sent
    raw_entry: dict     # Original transcript entry

    @property
    def prompt_id(self) -> str:
        """MD5 hash for stable identification."""
        return hashlib.md5(self.text.encode()).hexdigest()
```

#### AssistantContext

Reasoning/thinking that explains why a change was made:

```python
@dataclass
class AssistantContext:
    thinking: str = ""    # Extended thinking (if available)
    text: str = ""        # Regular assistant text
    timestamp: str = ""

    @property
    def summary(self) -> str:
        """Return thinking or text, preferring thinking."""
        return self.thinking or self.text
```

### Step 3: Package Your Plugin

Create a Python package with this structure:

```
agentgit-my-agent/
├── pyproject.toml
├── README.md
└── src/
    └── agentgit_my_agent/
        ├── __init__.py
        └── plugin.py
```

#### pyproject.toml

```toml
[project]
name = "agentgit-my-agent"
version = "0.1.0"
description = "MyAgent support for agentgit"
dependencies = ["agentgit>=0.1.0"]

[project.entry-points."agentgit"]
my_agent = "agentgit_my_agent.plugin:MyAgentPlugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The key is the `[project.entry-points."agentgit"]` section - this registers your plugin with agentgit.

#### __init__.py

```python
from agentgit_my_agent.plugin import MyAgentPlugin

__all__ = ["MyAgentPlugin"]
```

### Step 4: Install and Use

Install your plugin package:

```bash
pip install agentgit-my-agent
# or during development:
pip install -e ./agentgit-my-agent
```

The plugin is automatically discovered via entry points:

```bash
agentgit agents
# Output:
#   claude_code: Claude Code JSONL transcripts
#   codex: Codex CLI JSONL transcripts
#   my_agent: MyAgent JSONL transcripts
```

## Installing External Plugins

### Via agentgit CLI (Recommended)

Use `agentgit agents add` to install plugins:

```bash
agentgit agents add agentgit-aider
agentgit agents add agentgit-cursor
```

This will:
1. Install the package using uv (or pip as fallback)
2. Register it in `~/.agentgit/plugins.json`
3. Automatically discover the plugin via entry points

List installed plugins:

```bash
agentgit agents
agentgit agents list -v  # verbose mode
```

Remove a plugin:

```bash
agentgit agents remove agentgit-aider
```

### Via pip directly

You can also install plugin packages directly:

```bash
pip install agentgit-aider
# or
uv pip install agentgit-aider
```

Plugins that register the `agentgit` entry point are automatically discovered.

## Hook Reference

### agentgit_get_plugin_info

**Required.** Return plugin metadata.

```python
@hookimpl
def agentgit_get_plugin_info(self) -> dict[str, str]:
    return {
        "name": "my_agent",           # Short identifier
        "description": "MyAgent ...", # Human-readable description
    }
```

### agentgit_detect_format

**Required.** Auto-detect if a file is handled by this plugin.

- Return a format identifier string (e.g., `"my_agent_jsonl"`) if detected
- Return `None` if this plugin doesn't handle the file
- Only the first plugin to return non-None wins (`firstresult=True`)

```python
@hookimpl
def agentgit_detect_format(self, path: Path) -> str | None:
    if path.suffix != ".jsonl":
        return None
    # Check file contents for format-specific markers
    ...
```

### agentgit_parse_transcript

**Required.** Parse a transcript file into structured data.

- Only called if `format` matches what you returned from `detect_format`
- Return `None` if you don't handle this format

```python
@hookimpl
def agentgit_parse_transcript(
    self, path: Path, format: str
) -> Transcript | None:
    if format != FORMAT_MY_AGENT:
        return None
    # Parse and return Transcript
    ...
```

### agentgit_extract_operations

**Required.** Extract file operations from the parsed transcript.

- Multiple plugins can contribute operations (results are combined)
- Check `transcript.source_format` to only process your format

```python
@hookimpl
def agentgit_extract_operations(
    self, transcript: Transcript
) -> list[FileOperation]:
    if transcript.source_format != FORMAT_MY_AGENT:
        return []
    # Extract and return operations
    ...
```

### agentgit_enrich_operation

**Optional but recommended.** Add context to operations.

- Called for each operation after extraction
- Add the triggering prompt and assistant reasoning

```python
@hookimpl
def agentgit_enrich_operation(
    self, operation: FileOperation, transcript: Transcript
) -> FileOperation:
    if transcript.source_format != FORMAT_MY_AGENT:
        return operation
    # Add prompt and context
    ...
```

### agentgit_build_prompt_responses

**Optional.** Build the prompt-response grouping structure.

- Used for merge-based git history
- Groups operations by assistant turn, grouped under prompts

```python
@hookimpl
def agentgit_build_prompt_responses(
    self, transcript: Transcript
) -> list[PromptResponse]:
    if transcript.source_format != FORMAT_MY_AGENT:
        return []
    # Build and return prompt-response structure
    ...
```

### agentgit_discover_transcripts

**Optional but recommended.** Find transcripts for a project.

- Called when no transcript is explicitly provided
- If `project_path` is `None`, return all transcripts from all projects

```python
@hookimpl
def agentgit_discover_transcripts(
    self, project_path: Path | None
) -> list[Path]:
    # Find and return transcript paths
    ...
```

### agentgit_get_project_name

**Optional.** Extract project identifier from transcript location.

- Used for determining output directory
- Return `None` if you can't determine the project

```python
@hookimpl
def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
    # Extract project name from path
    ...
```

### agentgit_get_display_name

**Optional.** Return human-readable name for the UI.

```python
@hookimpl
def agentgit_get_display_name(self, transcript_path: Path) -> str | None:
    # Return display name
    ...
```

## Best Practices

### 1. Check Format Before Processing

Always check `transcript.source_format` before processing:

```python
@hookimpl
def agentgit_extract_operations(self, transcript: Transcript) -> list[FileOperation]:
    if transcript.source_format != FORMAT_MY_AGENT:
        return []  # Let other plugins handle it
    ...
```

### 2. Use Stable Tool IDs

The `tool_id` field enables incremental processing (skipping already-committed operations). Use unique, stable identifiers:

```python
FileOperation(
    tool_id=tool_call.get("id") or f"{timestamp}-{file_path}",
    ...
)
```

### 3. Preserve Raw Data

Store original data in `raw_entry` and `raw_tool_use` for debugging:

```python
TranscriptEntry(
    raw_entry=original_json_object,
    ...
)
```

### 4. Sort Operations by Timestamp

Return operations sorted chronologically:

```python
return sorted(operations, key=lambda op: op.timestamp)
```

### 5. Handle Missing Data Gracefully

Transcripts may have missing fields:

```python
timestamp = obj.get("timestamp", "")
content = obj.get("content") or ""
```

## Testing Your Plugin

Create tests in `tests/plugins/my_agent/`:

```python
# tests/plugins/my_agent/test_my_agent.py
import pytest
from pathlib import Path
from agentgit_my_agent import MyAgentPlugin

@pytest.fixture
def plugin():
    return MyAgentPlugin()

def test_detect_format(plugin, tmp_path):
    # Create test transcript
    transcript = tmp_path / "test.jsonl"
    transcript.write_text('{"my_agent_version": "1.0"}\n')

    assert plugin.agentgit_detect_format(transcript) == "my_agent_jsonl"

def test_parse_transcript(plugin, tmp_path):
    transcript = tmp_path / "test.jsonl"
    transcript.write_text('{"role": "user", "content": "Hello"}\n')

    result = plugin.agentgit_parse_transcript(
        transcript, "my_agent_jsonl"
    )

    assert result is not None
    assert len(result.prompts) == 1
```

## Examples

See the built-in plugins for complete examples:

- [`src/agentgit/formats/claude_code.py`](src/agentgit/formats/claude_code.py) - Claude Code plugin
- [`src/agentgit/formats/codex.py`](src/agentgit/formats/codex.py) - Codex plugin

## Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check existing plugins for implementation patterns
- The pluggy documentation covers advanced hook features
