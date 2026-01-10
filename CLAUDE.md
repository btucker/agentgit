# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests (requires 75% coverage)
uv run pytest

# Run a single test file
uv run pytest tests/plugins/claude_code/test_claude_code.py

# Run a specific test
uv run pytest tests/test_cli.py::TestCLI::test_help -v

# Run the CLI
uv run agentgit --help
uv run agentgit process session.jsonl -o ./output
uv run agentgit --watch session.jsonl
```

## Architecture

agentgit converts agent transcripts (like Claude Code JSONL) into git repositories where each file operation becomes a commit with rich metadata including the triggering prompt and assistant reasoning.

### Agent Plugin System (pluggy-based)

Agent plugins enable support for different AI coding assistant transcript formats. The system is built on pluggy and supports:

- **Built-in plugins**: claude_code, codex (registered in `plugins.py`)
- **Pip-installed plugins**: Auto-discovered via entry points
- **Config-file plugins**: Registered via `agentgit config agents add` (stored in `~/.agentgit/config.yml`)

Key hooks in `src/agentgit/plugins.py`:

- `agentgit_detect_format(path)` - Auto-detect transcript format
- `agentgit_parse_transcript(path, format)` - Parse into `Transcript` object
- `agentgit_extract_operations(transcript)` - Extract `FileOperation` list
- `agentgit_enrich_operation(operation, transcript)` - Add prompt/context metadata
- `agentgit_discover_transcripts(project_path)` - Find transcripts for a project

Plugins implement hooks with `@hookimpl` decorator. See `formats/claude_code.py` for the reference implementation.

Everything agent format-specific should go in the respective plugin.

**See [AGENT_PLUGINS.md](AGENT_PLUGINS.md) for the complete guide to creating agent plugins.**

### Core Data Flow

1. `parse_transcript()` - Detects format, parses file, extracts and enriches operations
2. `build_repo()` - Creates git repo from `FileOperation` list (supports incremental updates)
3. `transcript_to_repo()` - Convenience function combining both

### Key Data Structures (`core.py`)

- `Prompt` - User prompt with md5-based `prompt_id` for stable identification
- `FileOperation` - Single file change (WRITE/EDIT/DELETE) with metadata
- `AssistantContext` - Reasoning/thinking that explains why a change was made
- `Transcript` - Complete parsed transcript with entries, prompts, and operations

### Git Builder (`git_builder.py`)

- Normalizes file paths (strips common prefix)
- Creates rich commit messages with git trailers (Tool-Id, Timestamp, Prompt-Id, etc.)
- Supports incremental builds (skips already-processed operations by Tool-Id)
- Can interleave source repo commits by timestamp

### Project Naming Convention

Paths are encoded as `-Users-name-project` (leading slash becomes leading dash). This matches Claude Code's `~/.claude/projects/` naming and allows consistent project identification across different transcript sources.

## Test Organization

Agent plugin tests go in `tests/plugins/{plugin_name}/`.

## Development Process

Follow TDD: write a failing test first, confirm it fails, then write the code to make it pass.
