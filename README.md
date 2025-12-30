# agentgit

Convert agent transcripts into git repositories where each file operation becomes a commit with rich metadata.

## Features

- **Transcript to Git**: Parse agent transcripts and create a git repo with one commit per file operation
- **Rich Commit Messages**: Each commit includes the triggering prompt, assistant reasoning, and machine-parseable trailers
- **Incremental Updates**: Re-run on the same transcript to add only new operations
- **Watch Mode**: Monitor a transcript file and auto-commit new operations as they're added
- **Auto-Discovery**: Automatically find transcripts for the current project
- **Pluggable Formats**: Extensible plugin system for different transcript formats

## Installation

```bash
# Install globally with uv
uv tool install agentgit

# Or use directly without installing
uvx agentgit

# Traditional pip install
pip install agentgit
```

## Quick Start

```bash
# Auto-discover and process all transcripts for current project
agentgit

# Process a specific transcript
agentgit session.jsonl -o ./output

# Watch mode - auto-commit as the transcript grows
agentgit --watch session.jsonl

# List prompts from all project transcripts
agentgit prompts

# List file operations from all project transcripts
agentgit operations

# Discover transcripts for current project
agentgit discover
```

## Git Passthrough

Any unrecognized command is passed through to git, running on the agentgit-created repository. This lets you use familiar git commands to explore the generated history:

```bash
# View commit history
agentgit log --oneline -10

# View recent changes
agentgit diff HEAD~5..HEAD

# Show a specific commit
agentgit show HEAD

# Search commit messages
agentgit log --grep="config"

# View file history
agentgit log --follow -p -- src/config.py
```

## Programmatic Usage

```python
import agentgit

# Parse and build in one step
repo, repo_path, transcript = agentgit.transcript_to_repo("session.jsonl")

# Or parse and build separately
transcript = agentgit.parse_transcript("session.jsonl")
repo, repo_path, mapping = agentgit.build_repo(transcript.operations)

# Access parsed data
for prompt in transcript.prompts:
    print(f"[{prompt.short_id}] {prompt.text[:50]}...")

for op in transcript.operations:
    print(f"{op.operation_type.value}: {op.file_path}")
```

## Commit Message Format

Each commit includes structured metadata:

```
Edit config.py

Prompt #a1b2c3d4:
Update the database connection to use environment variables.

Context:
I'll modify the config to read from environment variables for better security.

Prompt-Id: a1b2c3d4e5f67890abcdef1234567890
Operation: edit
File: /project/src/config.py
Timestamp: 2025-01-01T10:00:05.000Z
Tool-Id: toolu_001
```

## Supported Formats

- **claude_code**: Claude Code JSONL transcripts (`~/.claude/projects/`)

## Adding Format Plugins

Create a plugin class with `@hookimpl` decorated methods:

```python
from agentgit import hookimpl

class MyFormatPlugin:
    @hookimpl
    def agentgit_detect_format(self, path):
        if path.suffix == ".myformat":
            return "my_format"
        return None

    @hookimpl
    def agentgit_parse_transcript(self, path, format):
        if format != "my_format":
            return None
        # Parse and return Transcript object
        ...
```

Register via entry points in `pyproject.toml`:

```toml
[project.entry-points."agentgit"]
my_format = "my_package:MyFormatPlugin"
```

## License

MIT
