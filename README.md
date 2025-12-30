# agentgit - *Currently Under Initial Development*

Convert agent transcripts into git repositories with a structure that makes agent work easy to understand using standard git tools.

## Features

- **Structured Git History**: Each prompt becomes a merge commit; each assistant response becomes a commit grouping related file changes
- **Two-Level View**: `git log --first-parent` shows prompts, `git log` shows all operations
- **Rich Commit Messages**: Includes triggering prompt, assistant reasoning, and machine-parseable trailers
- **Git Passthrough**: Run any git command directly on the generated repo (`agentgit log`, `agentgit diff`, etc.)
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

## Git Structure

agentgit creates a repository structure designed for understanding agent work at multiple levels of detail:

```
○ Merge: "Add user authentication" [prompt #a1b2c3d4]
|\
| ○ Implement auth module (auth.py, middleware.py, config.py)
| ○ Add login templates (login.html, styles.css)
|/
○ Merge: "Fix database connection bug" [prompt #x9y8z7w6]
|\
| ○ Fix connection pooling (db.py, config.py)
|/
○ Initial commit
```

**How it works:**

- Each **user prompt** becomes a **merge commit** on the main branch
- Each **assistant response** becomes a **commit** grouping all file changes from that response
- Multiple file edits in one response stay together as a single logical commit

**Viewing history:**

```bash
# High-level view: just the prompts
agentgit log --first-parent --oneline
# a1b2c3d4 Add user authentication
# x9y8z7w6 Fix database connection bug

# Detailed view: all operations
agentgit log --oneline
# f7e8d9c0 Implement auth module
# b3a2c1d0 Add login templates
# a1b2c3d4 Add user authentication
# ...

# Everything for one prompt
agentgit log x9y8z7w6^..x9y8z7w6

# Diff for entire prompt
agentgit diff x9y8z7w6^..x9y8z7w6
```

## Git Passthrough

Any unrecognized command is passed through to git, running on the agentgit-created repository. File paths are automatically translated:

```bash
# View commit history
agentgit log --oneline -10

# View changes for a specific file (local path works)
agentgit log --follow -p src/myproject/config.py

# Show a specific commit
agentgit show HEAD

# Search commit messages
agentgit log --grep="config"

# Compare prompts
agentgit diff HEAD~2..HEAD~1
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

**Assistant response commits** group related file changes:

```
Refactor auth to use dependency injection

Modified: auth.py, middleware.py
Created: injection.py

Context:
I'll refactor the auth module to use dependency injection for better testability.

Prompt-Id: a1b2c3d4e5f67890abcdef1234567890
Timestamp: 2025-01-01T10:00:05.000Z
```

**Prompt merge commits** summarize the work:

```
Add user authentication

Prompt #a1b2c3d4:
Add user authentication with login, logout, and session management.
Support both email/password and OAuth providers.

Prompt-Id: a1b2c3d4e5f67890abcdef1234567890
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
