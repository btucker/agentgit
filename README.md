# agentgit

Turn AI agent transcripts into git repositories.

## The Problem

You spent an hour with Claude Code or Codex making changes across your codebase. You can easily see *what* changed—just diff your working directory. But which changes came from which prompts? Why did the agent modify that file? What was it trying to accomplish?

The code changes are visible. The context behind them is lost.

## The Solution

agentgit converts agent transcripts into git repositories, connecting each change to the prompt that caused it:

- **Prompt → Merge commit**: See all changes from a single request
- **Response → Commit**: Each logical step becomes a commit with the agent's reasoning
- **`git log`, `git blame`, `git show`**: Standard tools reveal the story

```
$ agentgit show a1b2c3d
commit a1b2c3d
Prompt: "Add user authentication"

    Implement auth module (auth.py, middleware.py)

    Context: I'll add JWT-based authentication with
    middleware to protect the API routes...

$ agentgit log --first-parent --oneline
a1b2c3d Add user authentication
x9y8z7w Fix database connection bug
f3e4d5c Initial setup
```

Now `git blame` tells you not just what changed, but *why*.

## Installation

```bash
uv tool install agentgit
# or: pip install agentgit
```

## Usage

```bash
# Process transcripts for current project
agentgit

# Interactive transcript picker
agentgit discover

# Process specific transcript
agentgit session.jsonl -o ./output

# Watch mode - auto-commit as transcript grows
agentgit --watch

# Git commands work directly
agentgit log --oneline -10
agentgit diff HEAD~2..HEAD
agentgit show abc123
```

## Supported Agents

```bash
$ agentgit agents
claude_code: Claude Code JSONL transcripts
codex: OpenAI Codex CLI JSONL transcripts
```

## How It Works

agentgit reads transcript files from standard locations:
- Claude Code: `~/.claude/projects/`
- Codex: `~/.codex/sessions/`

It parses file operations (writes, edits, deletes) and reconstructs them as git commits, preserving timestamps and grouping changes by prompt.

**Git structure:**

```
○ Merge: "Add user authentication" [prompt #a1b2c3d4]
|\
| ○ Implement auth module (auth.py, middleware.py)
| ○ Add login templates (login.html, styles.css)
|/
○ Merge: "Fix database connection bug" [prompt #x9y8z7w6]
|\
| ○ Fix connection pooling (db.py)
|/
○ Initial commit
```

**Commit messages include context:**

```
Refactor auth to use dependency injection

Modified: auth.py, middleware.py
Created: injection.py

Context:
I'll refactor the auth module to use dependency injection for better testability.

Prompt-Id: a1b2c3d4e5f67890abcdef1234567890
```

## Programmatic Usage

```python
import agentgit

repo, repo_path, transcript = agentgit.transcript_to_repo("session.jsonl")

for prompt in transcript.prompts:
    print(f"[{prompt.short_id}] {prompt.text[:50]}...")
```

## Adding Format Plugins

```python
from agentgit import hookimpl

class MyFormatPlugin:
    @hookimpl
    def agentgit_detect_format(self, path):
        return "my_format" if path.suffix == ".myformat" else None

    @hookimpl
    def agentgit_parse_transcript(self, path, format):
        if format != "my_format":
            return None
        # Return Transcript object
```

Register in `pyproject.toml`:

```toml
[project.entry-points."agentgit"]
my_format = "my_package:MyFormatPlugin"
```

## License

MIT
