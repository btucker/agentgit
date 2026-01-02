# agentgit

See the context behind every line of code written by coding agents.

After a session with Claude Code or Codex, you can diff to see *what* changed—but not *why*. Which prompt triggered that refactor? What was the agent thinking when it modified that line?

agentgit preserves the full story—in a separate repo that never touches your codebase. Each prompt becomes a merge commit, each response becomes commits with the agent's reasoning, and standard git tools reveal everything:

```
$ agentgit log --first-parent --oneline
a1b2c3d Add user authentication
x9y8z7w Fix database connection bug
f3e4d5c Initial setup

$ agentgit show a1b2c3d
commit a1b2c3d
Prompt: "Add user authentication"

    Implement auth module (auth.py, middleware.py)

    Context: I'll add JWT-based authentication with
    middleware to protect the API routes...
```

Now `git blame` tells you not just what changed, but why.

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

## Commit Message Enhancement

agentgit can generate better commit messages using either heuristic rules or AI:

```bash
# Use rules-based enhancement (fast, no AI)
agentgit --enhancer rules

# Use Claude Code for AI-powered messages
agentgit --enhancer claude_code
```

**Available enhancers:**

- `rules` - Uses heuristics to generate messages from prompts and context. Fast, no external dependencies.
- `claude_code` - Uses Claude Code to generate intelligent commit messages. Requires the `llm` extra: `pip install 'agentgit[llm]'`

The `claude_code` enhancer uses efficient batch processing - all commit messages are generated in a single call.

**Preferences are saved per-project.** Once you set an enhancer, it's used automatically on future runs:

```bash
# First run: set the enhancer
agentgit --enhancer claude_code

# Future runs: enhancer is remembered
agentgit  # Uses claude_code automatically
```

Settings are stored in git config (`agentgit.enhancer`, `agentgit.enhanceModel`).

## How It Works

agentgit reads coding agent transcripts and builds a git history where each commit preserves the full context—prompt, reasoning, and changes together.

It creates a **separate repository** that shares content with your code repo:

```
~/.agentgit/projects/<repo-id>/    # Provenance history
├── .git/
│   └── objects/info/alternates → your-repo/.git/objects
└── refs/heads/session/...
```

The repos share git's object store. Same content = same blob SHA = automatic correlation between your code and its history.

**Source transcripts** are read from standard locations:
- Claude Code: `~/.claude/projects/`
- Codex: `~/.codex/sessions/`

**Git structure** (prompts as merge commits):

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

**Commit messages include full context:**

```
Refactor auth to use dependency injection

Modified: auth.py, middleware.py
Created: injection.py

Context:
I'll refactor the auth module to use dependency injection for better testability.

Prompt-Id: a1b2c3d4e5f67890abcdef1234567890
Tool-Id: toolu_abc123
Timestamp: 2025-01-01T10:30:00Z
```

**Repo identification** uses the first commit SHA (12 chars) of your code repo, so it stays stable even if you move or rename the project.

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

## Adding Enhancer Plugins

Enhancer plugins generate commit messages. They can use AI, heuristics, or any other approach:

```python
from agentgit import hookimpl

class MyEnhancerPlugin:
    @hookimpl
    def agentgit_get_ai_enhancer_info(self):
        return {
            "name": "my_enhancer",
            "description": "My custom commit message enhancer",
        }

    @hookimpl
    def agentgit_enhance_merge_message(self, prompt, turns, enhancer, model):
        if enhancer != "my_enhancer":
            return None
        # Generate a commit message from the prompt and turns
        return f"Implement: {prompt.text[:50]}"

    @hookimpl
    def agentgit_enhance_turn_message(self, turn, prompt, enhancer, model):
        if enhancer != "my_enhancer":
            return None
        # Generate a commit message for a single assistant turn
        files = turn.files_created + turn.files_modified
        return f"Update {', '.join(files[:3])}"

    @hookimpl
    def agentgit_enhance_operation_message(self, operation, enhancer, model):
        if enhancer != "my_enhancer":
            return None
        # Generate a commit message for a single file operation
        return f"Modify {operation.filename}"
```

Register in `pyproject.toml`:

```toml
[project.entry-points."agentgit"]
my_enhancer = "my_package:MyEnhancerPlugin"
```

Use with: `agentgit --enhance --enhancer my_enhancer`

## License

MIT
