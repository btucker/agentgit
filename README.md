# agentgit

See the context behind every line of code written by coding agents.

After a session with Claude Code or Codex, you can diff to see *what* changed—but not *why*. Which prompt triggered that refactor? What was the agent thinking when it modified that line?

agentgit transforms raw agent transcripts into a **structured transcript**—a separate git history that preserves the full story without touching your codebase.

**Each coding session becomes its own branch.** Each user prompt becomes a merge commit, with the agent's individual turns (thinking + changes) as child commits:

```
Your agentgit repo:

  main
   │
   └─── session/claude-code/add-user-auth
   │     │
   │     ├─ [MERGE] Prompt: "Add user authentication"
   │     │   ├─ Create auth.py with JWT utilities
   │     │   │  Context: I'll implement JWT token generation...
   │     │   ├─ Add authentication middleware
   │     │   │  Context: Creating middleware to verify tokens...
   │     │   └─ Update login component with auth flow
   │     │      Context: Integrating the auth into the UI...
   │     │
   │     └─ [MERGE] Prompt: "Add password reset"
   │         ├─ Create reset token generator
   │         └─ Add reset email template
   │
   └─── session/claude-code/fix-database-bugs
   │     │
   │     └─ [MERGE] Prompt: "Fix connection timeout"
   │         ├─ Debug connection pool settings
   │         │  Context: Found the pool size was too small...
   │         └─ Add retry logic with exponential backoff
   │            Context: Implementing retries to handle transient failures...
   │
   └─── session/codex/refactor-api
         │
         └─ [MERGE] Prompt: "Refactor payment endpoints"
             ├─ Extract payment handlers to separate module
             ├─ Consolidate error handling
             └─ Add request validation middleware
```

Each commit preserves the agent's reasoning and the full context:

```
$ agentgit log session/claude-code/add-user-auth --oneline
a1b2c3d Update login component
x9y8z7w Add middleware for protected routes
f3e4d5c Implement JWT authentication

$ agentgit show a1b2c3d
commit a1b2c3d
Prompt: "Add user authentication"

    Implement auth module (auth.py, middleware.py)

    Context: I'll add JWT-based authentication with
    middleware to protect the API routes...
```

**Use `agentgit blame` to see the agent's reasoning inline:**

```bash
$ agentgit blame auth.py
f3e4d5c (Agent        2024-01-15) def generate_jwt_token(user_id):
         → I'll implement JWT token generation using HS256 algorithm
f3e4d5c (Agent        2024-01-15)     payload = {"user_id": user_id, "exp": ...}
         → I'll implement JWT token generation using HS256 algorithm
f3e4d5c (Agent        2024-01-15)     return jwt.encode(payload, SECRET_KEY)
         → I'll implement JWT token generation using HS256 algorithm
x9y8z7w (Agent        2024-01-15) def verify_token(token):
         → Creating middleware to verify tokens before allowing access

# Show the full commit message with the original prompt
$ agentgit show f3e4d5c
commit f3e4d5c
Prompt: "Add user authentication"

    Create auth.py with JWT utilities

    Context: I'll implement JWT token generation using HS256 algorithm.
    The SECRET_KEY comes from environment variables for security...

# Use -L to blame specific line ranges
$ agentgit blame auth.py -L 10,20
```

Now `agentgit blame` shows not just *what* changed, but *why* the agent made that change and *what you asked for*—all inline, no extra commands needed. View all sessions with `agentgit branch`, compare approaches with `git diff session/A session/B`, or explore individual session histories.

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

## Transcript Enhancement

agentgit can enhance the structured transcript using either heuristic rules or AI:

```bash
# Use rules-based enhancement (fast, no AI)
agentgit --enhancer rules

# Use LLM for AI-powered enhancement
agentgit --enhancer llm --llm-model claude-cli-haiku
```

**Available enhancers:**

- `rules` - Uses heuristics to generate entries from prompts and context. Fast, no external dependencies.
- `llm` - Uses LLM to generate intelligent transcript entries. Requires `pip install 'agentgit[llm]'` which installs `llm` and `llm-claude-cli`.

The `llm` enhancer uses efficient batch processing - entries are generated in batched calls.

**Preferences are saved per-project.** Once you set an enhancer, it's used automatically on future runs:

```bash
# First run: set the enhancer and model
agentgit --enhancer llm --llm-model claude-cli-haiku

# Future runs: settings are remembered
agentgit  # Uses llm with claude-cli-haiku automatically
```

Settings are stored in git config (`agentgit.enhancer`, `agentgit.llmModel`).

## How It Works

agentgit reads coding agent transcripts and builds a git history where each commit preserves the full context—prompt, reasoning, and changes together.

It creates a **separate repository** that shares content with your code repo:

```
~/.agentgit/projects/<repo-id>/    # Structured transcript
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

**Transcript entries include full context:**

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

Enhancer plugins generate transcript entry summaries. They can use AI, heuristics, or any other approach:

```python
from agentgit import hookimpl

class MyEnhancerPlugin:
    @hookimpl
    def agentgit_get_enhancer_info(self):
        return {
            "name": "my_enhancer",
            "description": "My custom transcript enhancer",
        }

    @hookimpl
    def agentgit_enhance_prompt_summary(self, prompt, turns, enhancer, model):
        if enhancer != "my_enhancer":
            return None
        # Generate an entry summary from the prompt and turns
        return f"Implement: {prompt.text[:50]}"

    @hookimpl
    def agentgit_enhance_turn_summary(self, turn, prompt, enhancer, model):
        if enhancer != "my_enhancer":
            return None
        # Generate an entry summary for a single assistant turn
        files = turn.files_created + turn.files_modified
        return f"Update {', '.join(files[:3])}"
```

Register in `pyproject.toml`:

```toml
[project.entry-points."agentgit"]
my_enhancer = "my_package:MyEnhancerPlugin"
```

Use with: `agentgit --enhancer my_enhancer`

## License

MIT
