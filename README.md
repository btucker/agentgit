# agentgit

See the context behind every line of code written by coding agents.

After a session with Claude Code or Codex, you can diff to see *what* changed—but not *why*. Which prompt triggered that refactor? What was the agent thinking when it modified that line?

agentgit transforms raw agent transcripts into a **structured transcript**—a separate git history that preserves the full story without touching your codebase.

## How agentgit structures history

**User prompts become merge commits** — each prompt groups all the work done to address it.

**Scenes become feature commits** — each logical unit of work becomes one commit. What constitutes a "scene" depends on the agent:

- **Claude Code**: Uses `TodoWrite` boundaries when available. Each todo item marked in_progress → completed becomes a commit.
- **Codex**: Uses `functions.update_plan` step boundaries. Each plan step becomes a commit.
- **Fallback**: If no planning tool is used, groups by assistant turns.

**Context flows forward** — if the agent says "I'll refactor the auth module..." in one message and then edits files in the next, both messages' context appears in the commit.

```
Your agentgit repo:

  main
   │
   └─── session/claude-code/add-user-auth
   │     │
   │     ├─ [MERGE] Prompt: "Add user authentication"
   │     │   │
   │     │   ├─ ✓ Implement JWT authentication
   │     │   │    (creates auth.py, middleware.py)
   │     │   │    Thinking: I'll add JWT-based auth with protected routes...
   │     │   │
   │     │   └─ ✓ Add login form with validation
   │     │        (modifies login.tsx, adds validation.ts)
   │     │        Thinking: Adding client-side validation before API calls...
   │     │
   │     └─ [MERGE] Prompt: "Add password reset"
   │         │
   │         └─ ✓ Implement password reset flow
   │              (creates reset.py, email_templates/reset.html)
   │
   └─── session/codex/refactor-api
         │
         └─ [MERGE] Prompt: "Refactor payment endpoints"
             │
             ├─ ✓ Extract payment handlers to separate module
             └─ ✓ Add request validation middleware
```

Each commit preserves the agent's reasoning and the full context:

```
$ agentgit log session/claude-code/add-user-auth --oneline
a1b2c3d (HEAD) Prompt: "Add user authentication"
f3e4d5c ✓ Add login form with validation
x9y8z7w ✓ Implement JWT authentication

$ agentgit show x9y8z7w
commit x9y8z7w
✓ Implement JWT authentication

I'll add JWT-based authentication with middleware to protect the API routes.

## Thinking
The user wants authentication. I should:
1. Create a JWT utilities module for token generation/validation
2. Add middleware to protect routes

Created: auth.py, middleware.py
Modified: login.tsx

User Prompt: "Add user authentication"
---
Prompt-Id: a1b2c3d4
Todo-Item: Implement JWT authentication
Tool-Id: toolu_01ABC, toolu_01DEF
Timestamp: 2025-01-01T10:30:00Z
```

**Use `agentgit blame` to see which session and why:**

`agentgit blame` automatically detects if you're in your code repo and maps each line to its session:

```bash
$ cd ~/myproject  # Your actual code repo
$ agentgit blame auth.py
Using code repo: /Users/you/myproject

a1b2c3d session/claude-code/add-user-auth            def generate_jwt_token(user_id):
         → I'll implement JWT token generation using HS256 algorithm
a1b2c3d session/claude-code/add-user-auth                payload = {"user_id": user_id, ...}
         → I'll implement JWT token generation using HS256 algorithm
x9y8z7w session/claude-code/fix-auth-bugs                # Fixed expiration bug
         → The expiration was set to seconds instead of timestamp
a1b2c3d session/claude-code/add-user-auth                return jwt.encode(payload, SECRET_KEY)
         → Using SECRET_KEY from environment variables

# Blame a specific session branch
$ agentgit blame auth.py --session session/claude-code/add-user-auth

# Use -L to blame specific line ranges
$ agentgit blame auth.py -L 10,20
```

**How it works:** agentgit uses git's native blame on each session branch to find which commits wrote which lines. For each line in your code, it finds the earliest commit across all sessions that introduced that exact line content. Since both repos share git's object store via alternates, this is fast and accurate.

View all sessions with `agentgit branch`, compare approaches with `git diff session/A session/B`, or explore individual session histories.

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

### Convenience Commands

**agit** - Shorter alias for `agentgit`:
```bash
agit log --graph --oneline
agit show abc123
```

**alazygit** - Use [lazygit](https://github.com/jesseduffield/lazygit) with your agentgit repo:
```bash
alazygit              # Opens lazygit TUI for agentgit repo
alazygit --version    # Pass any lazygit args
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

The repos share git's object store via git alternates. **Same content = same blob SHA = automatic correlation between your code and its history.**

### How Blame Works: Native Git Blame

When you run `agentgit blame`:

1. **Blame each session branch** using git's native blame algorithm (Myers diff)
2. **Collect all line attributions** - which commit in which session wrote each line
3. **Find the earliest** - for each line in your code, return the oldest commit that wrote it
4. **Display inline** with session name and agent context

This approach uses git's built-in line-tracking capabilities rather than custom indexing. Git's blame algorithm already handles whitespace normalization, move detection, and accurate line provenance.

**Why this works:** Even when files evolve across sessions, git blame correctly tracks which commit introduced each line. By blaming all session branches and finding the earliest timestamp, we identify the original agent session that wrote each line of code.

**Source transcripts** are read from standard locations:
- Claude Code: `~/.claude/projects/`
- Codex: `~/.codex/sessions/`

**Git structure** (prompts as merge commits, scenes as feature commits):

```
○ Merge: "Add user authentication" [prompt #a1b2c3d4]
|\
| ○ ✓ Add login form with validation
| ○ ✓ Implement JWT authentication
|/
○ Merge: "Fix database connection bug" [prompt #x9y8z7w6]
|\
| ○ ✓ Fix connection pooling with retry logic
|/
○ Initial commit
```

**Scene commits include full context:**

```
✓ Implement JWT authentication

I'll add JWT-based authentication with middleware to protect the API routes.

## Thinking
The user wants authentication. I should create a JWT utilities module
for token generation/validation, then add middleware to protect routes.

Created: auth.py, middleware.py
Modified: login.tsx

User Prompt: "Add user authentication"
---
Prompt-Id: a1b2c3d4
Todo-Item: Implement JWT authentication
Tool-Id: toolu_01ABC, toolu_01DEF
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
