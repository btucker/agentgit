# Session Structure Design

## Overview

agentgit creates a separate git repository that shares content with the code repository, enabling AI provenance tracking without modifying the code repo.

## Core Concept: Two Repos, Shared Content

```
code-repo/                    # The real codebase
├── .git/
│   └── objects/              # Git object store
└── src/...

~/.agentgit/projects/<repo-id>/   # AI provenance history
├── .git/
│   └── objects/info/alternates → code-repo/.git/objects
└── refs/heads/session/...
```

The repos share git's object store via alternates. Same content = same blob SHA = automatic correlation without custom matching logic.

## Repo Identification

Each code repo is identified by its **first commit SHA** (truncated to 12 characters):

```bash
git rev-list --max-parents=0 HEAD
# → a1b2c3d4e5f6...
```

This provides:
- **Stability**: Doesn't change when repo is moved/renamed
- **Portability**: Same ID when cloned to different machines
- **Simplicity**: No path encoding edge cases

### Directory Layout

```
~/.agentgit/
└── projects/
    ├── a1b2c3d4e5f6/         # First code repo
    │   └── .git/
    │       ├── objects/info/alternates
    │       ├── config        # Stores agentgit.coderepo = /original/path
    │       └── refs/heads/session/...
    │
    └── f6e5d4c3b2a1/         # Another code repo
        └── .git/
```

## Session Branch Structure

Each Claude Code session becomes a branch:

```
session/<session-id>
├── merge: "fix login timeout"           ← Prompt (level 1)
│   ├── "Edit src/auth.py"               ← Operation (level 2)
│   └── "Edit src/config.py"
├── merge: "add retry logic"
│   └── "Create src/retry.py"
```

**Navigation:**
- `git log --first-parent session/abc` → shows prompts only
- `git log session/abc` → shows all operations

## Commit Metadata

Every commit includes trailers for queryability:

```
<Summary line>

<Body: reasoning/context>

Commit-Type: prompt | operation
Session-Id: <session-id>
Prompt-Id: <md5-hash>
Tool-Id: <unique-id>
Timestamp: <iso-8601>
File: <path>                   # (operations only)
Operation: write | edit | delete  # (operations only)
```

## Blame Flow

To find AI provenance for a line of code:

```bash
# 1. Blame in code repo → get blob SHA
$ git -C code-repo blame --porcelain src/auth.py | grep "^[a-f0-9]"
# → blob abc789

# 2. Find that blob in agentgit repo
$ git -C agentgit-repo log --all --find-object=abc789 --format="%H %s"
# → commit 123abc "Edit src/auth.py"

# 3. Get full provenance
$ git -C agentgit-repo show 123abc
# → Session-Id, Prompt-Id, reasoning, etc.
```

Wrapped in a single command:

```bash
$ agentgit blame src/auth.py:16
# → Session: abc, Prompt: "fix login timeout"
# → Reasoning: "Increased from 10s to handle slow networks"
```

## Handling Concurrent Sessions

Sessions are independent branches - no conflicts:

```
session/abc ──●──●──●
session/def ──●──●
```

Since sessions represent parallel realities (the AI in session A didn't see session B's changes), keeping them as separate branches is semantically correct.

## Why This Design?

### Why separate repos?
- Code repo stays clean (no extra branches/notes)
- Privacy: agentgit repo can be private/local
- No interference with existing workflows (CI, branch policies)

### Why shared object store?
- Git's content-addressing provides automatic correlation
- No custom matching/indexing needed
- Same blob SHA = same content

### Why first commit SHA as identifier?
- Survives repo moves/renames
- Same ID across clones
- No path encoding complexity

### Why prompts as merge commits?
- `--first-parent` gives clean prompt-level history
- Full detail available when needed
- Matches mental model of "prompt = unit of work"

## Commands

```bash
# Initialize agentgit for current repo
agentgit init

# Build sessions from transcripts
agentgit build

# Blame with provenance
agentgit blame <file>:<line>

# List sessions
agentgit sessions

# Show prompt history for a session
git log --first-parent session/<id>
```

## Implementation Notes

### Setup (agentgit init)

```python
def init_agentgit_repo(code_repo: Path) -> Path:
    repo_id = get_repo_id(code_repo)
    agentgit_repo = Path.home() / ".agentgit" / "projects" / repo_id

    # Create bare repo
    subprocess.run(["git", "init", "--bare", str(agentgit_repo)])

    # Set up alternates
    alternates = agentgit_repo / "objects" / "info" / "alternates"
    alternates.write_text(str(code_repo / ".git" / "objects"))

    # Store original path for reference
    subprocess.run([
        "git", "-C", str(agentgit_repo),
        "config", "agentgit.coderepo", str(code_repo)
    ])

    return agentgit_repo
```

### Lookup

```python
def get_repo_id(code_repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(code_repo), "rev-list", "--max-parents=0", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip()[:12]

def get_agentgit_repo(code_repo: Path) -> Path:
    repo_id = get_repo_id(code_repo)
    return Path.home() / ".agentgit" / "projects" / repo_id
```

## Edge Cases

| Case | Handling |
|------|----------|
| File modified after AI wrote it | Blob SHA differs → no match (correct) |
| Same content in multiple sessions | Multiple matches → show all with timestamps |
| Code repo rebased | Blob SHAs unchanged → matching still works |
| Repo moved/renamed | First commit SHA unchanged → still finds agentgit repo |
| Multiple roots (orphan branches) | Use primary branch's root commit |
