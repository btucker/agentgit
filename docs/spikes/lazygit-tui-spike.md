# Spike: Lazygit as TUI for agentgit

**Date:** 2026-01-07
**Status:** Spike Complete
**Recommendation:** Build native Python TUI with Textual, inspired by lazygit's UX

## Summary

Investigated whether lazygit could be used (directly, forked, or as inspiration) for agentgit's TUI needs. The conclusion is that **forking lazygit is not recommended** due to language mismatch, but **building a Python TUI with Textual using lazygit's UX patterns** is the recommended approach.

## Context

agentgit needs a TUI to:
1. Browse and filter agent sessions (Claude Code, Codex, etc.)
2. View rich commit history with agent context (prompts, reasoning)
3. Navigate between sessions and their generated commits
4. Watch for new sessions and process them interactively
5. Run blame with inline agent attribution

## Option 1: Use Lazygit Directly

### How it Would Work
- Launch lazygit pointed at agentgit-generated repos
- Use custom commands to add agentgit-specific functionality

### Pros
- Zero development effort for base git UI
- Excellent, battle-tested UX
- Active community and maintenance

### Cons
- **Session management not possible**: Can't list/filter Claude Code sessions
- **No custom panels**: Can't show prompt/reasoning context inline
- **Limited to git operations**: agentgit needs transcript-level operations
- **Custom commands are shell-only**: Can't integrate Python plugin system

### Verdict: ❌ Not Viable
Lazygit is a git client, not an extensible platform. agentgit needs session-level operations that lazygit can't support.

---

## Option 2: Fork Lazygit

### How it Would Work
- Fork the lazygit Go codebase
- Add agentgit-specific panels (sessions, prompts, reasoning)
- Replace git command layer with agentgit operations

### Pros
- Proven architecture and UX patterns
- MIT licensed, fully open source
- Layered architecture (GUI / Commands / Terminal) is well-separated

### Cons
- **Language mismatch**: agentgit is Python; lazygit is Go
  - Would need to maintain two codebases or rewrite agentgit in Go
  - Can't reuse agentgit's plugin system (pluggy-based)
  - Can't call agentgit's transcript parsers directly
- **Heavy fork burden**:
  - gocui library is vendored and customized
  - Deep coupling between panels and git concepts
  - Would essentially be rewriting 60-70% of the code
- **Two-way sync nightmare**:
  - Upstream lazygit changes would conflict with agentgit changes
  - Risk of fork diverging and becoming unmaintainable

### Verdict: ❌ Not Recommended
The Go/Python mismatch makes this impractical. The effort to fork and adapt lazygit exceeds building a Python TUI from scratch.

---

## Option 3: Build Python TUI with Textual (Recommended) ✅

### How it Would Work
- Use [Textual](https://textual.textualize.io/) framework (from Rich creators)
- Copy lazygit's multi-panel UX design patterns
- Integrate directly with agentgit's Python codebase

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ agentgit TUI (Textual)                                          │
├─────────────┬───────────────────────────────────────────────────┤
│ Sessions    │  Commit Details                                   │
│ ─────────── │  ─────────────                                    │
│ > session-1 │  Author: Claude Code                              │
│   session-2 │  Date: 2026-01-07                                 │
│   session-3 │                                                   │
├─────────────┤  Prompt:                                          │
│ Commits     │  "Add user authentication"                        │
│ ─────────── │                                                   │
│ > abc123    │  Reasoning:                                       │
│   def456    │  "I'll create a new auth module..."               │
│   ghi789    │                                                   │
├─────────────┤  Diff:                                            │
│ Files       │  +def authenticate(user, password):               │
│ ─────────── │  +    ...                                         │
│   auth.py   │                                                   │
│   tests.py  │                                                   │
└─────────────┴───────────────────────────────────────────────────┘
```

### Key Panels (Lazygit-Inspired)

| Panel | Purpose | Lazygit Equivalent |
|-------|---------|-------------------|
| Sessions | List discovered transcripts | (new) |
| Commits | Show session's git commits | Commits panel |
| Files | Files changed in commit | Files panel |
| Diff | Show file changes | Main panel |
| Prompt | User prompt that triggered changes | (new) |
| Reasoning | Assistant's thinking/context | (new) |
| Command Log | Show running operations | Command log |

### Pros
- **Native integration**: Direct access to agentgit's plugin system, parsers, and git builder
- **Same language**: Python throughout, easier to maintain
- **Textual is mature**:
  - Built by Rich creators (already a dependency)
  - 120 FPS renders, efficient delta updates
  - Works in terminal AND web browser
  - Active development, 250k+ PyPI downloads in Q1 2025
- **Lazygit UX patterns**: Copy the successful multi-panel, vim-style navigation
- **Extensible**: Can add agentgit-specific features without upstream constraints

### Cons
- More initial development effort than using lazygit directly
- Need to implement git visualization (though pygitzen shows it's doable)

### Prior Art: pygitzen
[pygitzen](https://github.com/SunnyTamang/pygitzen) is a Python lazygit clone built with Textual that proves this approach works:
- Multi-panel interface with vim navigation
- Uses Dulwich for git operations
- ~100 commits, actively maintained
- Could serve as reference implementation

---

## Implementation Approach

### Phase 1: Minimal Viable TUI
1. Session browser panel (list discovered transcripts)
2. Process session action (trigger existing `agentgit process`)
3. Basic commit list for selected session
4. Diff viewer for selected commit

### Phase 2: Rich Context
1. Prompt panel showing triggering user prompt
2. Reasoning panel showing assistant context
3. Keyboard navigation (vim-style j/k/h/l)
4. Search/filter sessions

### Phase 3: Advanced Features
1. Watch mode indicator (live updates)
2. Inline blame with agent attribution
3. Session comparison
4. Web mode (Textual's `textual serve`)

### Technical Details

```python
# Proposed structure
src/agentgit/
├── tui/
│   ├── __init__.py
│   ├── app.py           # Main Textual app
│   ├── screens/
│   │   ├── main.py      # Main multi-panel screen
│   │   └── session.py   # Session detail screen
│   ├── widgets/
│   │   ├── sessions.py  # Session list widget
│   │   ├── commits.py   # Commit list widget
│   │   ├── diff.py      # Diff viewer widget
│   │   └── context.py   # Prompt/reasoning widget
│   └── keybindings.py   # Vim-style navigation
```

### Dependencies to Add
```toml
[project.optional-dependencies]
tui = [
    "textual>=0.50.0",
]
```

---

## Decision Matrix

| Criteria | Use Lazygit | Fork Lazygit | Build with Textual |
|----------|-------------|--------------|-------------------|
| Development effort | None | Very High | Medium |
| Session management | ❌ | ✅ | ✅ |
| Custom panels | ❌ | ✅ | ✅ |
| Plugin integration | ❌ | ❌ | ✅ |
| Maintenance burden | Low | Very High | Medium |
| UX quality | Excellent | Good | Good (copy patterns) |
| Web support | ❌ | ❌ | ✅ |

---

## Recommendation

**Build a native Python TUI using Textual**, copying lazygit's proven UX patterns:

1. Multi-panel layout with tabbed sections
2. Vim-style keyboard navigation (j/k/h/l, q to quit)
3. Focus indicators with colored borders
4. Command log for operation feedback

This approach gives us:
- Full integration with agentgit's Python ecosystem
- Lazygit's excellent UX without the Go/Python mismatch
- Future web support via Textual's browser mode
- Sustainable maintenance with a single codebase

## Sources

- [Lazygit GitHub](https://github.com/jesseduffield/lazygit)
- [Lazygit Custom Commands](https://github.com/jesseduffield/lazygit/blob/master/docs/Custom_Command_Keybindings.md)
- [Textual Framework](https://textual.textualize.io/)
- [pygitzen - Python lazygit clone](https://github.com/SunnyTamang/pygitzen)
- [Lazygit Architecture - DeepWiki](https://deepwiki.com/jesseduffield/lazygit)
