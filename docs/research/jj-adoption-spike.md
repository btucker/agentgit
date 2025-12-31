# Research Spike: jj (Jujutsu) vs Git for agentgit

## Executive Summary

**Recommendation: Stay with Git for now, but monitor jj development.**

While jj offers compelling conceptual advantages for agentgit's use case, the practical barriers (no Python bindings, immature programmatic API) outweigh the benefits. The current GitPython-based implementation is sufficient and well-understood.

---

## What is jj (Jujutsu)?

[Jujutsu](https://github.com/jj-vcs/jj) is a Git-compatible version control system developed at Google. It reimagines VCS workflows while maintaining full Git compatibility—commits created with jj look like regular Git commits and can be pushed to any Git remote.

Key innovations:
- **Working copy as commit**: No staging area; every change automatically becomes a commit
- **First-class conflicts**: Conflicts are stored in commits, not blocking operations
- **Automatic rebasing**: Modifying a commit automatically rebases all descendants
- **Operation log with undo**: Every operation is recorded and reversible
- **Stable change IDs**: Commits have stable identifiers that survive rewrites

---

## agentgit's Current Git Usage

From analyzing `git_builder.py` (961 lines), agentgit uses Git as a **versioned archive with rich metadata**:

### Core Operations
1. **Per-operation commits** with full prompt text and assistant reasoning in trailers
2. **Incremental builds** using commit message trailers (`Tool-Id`, `Timestamp`) for deduplication
3. **Merge-based hierarchy** grouping operations by user prompt (feature branches per prompt)
4. **Worktree management** for isolated history branches
5. **Source commit interleaving** blending agent changes with real repo commits

### Key Implementation Details
- Uses GitPython (`git.Repo`) for all operations
- Rich commit messages with custom trailers (Prompt-Id, Tool-Id, Operation, File, etc.)
- In-memory file state tracking for incremental updates
- Regex-based trailer parsing for deduplication

---

## Potential Benefits of jj for agentgit

### 1. Automatic Rebasing ✅
**Problem in current git_builder.py**: Merge conflicts during `build_from_prompt_responses()` are logged and skipped (lines 664-674), potentially leaving incomplete history.

**jj advantage**: Conflicts are recorded in commits rather than blocking operations. Would allow building complete history even with conflicts, resolving later.

### 2. Operation Log / Undo ✅
**Problem**: If agentgit build fails partway, recovery requires manual git operations.

**jj advantage**: Every operation is recorded with snapshots. Could easily undo/retry partial builds.

### 3. Working Copy as Commit ✅
**Problem**: Current code manages file states in memory (`file_states` dict, lines 802-821), which could hit memory limits on large repos.

**jj advantage**: Working copy is always a commit, so state is naturally tracked on disk.

### 4. Stable Change IDs ✅
**Problem**: Tool-Id is used for deduplication, but relies on transcript providing unique IDs. Timestamp fallback is fragile.

**jj advantage**: jj's change IDs are stable across rewrites, providing a more robust deduplication mechanism.

### 5. No Staging Area Complexity ✅
**Problem**: Current code uses `repo.index.add()` and manual staging.

**jj advantage**: Simpler model—just modify files and they're part of the commit.

---

## Barriers to jj Adoption

### 1. No Python Bindings ❌ (Critical)
jj is written in Rust. The library (`jj-lib`) is a Rust crate with no Python bindings.

**Current state**:
- `jj-lib` crate exists on [crates.io](https://crates.io/crates/jj-lib)
- No `pyjj` or equivalent Python wrapper
- Would require subprocess calls to `jj` CLI

**Impact**: agentgit is Python-based using GitPython. Switching to jj would require either:
- Rewriting in Rust (major undertaking)
- Calling `jj` via subprocess (fragile, slow, loses type safety)
- Creating Python bindings via PyO3 (significant investment)

### 2. Immature Programmatic API ❌ (Critical)
jj is CLI-focused. There's no stable programmatic API for external tools.

**Current state**:
- [FR #3219](https://github.com/martinvonz/jj/issues/3219): Proposal for `jj api` command (gRPC-based) - still in draft
- [FR #5662](https://github.com/jj-vcs/jj/issues/5662): Request for `--json` output on all commands - not implemented
- Parsing CLI output is acknowledged as "brittle" by jj developers

**Impact**: Building agentgit on jj would mean depending on unstable CLI output formats.

### 3. GitPython Feature Parity ❌
GitPython provides:
- Direct repository manipulation
- Commit traversal and metadata parsing
- Index manipulation
- Worktree management

jj CLI equivalent operations would be slower (process spawning) and less flexible.

### 4. Experimental Status ⚠️
From jj README: "This is an **experimental** VCS" with "work-in-progress features, suboptimal UX, and workflow gaps."

Missing features include:
- Git submodule support (relevant if agentgit processes repos with submodules)
- Some performance issues noted in large repos

### 5. Team Learning Curve ⚠️
jj has different mental models:
- Anonymous branches by default
- "Bookmarks" instead of branches
- Different conflict resolution workflow

---

## Comparison Matrix

| Aspect | Git (current) | jj (potential) |
|--------|---------------|----------------|
| Python library | GitPython (mature) | None (subprocess only) |
| Programmatic API | Full control | CLI-only |
| Commit metadata | Trailers work well | Same (Git-compatible) |
| Conflict handling | Blocks operations | First-class, non-blocking |
| History rewriting | Manual rebases | Automatic |
| Operation undo | Manual | Built-in |
| Ecosystem maturity | Decades | ~3 years |
| Memory efficiency | Manual state tracking | Better (working-copy-as-commit) |
| Deduplication | Custom trailer parsing | Could use change IDs |

---

## Alternative: Hybrid Approach

If jj benefits are compelling, a hybrid approach could work:

1. **Keep GitPython** for core operations
2. **Use jj for specific features** via subprocess:
   - `jj op undo` for recovery
   - `jj rebase` for conflict-tolerant rebasing
3. **Git backend** ensures interoperability

However, this adds complexity without significant gain given current pain points are manageable.

---

## What Would Change the Calculus?

Revisit this decision if any of these occur:

1. **Python bindings appear**: A `pyjj` or PyO3 wrapper would enable direct integration
2. **Structured API ships**: `jj api` or comprehensive `--json` support
3. **GitPython limitations emerge**: Memory issues with large repos, conflict handling problems
4. **jj reaches 1.0**: Stability commitment and mature API

---

## Conclusion

**Stay with Git/GitPython** because:
1. No Python bindings for jj makes integration impractical
2. No stable programmatic API for jj
3. Current implementation works and is well-understood
4. jj's advantages don't address agentgit's actual pain points

**Monitor jj development** for:
- Python bindings or FFI
- `jj api` structured output feature
- 1.0 stability release

---

## Sources

- [jj-vcs/jj GitHub Repository](https://github.com/jj-vcs/jj)
- [jj-lib on crates.io](https://crates.io/crates/jj-lib)
- [Jujutsu: The Future of Version Control](https://medium.com/@shrmtv/jujutsu-150945f97753)
- [jj init - Chris Krycho](https://v5.chriskrycho.com/essays/jj-init/)
- [Tony Finn's jj blog post](https://tonyfinn.com/blog/jj/)
- [FR: Structured API #3219](https://github.com/martinvonz/jj/issues/3219)
- [FR: JSON output #5662](https://github.com/jj-vcs/jj/issues/5662)
- [What I've learned from jj](https://zerowidth.com/2025/what-ive-learned-from-jj/)
