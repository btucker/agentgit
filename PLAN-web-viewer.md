# Web-Based Git Viewer for agentgit

## Overview

Build a custom web-based viewer that displays agentgit repositories with a three-pane interface inspired by the [claude-code-transcripts PR #23](https://github.com/simonw/claude-code-transcripts/pull/23), but tailored to agentgit's architecture.

## Goals

1. **Visualize file evolution** - Show how files changed across prompts/turns
2. **Display rich metadata** - Prompts, assistant reasoning, tool IDs, timestamps
3. **Git blame-style annotations** - Link code lines to the prompts that created them
4. **Interactive navigation** - Click between code and transcript seamlessly

## Architecture Decision: Static vs Dynamic

### Option A: Static HTML Generation (Recommended)
- Generate self-contained HTML/JS/CSS files
- No server required for viewing
- Can be hosted on GitHub Pages, S3, or served locally
- Matches agentgit's "build artifact" philosophy

### Option B: Live Server
- Flask/FastAPI backend
- Real-time updates possible
- More complex deployment

**Recommendation:** Start with static generation (Option A), similar to how agentgit builds git repos as artifacts.

---

## Implementation Plan

### Phase 1: Core Data API (Python)

**New module: `src/agentgit/viewer_data.py`**

Extract structured JSON from agentgit repos for the web viewer:

```python
@dataclass
class ViewerData:
    files: dict[str, FileViewerData]      # path -> file data
    prompts: list[PromptViewerData]        # all prompts with metadata
    commits: list[CommitViewerData]        # commit timeline
    blame: dict[str, list[BlameRange]]     # path -> blame annotations

@dataclass
class FileViewerData:
    path: str
    final_content: str
    operations: list[OperationSummary]     # history of changes
    blame_ranges: list[BlameRange]

@dataclass
class BlameRange:
    start_line: int
    end_line: int
    prompt_id: str
    prompt_num: int                        # "User Prompt #1", "#2", etc.
    tool_id: str
    color_index: int                       # for consistent coloring
    assistant_context: str | None          # reasoning snippet

@dataclass
class PromptViewerData:
    prompt_id: str
    prompt_num: int
    text: str
    timestamp: str
    files_modified: list[str]
    turns: list[TurnSummary]
```

**Key functions:**
- `build_viewer_data(repo_path: Path) -> ViewerData`
- `compute_blame_ranges(repo: Repo, file_path: str) -> list[BlameRange]`
- `export_viewer_json(data: ViewerData, output_path: Path)`

### Phase 2: HTML/JS Generator

**New module: `src/agentgit/html_generator.py`**

Generate static HTML with embedded data:

```
output/
├── index.html              # Main viewer app
├── data/
│   ├── viewer-data.json    # Core data (or chunked for large repos)
│   └── files/              # File contents (optional, for large repos)
├── assets/
│   ├── app.js              # Main application logic
│   ├── styles.css          # Styling
│   └── vendor/             # Third-party libs (diff viewer, syntax highlight)
```

**Template approach:**
- Bundle templates in package (`src/agentgit/templates/`)
- Use Jinja2 for HTML generation
- Inline small data, chunk large sessions (like PR #23's approach)

### Phase 3: Frontend Components

**Technology stack:**
- Vanilla JS or Preact (lightweight React alternative)
- [@git-diff-view/core](https://github.com/MrWangJustToDo/git-diff-view) for diff rendering
- [Prism.js](https://prismjs.com/) or [Shiki](https://shiki.style/) for syntax highlighting
- CSS Grid for three-pane layout

**UI Components:**

```
┌─────────────────────────────────────────────────────────────────┐
│  agentgit viewer: project-name                    [prompt #3/7] │
├──────────┬────────────────────────────┬─────────────────────────┤
│ Files    │ Code View                  │ Transcript              │
│          │                            │                         │
│ 📁 src/  │  1 │ function foo() {     │ ┌─────────────────────┐ │
│   📄 a.ts│  2 │   return bar();  ◀───┼─│ User Prompt #2      │ │
│   📄 b.ts│  3 │ }                     │ │ "add a foo function"│ │
│ 📄 main  │  4 │                       │ └─────────────────────┘ │
│          │  5 │ // Added by prompt 3  │                         │
│          │  6 │ const x = 1;      ◀───┼─── User Prompt #3       │
│          │                            │                         │
│          │ [Diff View] [Blame View]   │ [Show All] [This File]  │
└──────────┴────────────────────────────┴─────────────────────────┘
```

**Key features:**
1. **File tree** - Shows all files with status indicators (added/modified/deleted)
2. **Code pane** - Syntax highlighted with blame annotations in gutter
3. **Transcript pane** - Scrollable prompt/response history
4. **Bidirectional linking** - Click blame → scroll to prompt, click prompt → highlight lines
5. **View modes** - Toggle between final state, blame view, and diff view
6. **Timeline scrubber** - Navigate through prompts chronologically

### Phase 4: CLI Integration

**New command: `agentgit view`**

```bash
# Generate and open viewer
agentgit view ./output-repo

# Generate to specific location
agentgit view ./output-repo -o ./viewer-output

# Serve locally with hot reload
agentgit view ./output-repo --serve --port 8080

# Generate for a transcript directly (builds temp repo first)
agentgit view session.jsonl
```

**Options:**
- `--output, -o` - Output directory for static files
- `--serve` - Start local HTTP server
- `--port` - Server port (default: 8080)
- `--open` - Open browser automatically
- `--embed-data` - Inline all data in HTML (single file output)

---

## Detailed Component Specifications

### Blame Computation

Leverage git's built-in blame, then map commits back to prompts:

```python
def compute_blame_ranges(repo: Repo, file_path: str) -> list[BlameRange]:
    # Run git blame with porcelain format
    blame_output = repo.git.blame(file_path, porcelain=True)

    # Parse blame output into line -> commit mappings
    # Group consecutive lines with same commit into ranges
    # Look up commit metadata to get prompt_id, tool_id
    # Assign color_index based on prompt_num
```

### Color Assignment

Follow PR #23's approach - color by prompt number, not individual operations:

```python
PROMPT_COLORS = [
    "#e6f3ff",  # Light blue
    "#fff3e6",  # Light orange
    "#e6ffe6",  # Light green
    "#ffe6e6",  # Light red
    "#f3e6ff",  # Light purple
    # ... 10-12 distinct colors
]

def get_color_for_prompt(prompt_num: int) -> str:
    return PROMPT_COLORS[prompt_num % len(PROMPT_COLORS)]
```

### Transcript Windowing

For large sessions, implement virtualized rendering:

```javascript
class TranscriptRenderer {
    constructor(container, prompts) {
        this.prompts = prompts;
        this.visibleRange = { start: 0, end: 20 };
        this.setupIntersectionObserver();
    }

    scrollToPrompt(promptNum) {
        // "Teleport" - clear and re-render around target
        this.visibleRange = {
            start: Math.max(0, promptNum - 5),
            end: promptNum + 15
        };
        this.render();
    }
}
```

### Diff View Integration

Use @git-diff-view for showing changes:

```javascript
import { DiffView } from '@git-diff-view/core';

function showDiffForOperation(operation) {
    const diffView = new DiffView({
        oldValue: operation.old_content,
        newValue: operation.new_content,
        fileName: operation.file_path,
        highlighter: shikiHighlighter,
    });

    diffViewContainer.innerHTML = '';
    diffViewContainer.appendChild(diffView.element);
}
```

---

## File Structure

```
src/agentgit/
├── viewer/
│   ├── __init__.py
│   ├── data.py           # ViewerData extraction from git repo
│   ├── blame.py          # Blame computation utilities
│   ├── generator.py      # HTML/static file generation
│   └── server.py         # Optional local dev server
├── templates/
│   ├── viewer/
│   │   ├── index.html    # Main template
│   │   ├── app.js        # Frontend application
│   │   └── styles.css    # Styling
│   └── partials/
│       ├── file_tree.html
│       ├── code_pane.html
│       └── transcript.html
```

---

## Dependencies to Add

```toml
[project.optional-dependencies]
viewer = [
    "jinja2>=3.0.0",       # HTML templating
    "pygments>=2.0.0",     # Syntax highlighting (server-side option)
]

[project.optional-dependencies]
serve = [
    "uvicorn>=0.20.0",     # ASGI server
    "starlette>=0.25.0",   # Lightweight web framework
]
```

---

## Implementation Order

1. **`viewer/data.py`** - ViewerData extraction (can test independently)
2. **`viewer/blame.py`** - Blame range computation
3. **Templates** - HTML/CSS/JS files
4. **`viewer/generator.py`** - Static site generation
5. **CLI command** - `agentgit view` integration
6. **`viewer/server.py`** - Optional live server mode
7. **Polish** - Responsive design, keyboard navigation, URL fragments

---

## Testing Strategy

- Unit tests for blame computation against known repos
- Snapshot tests for generated HTML structure
- E2E tests with Playwright (like PR #23)
- Visual regression tests for UI components

---

## Open Questions

1. **Bundling strategy** - Should we use esbuild/vite to bundle JS, or keep it simple with vanilla JS + CDN imports?

2. **Offline support** - Bundle all dependencies inline, or require CDN access?

3. **Large repo handling** - At what size do we chunk data? (PR #23 uses 500KB threshold)

4. **Diff library choice** - @git-diff-view/core vs react-diff-view vs custom?

5. **Framework choice** - Vanilla JS, Preact, or full React for frontend?
