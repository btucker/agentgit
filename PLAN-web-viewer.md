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
- **Svelte 5** - Latest version with runes for reactivity
- [@git-diff-view/svelte](https://github.com/MrWangJustToDo/git-diff-view) for diff rendering
- [Shiki](https://shiki.style/) for syntax highlighting (via CDN)
- CSS Grid for three-pane layout
- CDN imports (esm.sh, unpkg, or jsdelivr)

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

For large sessions, implement virtualized rendering with Svelte:

```svelte
<script>
  import { onMount } from 'svelte';

  let { prompts, activePromptNum } = $props();
  let visibleRange = $state({ start: 0, end: 20 });
  let container;

  function scrollToPrompt(promptNum) {
    // "Teleport" - update visible range around target
    visibleRange = {
      start: Math.max(0, promptNum - 5),
      end: Math.min(prompts.length, promptNum + 15)
    };
  }

  // Watch for external navigation requests
  $effect(() => {
    if (activePromptNum !== undefined) {
      scrollToPrompt(activePromptNum);
    }
  });
</script>

<div class="transcript" bind:this={container}>
  {#each prompts.slice(visibleRange.start, visibleRange.end) as prompt (prompt.prompt_id)}
    <PromptCard {prompt} />
  {/each}
</div>
```

### Diff View Integration

Use @git-diff-view/svelte for showing changes:

```svelte
<script>
  import { DiffView } from '@git-diff-view/svelte';
  import '@git-diff-view/svelte/styles/diff-view.css';

  let { operation } = $props();
</script>

<DiffView
  data={{
    oldFile: { content: operation.old_content, fileName: operation.file_path },
    newFile: { content: operation.new_content, fileName: operation.file_path },
  }}
  diffViewMode="split"
  diffViewHighlight={true}
/>
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

viewer-app/                # Svelte frontend (separate package)
├── package.json
├── vite.config.js         # Build config (outputs to src/agentgit/viewer/dist/)
├── src/
│   ├── App.svelte         # Main three-pane layout
│   ├── main.js            # Entry point
│   ├── lib/
│   │   ├── FileTree.svelte
│   │   ├── CodePane.svelte
│   │   ├── DiffPane.svelte
│   │   ├── Transcript.svelte
│   │   ├── BlameGutter.svelte
│   │   └── stores.js      # Svelte stores for state
│   └── styles/
│       └── app.css
└── public/
    └── index.html         # Template with data injection point
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

### Backend (Python)
1. **`viewer/data.py`** - ViewerData extraction (can test independently)
2. **`viewer/blame.py`** - Blame range computation
3. **`viewer/generator.py`** - Static site generation (injects data into built Svelte app)
4. **CLI command** - `agentgit view` integration
5. **`viewer/server.py`** - Optional live server mode with hot reload

### Frontend (Svelte)
1. **Scaffold** - `viewer-app/` with Vite + Svelte 5
2. **App.svelte** - Three-pane layout with CSS Grid
3. **FileTree.svelte** - File browser with status indicators
4. **CodePane.svelte** - Syntax highlighting + blame gutter
5. **Transcript.svelte** - Virtualized prompt/response list
6. **DiffPane.svelte** - git-diff-view integration
7. **Stores** - Shared state (selected file, active prompt, view mode)
8. **Polish** - Keyboard navigation, URL fragments, responsive design

---

## Testing Strategy

- Unit tests for blame computation against known repos
- Snapshot tests for generated HTML structure
- E2E tests with Playwright (like PR #23)
- Visual regression tests for UI components

---

## Technology Decisions

| Question | Decision |
|----------|----------|
| Bundling | Keep it simple - no bundler, use CDN imports |
| Diff library | [@git-diff-view](https://github.com/MrWangJustToDo/git-diff-view) |
| Framework | Svelte 5 (latest) |
| Offline support | CDN dependencies allowed |
| Large repo threshold | 500KB (following PR #23's approach)
