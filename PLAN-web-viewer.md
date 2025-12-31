# Web-Based Git Viewer for agentgit

## Overview

Build a custom web-based viewer that displays agentgit repositories with a three-pane interface inspired by the [claude-code-transcripts PR #23](https://github.com/simonw/claude-code-transcripts/pull/23), but tailored to agentgit's architecture.

## Goals

1. **Visualize file evolution** - Show how files changed across prompts/turns
2. **Display rich metadata** - Prompts, assistant reasoning, tool IDs, timestamps
3. **Git blame-style annotations** - Link code lines to the prompts that created them
4. **Interactive navigation** - Click between code and transcript seamlessly

## Architecture Decision: Integrated Watch + GUI

Instead of a separate `agentgit view` command, integrate the GUI into the existing `--watch` mode:

```
┌─────────────────┐    file change    ┌─────────────────┐
│  Transcript     │ ───────────────▶  │  Git Builder    │
│  (session.jsonl)│                   │  (incremental)  │
└─────────────────┘                   └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  ViewerData     │
                                      │  (regenerate)   │
                                      └────────┬────────┘
                                               │ WebSocket push
                                               ▼
                                      ┌─────────────────┐
                                      │  Browser        │
                                      │  (live update)  │
                                      └─────────────────┘
```

**Key behaviors:**
1. Watchdog detects transcript change → rebuilds git repo (existing)
2. After rebuild → regenerate ViewerData JSON
3. Push update via WebSocket → browser updates relevant panes
4. No full page reload - Svelte reactively updates with new data

**Modes:**
- `--watch` only: CLI output, no server (existing behavior)
- `--watch --gui`: Start server + open browser, live updates
- `--gui` without `--watch`: One-shot build + serve (for viewing completed sessions)

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

**Two ways to access the GUI:**

#### 1. Standalone `agentgit gui` command (view existing repo)

```bash
# View an already-built agentgit repo
agentgit gui ./output-repo

# With custom port
agentgit gui ./output-repo --port 3000

# Don't auto-open browser
agentgit gui ./output-repo --no-open
```

#### 2. `--gui` flag on `process` command (build + view)

```bash
# Existing: watch transcript, rebuild git repo (CLI output only)
agentgit process session.jsonl --watch

# New: watch + live GUI with WebSocket updates
agentgit process session.jsonl --watch --gui

# New: one-shot build + serve GUI (for completed sessions)
agentgit process session.jsonl --gui

# With custom port
agentgit process session.jsonl --watch --gui --port 3000

# Suppress auto-open browser
agentgit process session.jsonl --gui --no-open
```

**CLI options:**
- `--gui` - Start web server and serve the viewer (on `process` command)
- `--port` - Server port (default: 8080)
- `--no-open` - Don't auto-open browser

**Server endpoints:**
```
GET  /                    # Svelte app (static)
GET  /api/data            # ViewerData JSON
WS   /ws                  # WebSocket for live updates
GET  /api/file/:path      # Individual file content (for large repos)
```

**WebSocket protocol:**
```json
{ "type": "update", "version": 1234567890 }  // Server → Client: data changed
{ "type": "ping" }                            // Client → Server: keepalive
```

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
│   ├── server.py         # Starlette server (static files + API + WebSocket)
│   └── dist/             # Built Svelte app (generated, gitignored)
│       ├── index.html
│       ├── assets/
│       │   ├── app-[hash].js
│       │   └── app-[hash].css

viewer-app/                # Svelte frontend source (separate package)
├── package.json
├── vite.config.js         # Build outputs to src/agentgit/viewer/dist/
├── src/
│   ├── App.svelte         # Main three-pane layout
│   ├── main.js            # Entry point
│   ├── lib/
│   │   ├── FileTree.svelte
│   │   ├── CodePane.svelte
│   │   ├── DiffPane.svelte
│   │   ├── Transcript.svelte
│   │   ├── BlameGutter.svelte
│   │   ├── stores.js      # Svelte stores for state
│   │   └── websocket.js   # WebSocket client for live updates
│   └── styles/
│       └── app.css
└── public/
    └── index.html
```

**Build workflow:**
```bash
cd viewer-app && npm run build  # Outputs to ../src/agentgit/viewer/dist/
```

The built frontend is bundled with the Python package, so `pip install agentgit[gui]` includes everything needed.

---

## Dependencies to Add

```toml
[project.optional-dependencies]
gui = [
    "starlette>=0.25.0",   # ASGI web framework
    "uvicorn>=0.20.0",     # ASGI server
    "websockets>=10.0",    # WebSocket support for live updates
]
```

**Frontend dependencies (viewer-app/package.json):**
```json
{
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "svelte": "^5.0.0",
    "vite": "^5.0.0"
  },
  "dependencies": {
    "@git-diff-view/svelte": "^0.0.x",
    "shiki": "^1.0.0"
  }
}
```

---

## Implementation Order

### Backend (Python)
1. **`viewer/data.py`** - ViewerData extraction from git repo
2. **`viewer/blame.py`** - Blame range computation
3. **`viewer/server.py`** - Starlette server with WebSocket support
4. **CLI integration** - Add `--gui`, `--port`, `--no-open` to `process` command
5. **Watcher integration** - Hook ViewerData regeneration into existing watch loop

### Frontend (Svelte)
1. **Scaffold** - `viewer-app/` with Vite + Svelte 5
2. **App.svelte** - Three-pane layout with CSS Grid
3. **stores.js** - Shared state + WebSocket connection for live updates
4. **FileTree.svelte** - File browser with status indicators
5. **CodePane.svelte** - Syntax highlighting + blame gutter
6. **Transcript.svelte** - Virtualized prompt/response list
7. **DiffPane.svelte** - git-diff-view integration
8. **Polish** - Keyboard navigation, URL fragments, responsive design

### Integration
1. Build Svelte app → bundle into `src/agentgit/viewer/dist/`
2. Server serves static files from bundled dist
3. WebSocket notifies browser when watcher rebuilds repo

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
