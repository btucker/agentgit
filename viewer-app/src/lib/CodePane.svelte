<script>
  import { selectedFile, selectedFileData, activePromptNum } from './stores.js';

  let lines = $derived($selectedFileData?.final_content?.split('\n') || []);
  let blameRanges = $derived($selectedFileData?.blame_ranges || []);

  function getBlameForLine(lineNum) {
    return blameRanges.find(r => lineNum >= r.start_line && lineNum <= r.end_line);
  }

  function handleLineClick(lineNum) {
    const blame = getBlameForLine(lineNum);
    if (blame && blame.prompt_num) {
      activePromptNum.set(blame.prompt_num);
    }
  }

  function handleBlameHover(lineNum, entering) {
    // Could add hover preview here
  }
</script>

<div class="code-pane">
  {#if $selectedFile && $selectedFileData}
    <div class="file-path">{$selectedFile}</div>
    <div class="code-container">
      <div class="blame-gutter">
        {#each lines as _, i}
          {@const blame = getBlameForLine(i + 1)}
          <div
            class="blame-indicator"
            style="background: {blame?.color || 'transparent'}"
            title={blame?.assistant_context || `Prompt #${blame?.prompt_num || '?'}`}
            onclick={() => handleLineClick(i + 1)}
          ></div>
        {/each}
      </div>
      <div class="line-numbers">
        {#each lines as _, i}
          <div class="line-number">{i + 1}</div>
        {/each}
      </div>
      <div class="code-content">
        {#each lines as line, i}
          {@const blame = getBlameForLine(i + 1)}
          <div
            class="code-line"
            class:highlighted={blame?.prompt_num === $activePromptNum}
            onclick={() => handleLineClick(i + 1)}
          >{line || ' '}</div>
        {/each}
      </div>
    </div>
  {:else}
    <div class="empty-state">
      <p>Select a file to view</p>
    </div>
  {/if}
</div>

<style>
  .code-pane {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .file-path {
    padding: 6px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .code-container {
    flex: 1;
    display: flex;
    font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
    font-size: 12px;
    line-height: 20px;
    overflow: auto;
  }

  .blame-gutter {
    width: 8px;
    flex-shrink: 0;
    background: var(--bg-secondary);
  }

  .blame-indicator {
    height: 20px;
    cursor: pointer;
    transition: filter 0.1s;
  }

  .blame-indicator:hover {
    filter: brightness(0.85);
  }

  .line-numbers {
    padding: 0;
    text-align: right;
    color: var(--text-muted);
    user-select: none;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    flex-shrink: 0;
    min-width: 40px;
  }

  .line-number {
    padding: 0 8px;
    height: 20px;
  }

  .code-content {
    flex: 1;
    padding: 0;
  }

  .code-line {
    padding: 0 12px;
    height: 20px;
    white-space: pre;
    cursor: pointer;
  }

  .code-line:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .code-line.highlighted {
    background: rgba(0, 120, 212, 0.15);
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
  }
</style>
