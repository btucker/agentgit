<script>
  import { prompts, activePromptNum, selectedFile, viewerData } from './stores.js';

  // Windowed rendering for large transcripts
  let visibleStart = $state(0);
  let visibleEnd = $state(20);
  let container;

  let visiblePrompts = $derived($prompts.slice(visibleStart, visibleEnd));

  function scrollToPrompt(promptNum) {
    const index = promptNum - 1;
    visibleStart = Math.max(0, index - 5);
    visibleEnd = Math.min($prompts.length, index + 15);

    // Scroll the active prompt into view after render
    setTimeout(() => {
      const el = container?.querySelector(`[data-prompt="${promptNum}"]`);
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 0);
  }

  function handlePromptClick(prompt) {
    activePromptNum.set(prompt.prompt_num);
  }

  function getFilesForPrompt(prompt) {
    return prompt.files_modified || [];
  }

  function truncateText(text, maxLength = 300) {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
  }

  // Watch for external navigation
  $effect(() => {
    const num = $activePromptNum;
    if (num && (num < visibleStart + 1 || num > visibleEnd)) {
      scrollToPrompt(num);
    }
  });

  // Load more on scroll
  function handleScroll(e) {
    const { scrollTop, scrollHeight, clientHeight } = e.target;

    // Load more at top
    if (scrollTop < 100 && visibleStart > 0) {
      visibleStart = Math.max(0, visibleStart - 10);
    }

    // Load more at bottom
    if (scrollHeight - scrollTop - clientHeight < 100 && visibleEnd < $prompts.length) {
      visibleEnd = Math.min($prompts.length, visibleEnd + 10);
    }
  }
</script>

<div class="transcript" bind:this={container} onscroll={handleScroll}>
  {#if visibleStart > 0}
    <button class="load-more" onclick={() => { visibleStart = Math.max(0, visibleStart - 10); }}>
      Load earlier prompts ({visibleStart} hidden)
    </button>
  {/if}

  {#each visiblePrompts as prompt (prompt.prompt_id)}
    <div
      class="prompt-card"
      class:active={$activePromptNum === prompt.prompt_num}
      style="border-left-color: {prompt.color}"
      data-prompt={prompt.prompt_num}
      onclick={() => handlePromptClick(prompt)}
    >
      <div class="prompt-header">
        <span class="prompt-number">Prompt #{prompt.prompt_num}</span>
        {#if prompt.timestamp}
          <span class="prompt-timestamp">{prompt.timestamp}</span>
        {/if}
      </div>

      <div class="prompt-text">{truncateText(prompt.text)}</div>

      {#if getFilesForPrompt(prompt).length > 0}
        <div class="prompt-files">
          {#each getFilesForPrompt(prompt) as file}
            <span
              class="file-badge"
              class:current={file === $selectedFile}
            >{file.split('/').pop()}</span>
          {/each}
        </div>
      {/if}
    </div>
  {/each}

  {#if visibleEnd < $prompts.length}
    <button class="load-more" onclick={() => { visibleEnd = Math.min($prompts.length, visibleEnd + 10); }}>
      Load more prompts ({$prompts.length - visibleEnd} remaining)
    </button>
  {/if}

  {#if $prompts.length === 0}
    <div class="empty-state">No prompts found</div>
  {/if}
</div>

<style>
  .transcript {
    height: 100%;
    overflow-y: auto;
    padding: 4px 0;
  }

  .prompt-card {
    background: var(--bg-tertiary);
    border-radius: 6px;
    padding: 12px;
    margin: 8px;
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: box-shadow 0.15s;
  }

  .prompt-card:hover {
    background: #333;
  }

  .prompt-card.active {
    box-shadow: 0 0 0 1px var(--accent-color);
  }

  .prompt-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .prompt-number {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .prompt-timestamp {
    font-size: 10px;
    color: var(--text-muted);
  }

  .prompt-text {
    font-size: 13px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-primary);
  }

  .prompt-files {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .file-badge {
    font-size: 10px;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: var(--text-secondary);
  }

  .file-badge.current {
    background: var(--accent-color);
    color: white;
  }

  .load-more {
    width: calc(100% - 16px);
    margin: 8px;
    padding: 8px;
    background: var(--bg-tertiary);
    border: 1px dashed var(--border-color);
    color: var(--text-secondary);
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }

  .load-more:hover {
    background: #333;
    color: var(--text-primary);
  }

  .empty-state {
    padding: 32px;
    text-align: center;
    color: var(--text-muted);
  }
</style>
