<script>
  import { viewerData, connectionStatus, activePromptNum } from './lib/stores.js';
  import FileTree from './lib/FileTree.svelte';
  import CodePane from './lib/CodePane.svelte';
  import Transcript from './lib/Transcript.svelte';

  let projectName = $derived($viewerData?.project_name || 'agentgit viewer');
  let promptCount = $derived($viewerData?.prompts?.length || 0);
  let activeNum = $derived($activePromptNum);
</script>

<div class="app">
  <header class="header">
    <h1 class="title">{projectName}</h1>
    <div class="header-right">
      {#if activeNum}
        <span class="prompt-indicator">Prompt #{activeNum}/{promptCount}</span>
      {:else if promptCount > 0}
        <span class="prompt-indicator">{promptCount} prompts</span>
      {/if}
      <span class="connection-status {$connectionStatus}">
        {$connectionStatus === 'connected' ? 'Live' : $connectionStatus}
      </span>
    </div>
  </header>

  <main class="container">
    {#if $viewerData}
      <div class="panel files-panel">
        <div class="panel-header">Files</div>
        <div class="panel-content">
          <FileTree />
        </div>
      </div>

      <div class="panel code-panel">
        <div class="panel-header">
          Code
        </div>
        <div class="panel-content">
          <CodePane />
        </div>
      </div>

      <div class="panel transcript-panel">
        <div class="panel-header">Transcript</div>
        <div class="panel-content">
          <Transcript />
        </div>
      </div>
    {:else}
      <div class="loading">Loading viewer data...</div>
    {/if}
  </main>
</div>

<style>
  .app {
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .header {
    background: var(--bg-secondary);
    padding: 8px 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }

  .title {
    font-size: 14px;
    font-weight: 500;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .prompt-indicator {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .container {
    flex: 1;
    display: grid;
    grid-template-columns: 220px 1fr 380px;
    overflow: hidden;
  }

  .files-panel {
    min-width: 180px;
  }

  .code-panel {
    min-width: 300px;
  }

  .transcript-panel {
    min-width: 280px;
  }

  @media (max-width: 1200px) {
    .container {
      grid-template-columns: 180px 1fr 300px;
    }
  }

  @media (max-width: 900px) {
    .container {
      grid-template-columns: 1fr;
      grid-template-rows: 200px 1fr 250px;
    }

    .panel {
      border-right: none;
      border-bottom: 1px solid var(--border-color);
    }
  }
</style>
