<script>
  import { filePaths, selectedFile, viewerData } from './stores.js';

  function selectFile(path) {
    selectedFile.set(path);
  }

  function getFileStatus(path) {
    const file = $viewerData?.files?.[path];
    return file?.status || 'modified';
  }

  function getFileName(path) {
    return path.split('/').pop();
  }

  function getIndent(path) {
    return (path.split('/').length - 1) * 12;
  }
</script>

<div class="file-tree">
  {#each $filePaths as path (path)}
    <button
      class="file-item"
      class:selected={$selectedFile === path}
      style="padding-left: {16 + getIndent(path)}px"
      onclick={() => selectFile(path)}
    >
      <span class="status-dot {getFileStatus(path)}"></span>
      <span class="file-name" title={path}>{getFileName(path)}</span>
    </button>
  {/each}

  {#if $filePaths.length === 0}
    <div class="empty-state">No files</div>
  {/if}
</div>

<style>
  .file-tree {
    padding: 4px 0;
  }

  .file-item {
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    padding: 4px 8px 4px 16px;
    cursor: pointer;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .file-item:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .file-item.selected {
    background: var(--accent-color);
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .status-dot.added {
    background: #4caf50;
  }

  .status-dot.modified {
    background: #ff9800;
  }

  .file-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .empty-state {
    padding: 16px;
    color: var(--text-muted);
    text-align: center;
  }
</style>
