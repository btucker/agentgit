/**
 * Svelte stores for shared state and WebSocket connection.
 */

import { writable, derived } from 'svelte/store';

// Viewer data from the API
export const viewerData = writable(null);

// Currently selected file path
export const selectedFile = writable(null);

// Currently active prompt (for highlighting)
export const activePromptNum = writable(null);

// View mode: 'code' | 'diff'
export const viewMode = writable('code');

// WebSocket connection status: 'connecting' | 'connected' | 'disconnected'
export const connectionStatus = writable('connecting');

// Data version for cache busting
export const dataVersion = writable(0);

// Derived stores

// Get the currently selected file data
export const selectedFileData = derived(
  [viewerData, selectedFile],
  ([$viewerData, $selectedFile]) => {
    if (!$viewerData || !$selectedFile) return null;
    return $viewerData.files[$selectedFile] || null;
  }
);

// Get sorted list of file paths
export const filePaths = derived(
  viewerData,
  ($viewerData) => {
    if (!$viewerData) return [];
    return Object.keys($viewerData.files).sort();
  }
);

// Get prompts array
export const prompts = derived(
  viewerData,
  ($viewerData) => $viewerData?.prompts || []
);

// WebSocket management
let ws = null;
let reconnectTimeout = null;

export function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  connectionStatus.set('connecting');

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    connectionStatus.set('connected');
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === 'update' || msg.type === 'connected') {
        // Data has been updated, fetch new data
        dataVersion.set(msg.version);
        fetchViewerData();
      }
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e);
    }
  };

  ws.onclose = () => {
    connectionStatus.set('disconnected');
    console.log('WebSocket disconnected, reconnecting in 2s...');

    // Attempt to reconnect after delay
    clearTimeout(reconnectTimeout);
    reconnectTimeout = setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    ws.close();
  };
}

export async function fetchViewerData() {
  try {
    const response = await fetch('/api/data');
    if (!response.ok) throw new Error('Failed to fetch data');

    const data = await response.json();
    viewerData.set(data);

    // Auto-select first file if none selected
    const currentSelected = selectedFile;
    let hasSelection = false;
    currentSelected.subscribe(v => hasSelection = !!v)();

    if (!hasSelection && data.files && Object.keys(data.files).length > 0) {
      const firstFile = Object.keys(data.files).sort()[0];
      selectedFile.set(firstFile);
    }
  } catch (error) {
    console.error('Failed to fetch viewer data:', error);
  }
}

// Initialize on module load
if (typeof window !== 'undefined') {
  fetchViewerData();
  connectWebSocket();
}
