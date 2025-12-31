"""Starlette server for the agentgit viewer."""

from __future__ import annotations

import asyncio
import json
import logging
import webbrowser
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Track connected WebSocket clients
_websocket_clients: set = set()
_current_version: int = 0


def get_dist_path() -> Path:
    """Get the path to the built Svelte app."""
    return Path(__file__).parent / "dist"


async def notify_clients(version: int) -> None:
    """Notify all connected clients of a data update."""
    global _current_version
    _current_version = version

    message = json.dumps({"type": "update", "version": version})

    # Send to all connected clients
    disconnected = set()
    for ws in _websocket_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)

    # Clean up disconnected clients
    _websocket_clients.difference_update(disconnected)


def create_app(
    repo_path: Path,
    get_viewer_data: Callable,
) -> "Starlette":
    """Create the Starlette application.

    Args:
        repo_path: Path to the agentgit repository.
        get_viewer_data: Function that returns current ViewerData.

    Returns:
        Starlette application instance.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, FileResponse, HTMLResponse
        from starlette.routing import Route, WebSocketRoute
        from starlette.staticfiles import StaticFiles
        from starlette.websockets import WebSocket
    except ImportError:
        raise ImportError(
            "Starlette is required for the GUI. Install with: pip install agentgit[gui]"
        )

    async def api_data(request):
        """Return the current ViewerData as JSON."""
        data = get_viewer_data()
        return JSONResponse(data.to_dict())

    async def api_file(request):
        """Return content of a specific file."""
        file_path = request.path_params["path"]
        data = get_viewer_data()

        if file_path in data.files:
            return JSONResponse({
                "path": file_path,
                "content": data.files[file_path].final_content,
                "blame_ranges": [
                    {
                        "start_line": r.start_line,
                        "end_line": r.end_line,
                        "prompt_id": r.prompt_id,
                        "prompt_num": r.prompt_num,
                        "color": r.color,
                        "assistant_context": r.assistant_context,
                    }
                    for r in data.files[file_path].blame_ranges
                ],
            })
        return JSONResponse({"error": "File not found"}, status_code=404)

    async def websocket_endpoint(websocket: WebSocket):
        """Handle WebSocket connections for live updates."""
        await websocket.accept()
        _websocket_clients.add(websocket)

        # Send current version immediately
        await websocket.send_text(
            json.dumps({"type": "connected", "version": _current_version})
        )

        try:
            while True:
                # Handle incoming messages (ping/pong, etc.)
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

        except Exception:
            pass
        finally:
            _websocket_clients.discard(websocket)

    async def index(request):
        """Serve the main index.html."""
        dist_path = get_dist_path()
        index_path = dist_path / "index.html"

        if index_path.exists():
            return FileResponse(index_path)

        # Fallback: serve a simple HTML page that loads from CDN
        return HTMLResponse(_get_fallback_html())

    def _get_fallback_html() -> str:
        """Return fallback HTML when dist is not built."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>agentgit viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: #252526;
            padding: 8px 16px;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        header h1 { font-size: 14px; font-weight: 500; }
        .container {
            flex: 1;
            display: grid;
            grid-template-columns: 200px 1fr 350px;
            overflow: hidden;
        }
        .panel {
            border-right: 1px solid #3c3c3c;
            overflow: auto;
            padding: 8px;
        }
        .panel:last-child { border-right: none; }
        .panel h2 {
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
        }
        .file-tree { font-size: 13px; }
        .file-item {
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 4px;
        }
        .file-item:hover { background: #2a2d2e; }
        .file-item.selected { background: #094771; }
        .code-view {
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 13px;
            line-height: 1.5;
            white-space: pre;
            padding: 8px;
        }
        .prompt-card {
            background: #2d2d2d;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
        }
        .prompt-header {
            font-size: 11px;
            color: #888;
            margin-bottom: 4px;
        }
        .prompt-text {
            font-size: 13px;
            white-space: pre-wrap;
        }
        .status {
            padding: 20px;
            text-align: center;
            color: #888;
        }
        .loading { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <header>
        <h1>agentgit viewer</h1>
        <span id="status" class="loading">Loading...</span>
    </header>
    <div class="container">
        <div class="panel" id="files">
            <h2>Files</h2>
            <div class="file-tree" id="file-tree"></div>
        </div>
        <div class="panel" id="code">
            <h2>Code</h2>
            <div class="code-view" id="code-view">Select a file to view</div>
        </div>
        <div class="panel" id="transcript">
            <h2>Transcript</h2>
            <div id="prompts"></div>
        </div>
    </div>
    <script>
        let data = null;
        let selectedFile = null;
        let ws = null;

        async function loadData() {
            const resp = await fetch('/api/data');
            data = await resp.json();
            render();
        }

        function render() {
            document.getElementById('status').textContent = data.project_name || 'Ready';
            document.getElementById('status').classList.remove('loading');

            // Render file tree
            const fileTree = document.getElementById('file-tree');
            const paths = Object.keys(data.files).sort();
            fileTree.innerHTML = paths.map(p =>
                `<div class="file-item ${p === selectedFile ? 'selected' : ''}" data-path="${p}">${p}</div>`
            ).join('');

            fileTree.querySelectorAll('.file-item').forEach(el => {
                el.onclick = () => selectFile(el.dataset.path);
            });

            // Render prompts
            const prompts = document.getElementById('prompts');
            prompts.innerHTML = data.prompts.map(p => `
                <div class="prompt-card" style="border-left: 3px solid ${p.color}">
                    <div class="prompt-header">Prompt #${p.prompt_num}</div>
                    <div class="prompt-text">${escapeHtml(p.text.slice(0, 200))}${p.text.length > 200 ? '...' : ''}</div>
                </div>
            `).join('');

            // Render code if file selected
            if (selectedFile && data.files[selectedFile]) {
                const file = data.files[selectedFile];
                const lines = file.final_content.split('\\n');
                document.getElementById('code-view').innerHTML = lines.map((line, i) => {
                    const range = file.blame_ranges.find(r => i + 1 >= r.start_line && i + 1 <= r.end_line);
                    const bg = range ? range.color : 'transparent';
                    return `<div style="background: ${bg}">${String(i + 1).padStart(4)} | ${escapeHtml(line)}</div>`;
                }).join('');
            }
        }

        function selectFile(path) {
            selectedFile = path;
            render();
        }

        function escapeHtml(text) {
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function connectWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'update') {
                    loadData();
                }
            };

            ws.onclose = () => {
                setTimeout(connectWebSocket, 2000);
            };
        }

        loadData();
        connectWebSocket();
    </script>
</body>
</html>"""

    # Build routes
    routes = [
        Route("/", index),
        Route("/api/data", api_data),
        Route("/api/file/{path:path}", api_file),
        WebSocketRoute("/ws", websocket_endpoint),
    ]

    # Add static file serving if dist exists
    dist_path = get_dist_path()
    if dist_path.exists():
        app = Starlette(routes=routes)
        app.mount("/assets", StaticFiles(directory=dist_path / "assets"), name="assets")
    else:
        app = Starlette(routes=routes)

    return app


def run_server(
    repo_path: Path,
    get_viewer_data: Callable,
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Run the viewer server.

    Args:
        repo_path: Path to the agentgit repository.
        get_viewer_data: Function that returns current ViewerData.
        port: Port to run on.
        open_browser: Whether to open browser automatically.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for the GUI. Install with: pip install agentgit[gui]"
        )

    app = create_app(repo_path, get_viewer_data)

    if open_browser:
        # Open browser after a short delay
        def open_browser_delayed():
            import time
            time.sleep(0.5)
            webbrowser.open(f"http://localhost:{port}")

        import threading
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    logger.info("Starting viewer at http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


async def run_server_async(
    repo_path: Path,
    get_viewer_data: Callable,
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Run the viewer server asynchronously (for integration with watch mode).

    Args:
        repo_path: Path to the agentgit repository.
        get_viewer_data: Function that returns current ViewerData.
        port: Port to run on.
        open_browser: Whether to open browser automatically.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for the GUI. Install with: pip install agentgit[gui]"
        )

    app = create_app(repo_path, get_viewer_data)

    if open_browser:
        asyncio.get_event_loop().call_later(
            0.5, lambda: webbrowser.open(f"http://localhost:{port}")
        )

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
