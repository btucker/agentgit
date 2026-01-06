"""Test parsing of cached web session files."""
import json
from pathlib import Path

import pytest

from agentgit import parse_transcript
from agentgit.formats.claude_code_web import WEB_SESSION_CACHE_DIR


def test_parse_cached_web_session(tmp_path, monkeypatch):
    """Test that cached web session JSONL files can be parsed."""
    # Mock the cache directory to use tmp_path
    cache_dir = tmp_path / "web-sessions"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr("agentgit.formats.claude_code_web.WEB_SESSION_CACHE_DIR", cache_dir)

    # Create a minimal web session JSONL file (same format as regular claude_code)
    session_file = cache_dir / "session_test123.jsonl"

    entries = [
        {
            "type": "user",
            "timestamp": "2025-01-05T19:00:00.000Z",
            "sessionId": "session_test123",
            "cwd": "/test/project",
            "message": {
                "content": "Create a test file",
                "role": "user"
            }
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-05T19:00:05.000Z",
            "sessionId": "session_test123",
            "cwd": "/test/project",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "I'll create a test file for you."
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_test123",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/project/test.py",
                            "content": "# Test file\nprint('hello')\n"
                        }
                    }
                ],
                "role": "assistant"
            }
        }
    ]

    with open(session_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # This should successfully parse the file
    transcript = parse_transcript(session_file)

    assert transcript is not None
    assert transcript.source_format == "claude_code_web_jsonl"
    assert len(transcript.operations) > 0
    # File paths are normalized relative to project root
    assert "test.py" in transcript.operations[0].file_path
