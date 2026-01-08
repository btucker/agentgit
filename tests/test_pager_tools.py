"""Tests for tool call formatting in the pager."""

import pytest


class TestToolCallFormatting:
    """Tests for enhanced tool call formatting."""

    def test_format_todo_list(self):
        """Test TodoWrite formatting."""
        from agentgit.pager import format_todo_list

        todos = [
            {"content": "Task 1", "status": "completed"},
            {"content": "Task 2", "status": "in_progress"},
            {"content": "Task 3", "status": "pending"},
        ]

        result = format_todo_list(todos, "tool_123")

        # Check header
        assert "### ☰ Task List" in result
        # Check icons are present
        assert "✓ Task 1" in result  # completed
        assert "→ Task 2" in result  # in_progress
        assert "○ Task 3" in result  # pending

    def test_format_bash_tool(self):
        """Test Bash tool formatting."""
        from agentgit.pager import format_bash_tool

        result = format_bash_tool(
            "npm run build && npm test",
            "Build and test the project",
            "tool_456"
        )

        # Check header
        assert "### $ Bash" in result
        # Check description
        assert "*Build and test the project*" in result
        # Check command in code block
        assert "```bash" in result
        assert "npm run build && npm test" in result

    def test_enhance_tool_calls_todowrite(self):
        """Test that TodoWrite JSON is replaced with formatted list."""
        from agentgit.pager import enhance_tool_calls

        text = """## Tool Calls

- **TodoWrite** (`toolu_123`)
  ```json
  {
  "todos": [
    {"content": "Fix bug", "status": "completed"},
    {"content": "Add test", "status": "pending"}
  ]
}
  ```
"""

        result = enhance_tool_calls(text)

        # Should replace JSON with formatted list
        assert "### ☰ Task List" in result
        assert "✓ Fix bug" in result
        assert "○ Add test" in result
        # Should not contain JSON
        assert "```json" not in result

    def test_enhance_tool_calls_bash(self):
        """Test that Bash JSON is replaced with formatted output."""
        from agentgit.pager import enhance_tool_calls

        text = """## Tool Calls

- **Bash** (`toolu_456`)
  ```json
  {
  "command": "ls -la",
  "description": "List files"
}
  ```
"""

        result = enhance_tool_calls(text)

        # Should replace JSON with formatted bash
        assert "### $ Bash" in result
        assert "*List files*" in result
        assert "```bash" in result
        assert "ls -la" in result

    def test_enhance_preserves_other_tools(self):
        """Test that non-special tools are left as JSON."""
        from agentgit.pager import enhance_tool_calls

        text = """## Tool Calls

- **Read** (`toolu_789`)
  ```json
  {
  "file_path": "/tmp/test.txt"
}
  ```
"""

        result = enhance_tool_calls(text)

        # Should preserve JSON for unknown tools
        assert "```json" in result
        assert "file_path" in result

    def test_enhance_handles_malformed_json(self):
        """Test that malformed JSON doesn't break the pager."""
        from agentgit.pager import enhance_tool_calls

        text = """## Tool Calls

- **TodoWrite** (`toolu_123`)
  ```json
  {
  "todos": [INVALID JSON
}
  ```
"""

        # Should not raise exception
        result = enhance_tool_calls(text)

        # Should preserve original when JSON parsing fails
        assert "```json" in result
        assert "INVALID JSON" in result
