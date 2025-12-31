"""Tests for AI enhancement functionality."""

import pytest

from agentgit.ai_commit import AICommitConfig, get_available_enhancers
from agentgit.core import AssistantTurn, FileOperation, OperationType, Prompt
from agentgit.enhancers.claude_cli import ClaudeCLIEnhancerPlugin


class TestAICommitConfig:
    """Tests for AICommitConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = AICommitConfig()
        assert config.enhancer == "claude_cli"
        assert config.model == "haiku"
        assert config.enabled is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = AICommitConfig(enhancer="codex", model="gpt-4", enabled=False)
        assert config.enhancer == "codex"
        assert config.model == "gpt-4"
        assert config.enabled is False


class TestGetAvailableEnhancers:
    """Tests for get_available_enhancers."""

    def test_returns_list_of_enhancers(self):
        """Should return list of available enhancer plugins."""
        enhancers = get_available_enhancers()
        assert isinstance(enhancers, list)
        # Should have at least the claude_cli enhancer
        names = [e["name"] for e in enhancers]
        assert "claude_cli" in names

    def test_enhancer_has_name_and_description(self):
        """Each enhancer should have name and description."""
        enhancers = get_available_enhancers()
        for enhancer in enhancers:
            assert "name" in enhancer
            assert "description" in enhancer


class TestClaudeCLIEnhancerPlugin:
    """Tests for ClaudeCLIEnhancerPlugin."""

    def test_get_ai_enhancer_info(self):
        """Should return plugin info."""
        plugin = ClaudeCLIEnhancerPlugin()
        info = plugin.agentgit_get_ai_enhancer_info()
        assert info["name"] == "claude_cli"
        assert "description" in info

    def test_enhance_operation_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = ClaudeCLIEnhancerPlugin()
        operation = FileOperation(
            file_path="/test/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="print('hello')",
        )
        result = plugin.agentgit_enhance_operation_message(
            operation=operation,
            enhancer="other_enhancer",
            model="haiku",
        )
        assert result is None

    def test_enhance_turn_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = ClaudeCLIEnhancerPlugin()
        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_turn_message(
            turn=turn,
            prompt=None,
            enhancer="other_enhancer",
            model="haiku",
        )
        assert result is None

    def test_enhance_merge_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = ClaudeCLIEnhancerPlugin()
        prompt = Prompt(text="Test prompt", timestamp="2025-01-01T00:00:00Z")
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[],
            enhancer="other_enhancer",
            model="haiku",
        )
        assert result is None


class TestContextBuilders:
    """Tests for context building helpers."""

    def test_build_operation_context_write(self):
        """Should build context for write operation."""
        from agentgit.enhancers.claude_cli import _build_operation_context

        operation = FileOperation(
            file_path="/project/src/test.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="def hello(): pass",
        )
        context = _build_operation_context(operation)
        assert "test.py" in context
        assert "Created" in context
        assert "def hello()" in context

    def test_build_operation_context_edit(self):
        """Should build context for edit operation."""
        from agentgit.enhancers.claude_cli import _build_operation_context

        operation = FileOperation(
            file_path="/project/src/test.py",
            operation_type=OperationType.EDIT,
            timestamp="2025-01-01T00:00:00Z",
            old_string="old code",
            new_string="new code",
        )
        context = _build_operation_context(operation)
        assert "test.py" in context
        assert "Modified" in context
        assert "old code" in context
        assert "new code" in context

    def test_build_turn_context(self):
        """Should build context for assistant turn."""
        from agentgit.enhancers.claude_cli import _build_turn_context

        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/a.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                ),
                FileOperation(
                    file_path="/project/src/b.py",
                    operation_type=OperationType.EDIT,
                    timestamp="2025-01-01T00:00:01Z",
                    old_string="old",
                    new_string="new",
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        context = _build_turn_context(turn)
        assert "a.py" in context
        assert "b.py" in context

    def test_truncate_text(self):
        """Should truncate long text."""
        from agentgit.enhancers.claude_cli import _truncate_text

        short = "short"
        assert _truncate_text(short, 10) == "short"

        long = "a" * 100
        truncated = _truncate_text(long, 10)
        assert len(truncated) == 10
        assert truncated.endswith("...")

    def test_clean_message(self):
        """Should clean up AI-generated message."""
        from agentgit.enhancers.claude_cli import _clean_message

        # Remove quotes
        assert _clean_message('"Add feature"') == "Add feature"
        assert _clean_message("'Add feature'") == "Add feature"

        # Truncate long messages
        long_msg = "A" * 100
        cleaned = _clean_message(long_msg)
        assert len(cleaned) == 72
        assert cleaned.endswith("...")
