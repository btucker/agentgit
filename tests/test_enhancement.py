"""Tests for enhancement functionality."""

import pytest

from agentgit.enhance import EnhanceConfig, get_available_enhancers
from agentgit.core import AssistantTurn, FileOperation, OperationType, Prompt
from agentgit.enhancers.claude_code import ClaudeCodeEnhancerPlugin
from agentgit.enhancers.rules import (
    RulesEnhancerPlugin,
    _prompt_needs_context,
    _extract_action_from_prompt,
    _summarize_files,
)


class TestEnhanceConfig:
    """Tests for EnhanceConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = EnhanceConfig()
        assert config.enhancer == "rules"  # Default is now rules
        assert config.model == "haiku"
        assert config.enabled is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = EnhanceConfig(enhancer="claude_code", model="sonnet", enabled=False)
        assert config.enhancer == "claude_code"
        assert config.model == "sonnet"
        assert config.enabled is False


class TestGetAvailableEnhancers:
    """Tests for get_available_enhancers."""

    def test_returns_list_of_enhancers(self):
        """Should return list of available enhancer plugins."""
        enhancers = get_available_enhancers()
        assert isinstance(enhancers, list)
        names = [e["name"] for e in enhancers]
        assert "rules" in names
        assert "claude_code" in names

    def test_enhancer_has_name_and_description(self):
        """Each enhancer should have name and description."""
        enhancers = get_available_enhancers()
        for enhancer in enhancers:
            assert "name" in enhancer
            assert "description" in enhancer


class TestRulesEnhancerPlugin:
    """Tests for RulesEnhancerPlugin."""

    def test_get_enhancer_info(self):
        """Should return plugin info."""
        plugin = RulesEnhancerPlugin()
        info = plugin.agentgit_get_ai_enhancer_info()
        assert info["name"] == "rules"
        assert "description" in info

    def test_enhance_operation_message(self):
        """Should generate message for operation."""
        plugin = RulesEnhancerPlugin()
        operation = FileOperation(
            file_path="/project/src/test.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="print('hello')",
        )
        result = plugin.agentgit_enhance_operation_message(
            operation=operation,
            enhancer="rules",
            model=None,
        )
        assert result is not None
        assert "test.py" in result
        assert result.startswith("Add")

    def test_enhance_operation_message_edit(self):
        """Should generate Update message for edit operation."""
        plugin = RulesEnhancerPlugin()
        operation = FileOperation(
            file_path="/project/src/config.json",
            operation_type=OperationType.EDIT,
            timestamp="2025-01-01T00:00:00Z",
            old_string="old",
            new_string="new",
        )
        result = plugin.agentgit_enhance_operation_message(
            operation=operation,
            enhancer="rules",
            model=None,
        )
        assert result is not None
        assert "config.json" in result
        assert result.startswith("Update")

    def test_enhance_operation_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = RulesEnhancerPlugin()
        operation = FileOperation(
            file_path="/test/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="print('hello')",
        )
        result = plugin.agentgit_enhance_operation_message(
            operation=operation,
            enhancer="claude_code",
            model=None,
        )
        assert result is None

    def test_enhance_turn_message(self):
        """Should generate message for turn with multiple files."""
        plugin = RulesEnhancerPlugin()
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
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:01Z",
                    content="more code",
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_turn_message(
            turn=turn,
            prompt=None,
            enhancer="rules",
            model=None,
        )
        assert result is not None
        assert "a.py" in result or "b.py" in result

    def test_enhance_merge_message_good_prompt(self):
        """Should use prompt text for merge if descriptive."""
        plugin = RulesEnhancerPlugin()
        prompt = Prompt(
            text="Add authentication middleware to the Express app",
            timestamp="2025-01-01T00:00:00Z"
        )
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/auth.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[turn],
            enhancer="rules",
            model=None,
        )
        assert result == "Add authentication middleware to the Express app"

    def test_enhance_merge_message_short_prompt(self):
        """Should summarize files for short prompts."""
        plugin = RulesEnhancerPlugin()
        prompt = Prompt(text="yes", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/auth.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                ),
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[turn],
            enhancer="rules",
            model=None,
        )
        assert result is not None
        assert "auth.py" in result


class TestPromptNeedsContext:
    """Tests for _prompt_needs_context helper."""

    def test_short_prompts_need_context(self):
        """Short prompts without action verbs should need context."""
        assert _prompt_needs_context("yes") is True
        assert _prompt_needs_context("ok") is True
        assert _prompt_needs_context("sounds good") is True

    def test_action_prompts_dont_need_context(self):
        """Short prompts with action verbs should not need context."""
        assert _prompt_needs_context("add tests") is False
        assert _prompt_needs_context("fix the bug") is False

    def test_affirmative_responses_need_context(self):
        """Common affirmatives should need context."""
        assert _prompt_needs_context("sounds good, let's do that") is True
        assert _prompt_needs_context("perfect, go ahead") is True
        assert _prompt_needs_context("approved") is True

    def test_numbered_references_need_context(self):
        """Numbered references should need context."""
        assert _prompt_needs_context("let's go with 1, 2, and 3") is True
        assert _prompt_needs_context("options 2 and 4 please") is True

    def test_referential_prompts_need_context(self):
        """Referential prompts should need context."""
        assert _prompt_needs_context("that looks good") is True
        assert _prompt_needs_context("this one please") is True

    def test_detailed_prompts_dont_need_context(self):
        """Detailed, self-contained prompts should not need context."""
        assert _prompt_needs_context(
            "Please add a new function called calculate_total that sums all items"
        ) is False
        assert _prompt_needs_context(
            "Create a REST API endpoint for user authentication with JWT tokens"
        ) is False


class TestExtractActionFromPrompt:
    """Tests for _extract_action_from_prompt helper."""

    def test_extracts_add(self):
        """Should extract Add action."""
        assert _extract_action_from_prompt("add a new feature") == "Add"
        assert _extract_action_from_prompt("please add tests") == "Add"

    def test_extracts_fix(self):
        """Should extract Fix action."""
        assert _extract_action_from_prompt("fix the bug in login") == "Fix"

    def test_extracts_update(self):
        """Should extract Update action."""
        assert _extract_action_from_prompt("update the config file") == "Update"

    def test_extracts_remove(self):
        """Should extract Remove action."""
        assert _extract_action_from_prompt("remove unused imports") == "Remove"

    def test_returns_none_for_no_match(self):
        """Should return None when no action pattern matches."""
        assert _extract_action_from_prompt("yes please") is None
        assert _extract_action_from_prompt("sounds good") is None


class TestSummarizeFiles:
    """Tests for _summarize_files helper."""

    def test_single_file(self):
        """Should return single filename."""
        assert _summarize_files(["src/auth.py"]) == "auth.py"

    def test_two_files(self):
        """Should join with 'and'."""
        result = _summarize_files(["src/a.py", "src/b.py"])
        assert result == "a.py and b.py"

    def test_three_files(self):
        """Should join with comma and 'and'."""
        result = _summarize_files(["src/a.py", "src/b.py", "src/c.py"])
        assert result == "a.py, b.py and c.py"

    def test_many_files(self):
        """Should summarize with count."""
        files = ["a.py", "b.py", "c.py", "d.py", "e.py"]
        result = _summarize_files(files)
        assert "4 other files" in result

    def test_empty_list(self):
        """Should return empty string for empty list."""
        assert _summarize_files([]) == ""


class TestClaudeCodeEnhancerPlugin:
    """Tests for ClaudeCodeEnhancerPlugin."""

    def test_get_enhancer_info(self):
        """Should return plugin info."""
        plugin = ClaudeCodeEnhancerPlugin()
        info = plugin.agentgit_get_ai_enhancer_info()
        assert info["name"] == "claude_code"
        assert "description" in info

    def test_enhance_operation_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = ClaudeCodeEnhancerPlugin()
        operation = FileOperation(
            file_path="/test/file.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="print('hello')",
        )
        result = plugin.agentgit_enhance_operation_message(
            operation=operation,
            enhancer="rules",
            model="haiku",
        )
        assert result is None


class TestContextBuilders:
    """Tests for claude_code context building helpers."""

    def test_build_operation_context_write(self):
        """Should build context for write operation."""
        from agentgit.enhancers.claude_code import _build_operation_context

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
        from agentgit.enhancers.claude_code import _build_operation_context

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

    def test_truncate_text(self):
        """Should truncate long text."""
        from agentgit.enhancers.claude_code import _truncate_text

        short = "short"
        assert _truncate_text(short, 10) == "short"

        long = "a" * 100
        truncated = _truncate_text(long, 10)
        assert len(truncated) == 10
        assert truncated.endswith("...")

    def test_clean_message(self):
        """Should clean up generated message."""
        from agentgit.enhancers.claude_code import _clean_message

        # Remove quotes
        assert _clean_message('"Add feature"') == "Add feature"
        assert _clean_message("'Add feature'") == "Add feature"

        # Truncate long messages
        long_msg = "A" * 100
        cleaned = _clean_message(long_msg)
        assert len(cleaned) == 72
        assert cleaned.endswith("...")
