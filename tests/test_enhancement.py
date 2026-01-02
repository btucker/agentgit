"""Tests for enhancement functionality."""

import pytest

from agentgit.enhance import (
    EnhanceConfig,
    get_available_enhancers,
    generate_operation_commit_message,
    generate_turn_commit_message,
    generate_merge_commit_message,
    preprocess_batch_enhancement,
)
from agentgit.core import AssistantContext, AssistantTurn, FileOperation, OperationType, Prompt, PromptResponse
from agentgit.enhancers.llm import LLMEnhancerPlugin
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
        config = EnhanceConfig(enhancer="llm", model="sonnet", enabled=False)
        assert config.enhancer == "llm"
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
        assert "llm" in names

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
            enhancer="llm",
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


class TestPromptNeedsContextExtended:
    """Extended tests for _prompt_needs_context edge cases."""

    def test_contextual_starters(self):
        """Test various contextual starters."""
        assert _prompt_needs_context("ok let's do it") is True
        assert _prompt_needs_context("sure thing") is True
        assert _prompt_needs_context("go ahead and do it") is True
        assert _prompt_needs_context("sounds good to me") is True
        assert _prompt_needs_context("perfect, that's what I want") is True
        assert _prompt_needs_context("proceed with the plan") is True
        assert _prompt_needs_context("continue from here") is True
        assert _prompt_needs_context("yep") is True
        assert _prompt_needs_context("nope") is True
        assert _prompt_needs_context("skip this one") is True
        assert _prompt_needs_context("both of them") is True
        assert _prompt_needs_context("the first one") is True
        assert _prompt_needs_context("option 2") is True

    def test_referential_starts(self):
        """Test referential start patterns."""
        assert _prompt_needs_context("that should work") is True
        assert _prompt_needs_context("this is good") is True
        assert _prompt_needs_context("it looks correct") is True
        assert _prompt_needs_context("those are the files") is True
        assert _prompt_needs_context("these changes are fine") is True

    def test_medium_length_non_referential(self):
        """Test medium length prompts that don't need context."""
        # Over 50 chars and doesn't start with referential words
        result = _prompt_needs_context(
            "Create a new configuration file with default settings for the app"
        )
        assert result is False


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


class TestLLMModelFunctions:
    """Tests for _get_model and _run_llm functions."""

    def test_get_model_caches_instances(self, monkeypatch):
        """Should cache model instances."""
        from agentgit.enhancers import llm as llm_enhancer

        # Clear cache before test
        llm_enhancer._model_cache.clear()

        mock_model = object()
        call_count = [0]

        def mock_get_model(model_id):
            call_count[0] += 1
            return mock_model

        # Mock the llm module
        class MockLLM:
            @staticmethod
            def get_model(model_id):
                return mock_get_model(model_id)

        monkeypatch.setattr(llm_enhancer, "llm", MockLLM, raising=False)
        import sys
        sys.modules["llm"] = MockLLM

        result1 = llm_enhancer._get_model("test-model")
        result2 = llm_enhancer._get_model("test-model")

        assert result1 is result2  # Same cached instance
        assert call_count[0] == 1  # Only called once due to caching

        # Clean up
        del sys.modules["llm"]
        llm_enhancer._model_cache.clear()

    def test_get_model_import_error(self, monkeypatch):
        """Should handle ImportError gracefully."""
        from agentgit.enhancers import llm as llm_enhancer

        llm_enhancer._model_cache.clear()

        def mock_import(*args, **kwargs):
            raise ImportError("llm not installed")

        monkeypatch.setattr("builtins.__import__", mock_import)

        result = llm_enhancer._get_model("test-model")
        assert result is None

        llm_enhancer._model_cache.clear()

    def test_run_llm_returns_none_when_model_unavailable(self, monkeypatch):
        """Should return None when model is not available."""
        from agentgit.enhancers import llm as llm_enhancer

        monkeypatch.setattr(llm_enhancer, "_get_model", lambda x: None)

        result = llm_enhancer._run_llm("test prompt")
        assert result is None

    def test_run_llm_handles_exceptions(self, monkeypatch):
        """Should handle exceptions from model.prompt()."""
        from agentgit.enhancers import llm as llm_enhancer

        class MockModel:
            def prompt(self, text):
                raise Exception("API error")

        monkeypatch.setattr(llm_enhancer, "_get_model", lambda x: MockModel())

        result = llm_enhancer._run_llm("test prompt")
        assert result is None


class TestLLMEnhancerPlugin:
    """Tests for LLMEnhancerPlugin."""

    def test_get_enhancer_info(self):
        """Should return plugin info."""
        plugin = LLMEnhancerPlugin()
        info = plugin.agentgit_get_ai_enhancer_info()
        assert info["name"] == "llm"
        assert "description" in info

    def test_enhance_turn_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = LLMEnhancerPlugin()
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/test/file.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_turn_message(
            turn=turn,
            prompt=None,
            enhancer="rules",
            model="haiku",
        )
        assert result is None

    def test_enhance_merge_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = LLMEnhancerPlugin()
        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[],
            enhancer="rules",
            model="haiku",
        )
        assert result is None

    def test_enhance_operation_message_wrong_enhancer(self):
        """Should return None for wrong enhancer type."""
        plugin = LLMEnhancerPlugin()
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


class TestBatchProcessing:
    """Tests for batch processing in llm enhancer."""

    def test_batch_chunking(self, monkeypatch):
        """Should chunk items when over MAX_BATCH_SIZE."""
        from agentgit.enhancers import llm as llm_enhancer
        from agentgit.core import PromptResponse

        # Clear cache before test
        llm_enhancer.clear_message_cache()

        # Track calls to _run_llm
        calls = []

        def mock_run_llm(prompt, model="claude-cli-haiku"):
            calls.append(prompt)
            # Return valid JSON for 25 items
            items = {}
            for i in range(1, 26):
                items[str(i)] = f"Message {i}"
            return str(items).replace("'", '"')

        monkeypatch.setattr(llm_enhancer, "_run_llm", mock_run_llm)
        monkeypatch.setattr(llm_enhancer, "MAX_BATCH_SIZE", 25)

        # Create 50 prompt responses (should result in 2 batches)
        prompt_responses = []
        for i in range(50):
            prompt = Prompt(text=f"Prompt {i}", timestamp=f"2025-01-01T{i:02d}:00:00Z")
            turn = AssistantTurn(
                operations=[
                    FileOperation(
                        file_path=f"/project/file{i}.py",
                        operation_type=OperationType.WRITE,
                        timestamp=f"2025-01-01T{i:02d}:00:00Z",
                        content="code",
                        tool_id=f"tool_{i}",
                    )
                ],
                timestamp=f"2025-01-01T{i:02d}:00:00Z",
            )
            prompt_responses.append(PromptResponse(prompt=prompt, turns=[turn]))

        result = llm_enhancer.batch_enhance_prompt_responses(prompt_responses)

        # Should have made multiple calls due to chunking
        assert len(calls) >= 2
        assert isinstance(result, dict)

    def test_batch_empty_list(self):
        """Should handle empty list."""
        from agentgit.enhancers import llm as llm_enhancer

        llm_enhancer.clear_message_cache()
        result = llm_enhancer.batch_enhance_prompt_responses([])
        assert result == {}

    def test_batch_uses_cache(self, monkeypatch):
        """Should use cached results and not re-process."""
        from agentgit.enhancers import llm as llm_enhancer
        from agentgit.core import PromptResponse

        llm_enhancer.clear_message_cache()

        calls = []

        def mock_run_llm(prompt, model="claude-cli-haiku"):
            calls.append(prompt)
            # Return messages for both merge and turn items
            return '{"1": "Merge message", "2": "Turn message"}'

        monkeypatch.setattr(llm_enhancer, "_run_llm", mock_run_llm)

        prompt = Prompt(text="Test prompt", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/test.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                    tool_id="tool_1",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn])

        # First call should use API
        llm_enhancer.batch_enhance_prompt_responses([pr])
        first_call_count = len(calls)
        assert first_call_count == 1

        # Second call with same data should use cache
        llm_enhancer.batch_enhance_prompt_responses([pr])
        assert len(calls) == first_call_count  # No new calls

    def test_clear_message_cache(self):
        """Should clear the message cache."""
        from agentgit.enhancers import llm as llm_enhancer

        # Set up some cache entries
        llm_enhancer._message_cache["test_key"] = "test_value"
        assert len(llm_enhancer._message_cache) > 0

        llm_enhancer.clear_message_cache()
        assert llm_enhancer._message_cache == {}

    def test_get_prompt_key(self):
        """Should generate consistent prompt keys."""
        from agentgit.enhancers.llm import _get_prompt_key

        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        key1 = _get_prompt_key(prompt)
        key2 = _get_prompt_key(prompt)
        assert key1 == key2
        assert key1.startswith("prompt:")

    def test_get_turn_key(self):
        """Should generate consistent turn keys."""
        from agentgit.enhancers.llm import _get_turn_key

        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/test.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                    tool_id="tool_abc",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        key1 = _get_turn_key(turn)
        key2 = _get_turn_key(turn)
        assert key1 == key2
        assert key1.startswith("turn:")
        assert "tool_abc" in key1

    def test_get_turn_key_no_tool_id(self):
        """Should handle turns without tool_id."""
        from agentgit.enhancers.llm import _get_turn_key

        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/test.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        key = _get_turn_key(turn)
        assert key.startswith("turn:")

    def test_batch_handles_json_decode_error(self, monkeypatch):
        """Should handle malformed JSON response gracefully."""
        from agentgit.enhancers import llm as llm_enhancer
        from agentgit.core import PromptResponse

        llm_enhancer.clear_message_cache()

        def mock_run_llm(prompt, model="claude-cli-haiku"):
            return "not valid json at all"

        monkeypatch.setattr(llm_enhancer, "_run_llm", mock_run_llm)

        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/test.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                    tool_id="tool_1",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )

        # Should not raise, just return empty/partial cache
        result = llm_enhancer.batch_enhance_prompt_responses(
            [PromptResponse(prompt=prompt, turns=[turn])]
        )
        assert isinstance(result, dict)

    def test_batch_handles_markdown_code_blocks(self, monkeypatch):
        """Should strip markdown code blocks from response."""
        from agentgit.enhancers import llm as llm_enhancer
        from agentgit.core import PromptResponse

        llm_enhancer.clear_message_cache()

        def mock_run_llm(prompt, model="claude-cli-haiku"):
            return '```json\n{"1": "Add feature", "2": "Fix bug"}\n```'

        monkeypatch.setattr(llm_enhancer, "_run_llm", mock_run_llm)

        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/test.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                    tool_id="tool_1",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )

        result = llm_enhancer.batch_enhance_prompt_responses(
            [PromptResponse(prompt=prompt, turns=[turn])]
        )
        # Should have parsed successfully
        assert len(result) > 0


class TestBuildTurnContext:
    """Tests for _build_turn_context helper."""

    def test_build_turn_context_basic(self):
        """Should build context for a turn."""
        from agentgit.enhancers.llm import _build_turn_context
        from agentgit.core import AssistantContext

        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/auth.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="def auth(): pass",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
            context=AssistantContext(thinking="Adding authentication"),
        )
        context = _build_turn_context(turn)
        assert "auth.py" in context
        assert "Created" in context
        assert "Adding authentication" in context

    def test_build_turn_context_many_operations(self):
        """Should limit operations shown."""
        from agentgit.enhancers.llm import _build_turn_context

        operations = [
            FileOperation(
                file_path=f"/project/file{i}.py",
                operation_type=OperationType.WRITE,
                timestamp="2025-01-01T00:00:00Z",
                content="code",
            )
            for i in range(10)
        ]
        turn = AssistantTurn(operations=operations, timestamp="2025-01-01T00:00:00Z")
        context = _build_turn_context(turn)
        # Should mention there are more operations
        assert "more operations" in context


class TestContextBuilders:
    """Tests for llm context building helpers."""

    def test_build_operation_context_write(self):
        """Should build context for write operation."""
        from agentgit.enhancers.llm import _build_operation_context

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
        from agentgit.enhancers.llm import _build_operation_context

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
        from agentgit.enhancers.llm import _truncate_text

        short = "short"
        assert _truncate_text(short, 10) == "short"

        long = "a" * 100
        truncated = _truncate_text(long, 10)
        assert len(truncated) == 10
        assert truncated.endswith("...")

    def test_clean_message(self):
        """Should clean up generated message."""
        from agentgit.enhancers.llm import _clean_message

        # Remove quotes
        assert _clean_message('"Add feature"') == "Add feature"
        assert _clean_message("'Add feature'") == "Add feature"

        # Truncate long messages
        long_msg = "A" * 100
        cleaned = _clean_message(long_msg)
        assert len(cleaned) == 72
        assert cleaned.endswith("...")


class TestGenerateCommitMessages:
    """Tests for the generate_*_commit_message functions."""

    def test_generate_operation_commit_message(self):
        """Should generate a commit message for an operation."""
        operation = FileOperation(
            file_path="/project/src/auth.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="def authenticate(): pass",
        )
        config = EnhanceConfig(enhancer="rules", enabled=True)
        message = generate_operation_commit_message(operation, config)
        assert message is not None
        assert "auth.py" in message

    def test_generate_operation_commit_message_disabled(self):
        """Should return None when disabled."""
        operation = FileOperation(
            file_path="/project/src/auth.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="code",
        )
        config = EnhanceConfig(enabled=False)
        message = generate_operation_commit_message(operation, config)
        assert message is None

    def test_generate_operation_commit_message_default_config(self):
        """Should use default config if not provided."""
        operation = FileOperation(
            file_path="/project/src/test.py",
            operation_type=OperationType.WRITE,
            timestamp="2025-01-01T00:00:00Z",
            content="code",
        )
        message = generate_operation_commit_message(operation)
        assert message is not None

    def test_generate_turn_commit_message(self):
        """Should generate a commit message for a turn."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/utils.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        config = EnhanceConfig(enhancer="rules", enabled=True)
        message = generate_turn_commit_message(turn, config=config)
        assert message is not None

    def test_generate_turn_commit_message_with_prompt(self):
        """Should use prompt when generating turn message."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/utils.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        prompt = Prompt(text="Add utility functions", timestamp="2025-01-01T00:00:00Z")
        config = EnhanceConfig(enhancer="rules", enabled=True)
        message = generate_turn_commit_message(turn, prompt=prompt, config=config)
        assert message is not None

    def test_generate_turn_commit_message_disabled(self):
        """Should return None when disabled."""
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/utils.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        config = EnhanceConfig(enabled=False)
        message = generate_turn_commit_message(turn, config=config)
        assert message is None

    def test_generate_merge_commit_message(self):
        """Should generate a merge commit message."""
        prompt = Prompt(
            text="Add user authentication with JWT tokens",
            timestamp="2025-01-01T00:00:00Z",
        )
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/project/src/auth.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        config = EnhanceConfig(enhancer="rules", enabled=True)
        message = generate_merge_commit_message(prompt, [turn], config)
        assert message is not None

    def test_generate_merge_commit_message_disabled(self):
        """Should return None when disabled."""
        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        config = EnhanceConfig(enabled=False)
        message = generate_merge_commit_message(prompt, [], config)
        assert message is None


class TestPreprocessBatchEnhancement:
    """Tests for preprocess_batch_enhancement function."""

    def test_preprocess_batch_enhancement_disabled(self):
        """Should do nothing when disabled."""
        config = EnhanceConfig(enabled=False)
        # Should not raise
        preprocess_batch_enhancement([], config)

    def test_preprocess_batch_enhancement_rules(self):
        """Should be no-op for rules enhancer."""
        config = EnhanceConfig(enhancer="rules", enabled=True)
        # Should not raise
        preprocess_batch_enhancement([], config)

    def test_preprocess_batch_enhancement_llm(self, monkeypatch):
        """Should call batch_enhance_prompt_responses for llm enhancer."""
        from agentgit.enhancers import llm as llm_enhancer

        calls = []

        def mock_batch(prompt_responses, model):
            calls.append((prompt_responses, model))
            return {}

        monkeypatch.setattr(llm_enhancer, "batch_enhance_prompt_responses", mock_batch)

        config = EnhanceConfig(enhancer="llm", model="sonnet", enabled=True)
        prompt = Prompt(text="Test", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/test.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        pr = PromptResponse(prompt=prompt, turns=[turn])

        preprocess_batch_enhancement([pr], config)
        assert len(calls) == 1
        assert calls[0][1] == "sonnet"

    def test_preprocess_batch_enhancement_default_config(self, monkeypatch):
        """Should use default config if not provided."""
        # Default config uses 'rules' which is a no-op
        # Just verify it doesn't raise
        preprocess_batch_enhancement([])


class TestCurateTurnContext:
    """Tests for curate_turn_context function."""

    def test_curate_disabled(self):
        """Should return None when enhancement is disabled."""
        from agentgit.enhance import curate_turn_context

        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
            context=AssistantContext(thinking="Some reasoning"),
        )
        config = EnhanceConfig(enabled=False)
        assert curate_turn_context(turn, config) is None

    def test_curate_with_rules_enhancer(self):
        """Rules enhancer doesn't implement curation, returns None."""
        from agentgit.enhance import curate_turn_context

        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
            context=AssistantContext(thinking="Some reasoning"),
        )
        config = EnhanceConfig(enhancer="rules", enabled=True)
        # Rules enhancer doesn't curate, returns None
        assert curate_turn_context(turn, config) is None


class TestMergeMessagePreservesPrompt:
    """Tests for merge message preserving exact user prompts."""

    def test_self_contained_prompt_used_as_is(self, monkeypatch):
        """Self-contained prompts should be used directly without LLM."""
        from agentgit.enhancers import llm as llm_enhancer

        llm_enhancer.clear_message_cache()

        # Mock _run_llm to ensure it's NOT called
        calls = []

        def mock_run_llm(prompt, model="claude-cli-haiku"):
            calls.append(prompt)
            return "Should not be used"

        monkeypatch.setattr(llm_enhancer, "_run_llm", mock_run_llm)

        plugin = LLMEnhancerPlugin()
        prompt = Prompt(
            text="Add user authentication",
            timestamp="2025-01-01T00:00:00Z",
        )
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/auth.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[turn],
            enhancer="llm",
            model=None,
        )
        # Should use the prompt directly, no LLM call
        assert result == "Add user authentication"
        assert len(calls) == 0

    def test_referential_prompt_gets_context(self, monkeypatch):
        """Referential prompts should get context appended."""
        from agentgit.enhancers import llm as llm_enhancer

        llm_enhancer.clear_message_cache()

        def mock_run_llm(prompt, model="claude-cli-haiku"):
            return "Add JWT authentication"

        monkeypatch.setattr(llm_enhancer, "_run_llm", mock_run_llm)

        plugin = LLMEnhancerPlugin()
        prompt = Prompt(text="yes", timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[
                FileOperation(
                    file_path="/auth.py",
                    operation_type=OperationType.WRITE,
                    timestamp="2025-01-01T00:00:00Z",
                    content="jwt code",
                )
            ],
            timestamp="2025-01-01T00:00:00Z",
            context=AssistantContext(thinking="Implementing JWT auth"),
        )
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[turn],
            enhancer="llm",
            model=None,
        )
        # Should combine: "yes - Add JWT authentication"
        assert result == "yes - Add JWT authentication"

    def test_long_prompt_truncated(self, monkeypatch):
        """Long self-contained prompts should be truncated."""
        from agentgit.enhancers import llm as llm_enhancer

        llm_enhancer.clear_message_cache()

        plugin = LLMEnhancerPlugin()
        long_text = "Add " + "x" * 100
        prompt = Prompt(text=long_text, timestamp="2025-01-01T00:00:00Z")
        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_enhance_merge_message(
            prompt=prompt,
            turns=[turn],
            enhancer="llm",
            model=None,
        )
        assert len(result) <= 72
        assert result.endswith("...")


class TestCurateTurnContextHook:
    """Tests for the agentgit_curate_turn_context hook."""

    def test_wrong_enhancer_returns_none(self):
        """Should return None for non-llm enhancer."""
        plugin = LLMEnhancerPlugin()
        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
            context=AssistantContext(thinking="Some context"),
        )
        result = plugin.agentgit_curate_turn_context(
            turn=turn, enhancer="rules", model=None
        )
        assert result is None

    def test_no_context_returns_none(self):
        """Should return None when turn has no context."""
        plugin = LLMEnhancerPlugin()
        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
        )
        result = plugin.agentgit_curate_turn_context(
            turn=turn, enhancer="llm", model=None
        )
        assert result is None

    def test_short_context_returned_as_is(self):
        """Short context should be returned without LLM call."""
        plugin = LLMEnhancerPlugin()
        short_context = "Brief reasoning about the change."
        turn = AssistantTurn(
            operations=[],
            timestamp="2025-01-01T00:00:00Z",
            context=AssistantContext(thinking=short_context),
        )
        result = plugin.agentgit_curate_turn_context(
            turn=turn, enhancer="llm", model=None
        )
        assert result == short_context
