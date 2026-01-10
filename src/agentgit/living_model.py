"""
Living Mental Model - Real-time visualization that evolves with your codebase.

This module ties together the mental model with the transcript watcher,
creating a visualization that automatically adapts as an AI coding agent
makes changes to the codebase.

The key insight: as you work with AI coding agents, the important thing isn't
the specific code structure, but maintaining a clear mental model of how
the system works conceptually. This tool keeps that model current.
"""

from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import os

from .core import FileOperation, Transcript
from .mental_model import (
    MentalModel,
    ModelUpdate,
    apply_update,
    generate_update_prompt,
    parse_update_response,
)


class LivingMentalModel:
    """
    A mental model that automatically updates as code changes.

    Usage:
        model = LivingMentalModel(project_name="My App")

        # When changes happen (e.g., from watcher callback):
        model.process_changes(operations)

        # Get current visualization:
        print(model.to_mermaid())

        # Save/load state:
        model.save("mental_model.json")
        model = LivingMentalModel.load("mental_model.json")
    """

    def __init__(
        self,
        project_name: str = "",
        model_path: Optional[Path] = None,
        ai_callback: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize a living mental model.

        Args:
            project_name: Name for this project's model
            model_path: Optional path to persist the model (auto-saves on update)
            ai_callback: Function to call AI for model updates.
                        Signature: (prompt: str) -> str (AI response)
                        If None, uses a simple rule-based updater.
        """
        self.model = MentalModel(project_name=project_name)
        self.model_path = Path(model_path) if model_path else None
        self.ai_callback = ai_callback

        # Track what we've seen for incremental updates
        self._processed_tool_ids: set[str] = set()

        # Load existing model if path provided and exists
        if self.model_path and self.model_path.exists():
            self._load()

    def process_changes(
        self,
        operations: list[FileOperation],
        force: bool = False,
    ) -> Optional[ModelUpdate]:
        """
        Process a batch of file operations and update the mental model.

        Args:
            operations: List of file operations from a transcript
            force: If True, reprocess even already-seen operations

        Returns:
            The ModelUpdate that was applied, or None if no update needed
        """
        # Filter to new operations only (unless forced)
        if not force:
            new_ops = [
                op for op in operations
                if op.tool_id and op.tool_id not in self._processed_tool_ids
            ]
        else:
            new_ops = operations

        if not new_ops:
            return None

        # Mark as processed
        for op in new_ops:
            if op.tool_id:
                self._processed_tool_ids.add(op.tool_id)

        # Get the update
        update = self._compute_update(new_ops)

        if update and not update.is_empty():
            # Get prompt_id from first operation with a prompt
            prompt_id = None
            for op in new_ops:
                if op.prompt:
                    prompt_id = op.prompt.prompt_id
                    break

            apply_update(self.model, update, prompt_id)

            # Auto-save if path configured
            if self.model_path:
                self._save()

        return update

    def process_transcript(self, transcript: Transcript) -> list[ModelUpdate]:
        """
        Process an entire transcript and return all updates made.

        This is useful for building a mental model from a complete session.
        """
        updates = []

        # Group operations by prompt for better context
        ops_by_prompt: dict[str, list[FileOperation]] = {}
        for op in transcript.operations:
            key = op.prompt.prompt_id if op.prompt else "unknown"
            if key not in ops_by_prompt:
                ops_by_prompt[key] = []
            ops_by_prompt[key].append(op)

        # Process each group
        for prompt_id, ops in ops_by_prompt.items():
            update = self.process_changes(ops)
            if update:
                updates.append(update)

        return updates

    def _compute_update(self, operations: list[FileOperation]) -> Optional[ModelUpdate]:
        """Compute an update to the mental model based on operations."""

        if self.ai_callback:
            return self._compute_update_with_ai(operations)
        else:
            return self._compute_update_rule_based(operations)

    def _compute_update_with_ai(self, operations: list[FileOperation]) -> Optional[ModelUpdate]:
        """Use AI to interpret changes and propose model updates."""

        prompt = generate_update_prompt(self.model, operations)
        response = self.ai_callback(prompt)
        return parse_update_response(response)

    def _compute_update_rule_based(self, operations: list[FileOperation]) -> Optional[ModelUpdate]:
        """
        Simple rule-based updater for when no AI is available.

        This provides basic structure based on file patterns, but won't
        capture semantic meaning like an AI would.
        """
        from .mental_model import MentalModelNode, MentalModelEdge

        update = ModelUpdate(reasoning="Rule-based inference from file patterns")

        # Simple heuristics based on file paths
        for op in operations:
            path = op.file_path.lower()
            path_parts = Path(op.file_path).parts

            # Skip if we already have a node for this
            # (very simplistic - AI would do much better)
            if len(path_parts) < 2:
                continue

            # Infer component from directory structure
            if "src" in path_parts:
                idx = path_parts.index("src")
                if idx + 1 < len(path_parts):
                    component_name = path_parts[idx + 1]
                    node_id = f"component_{component_name}"

                    if node_id not in self.model.nodes:
                        # Infer type from common patterns
                        node_type = "component"
                        if "api" in component_name or "routes" in component_name:
                            node_type = "interface"
                        elif "service" in component_name:
                            node_type = "service"
                        elif "model" in component_name or "db" in component_name:
                            node_type = "data"

                        update.add_nodes.append(MentalModelNode(
                            id=node_id,
                            label=component_name.replace("_", " ").title(),
                            node_type=node_type,
                            description=f"Inferred from {op.file_path}",
                        ))

        return update if not update.is_empty() else None

    def to_mermaid(self) -> str:
        """Get current model as Mermaid diagram."""
        return self.model.to_mermaid()

    def to_json(self) -> str:
        """Get current model as JSON."""
        return self.model.to_json()

    def save(self, path: Optional[Path] = None) -> None:
        """Save the model to a file."""
        save_path = Path(path) if path else self.model_path
        if not save_path:
            raise ValueError("No path provided and no default path configured")
        save_path.write_text(self.model.to_json())

    def _save(self) -> None:
        """Internal save to configured path."""
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_path.write_text(self.model.to_json())

    def _load(self) -> None:
        """Internal load from configured path."""
        if self.model_path and self.model_path.exists():
            self.model = MentalModel.from_json(self.model_path.read_text())

    @classmethod
    def load(cls, path: Path, ai_callback: Optional[Callable[[str], str]] = None) -> "LivingMentalModel":
        """Load a model from a file."""
        instance = cls(model_path=path, ai_callback=ai_callback)
        return instance


def create_watcher_callback(
    living_model: LivingMentalModel,
    output_path: Optional[Path] = None,
    on_update: Optional[Callable[[MentalModel, ModelUpdate], None]] = None,
):
    """
    Create a callback function for use with TranscriptWatcher.

    This integrates the living model with agentgit's watch functionality,
    automatically updating the mental model as the transcript grows.

    Usage:
        from agentgit import watch_transcript
        from agentgit.living_model import LivingMentalModel, create_watcher_callback

        model = LivingMentalModel("My Project")
        callback = create_watcher_callback(model, output_path=Path("model.md"))

        # Watch will call our callback on each update
        watch_transcript(
            "transcript.jsonl",
            output_dir="./output",
            on_update=lambda count: callback(latest_operations)
        )
    """

    def callback(transcript: Transcript):
        """Process new operations and update model."""
        update = living_model.process_changes(transcript.operations)

        if update and not update.is_empty():
            # Write updated visualization
            if output_path:
                mermaid = living_model.to_mermaid()
                output_path.write_text(f"# Mental Model\n\n```mermaid\n{mermaid}\n```\n")

            # Call user callback
            if on_update:
                on_update(living_model.model, update)

    return callback


# Convenience function for quick visualization
def visualize_transcript(
    transcript: Transcript,
    ai_callback: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Generate a mental model visualization from a transcript.

    This is a one-shot function for visualizing a complete transcript.
    For live updates, use LivingMentalModel with a watcher.
    """
    model = LivingMentalModel(
        project_name=transcript.session_id or "Unknown Project",
        ai_callback=ai_callback,
    )
    model.process_transcript(transcript)
    return model.to_mermaid()


# Example with mock AI for testing
if __name__ == "__main__":
    from .core import FileOperation, OperationType, Prompt, AssistantContext

    # Mock AI callback that returns a simple response
    def mock_ai(prompt: str) -> str:
        return """```json
{
    "reasoning": "The changes introduce a new authentication system with user management",
    "updates": {
        "add_nodes": [
            {"id": "auth", "label": "Authentication", "node_type": "service", "description": "Handles user auth"},
            {"id": "users", "label": "User Management", "node_type": "component", "description": "User CRUD"}
        ],
        "add_edges": [
            {"source_id": "auth", "target_id": "users", "relationship": "uses", "label": "validates"}
        ],
        "remove_node_ids": [],
        "remove_edges": []
    }
}
```"""

    # Create living model
    model = LivingMentalModel(project_name="Test App", ai_callback=mock_ai)

    # Simulate some operations
    ops = [
        FileOperation(
            file_path="src/auth/login.py",
            operation_type=OperationType.WRITE,
            content="# Login handler",
            timestamp=datetime.now(),
            tool_id="op1",
            prompt=Prompt(text="Add user authentication", timestamp=datetime.now()),
            assistant_context=AssistantContext(
                thinking="I'll create an authentication module with login/logout"
            ),
        ),
        FileOperation(
            file_path="src/users/models.py",
            operation_type=OperationType.WRITE,
            content="# User model",
            timestamp=datetime.now(),
            tool_id="op2",
        ),
    ]

    # Process changes
    update = model.process_changes(ops)

    print("=== Update Applied ===")
    if update:
        print(f"Reasoning: {update.reasoning}")
        print(f"Added nodes: {[n.label for n in update.add_nodes]}")
        print(f"Added edges: {len(update.add_edges)}")

    print("\n=== Current Mental Model ===")
    print(model.to_mermaid())
