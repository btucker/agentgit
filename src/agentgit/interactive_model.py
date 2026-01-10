"""
Interactive Mental Model - Time travel and focused prompting.

Extends the living mental model with:
1. Version history / snapshots for "time travel" through the model's evolution
2. Interactive prompting - clicking on diagram elements generates contextual prompts

The key insight: the diagram isn't just a visualization, it's an INTERFACE for
directing the AI agent. Clicking on a component lets you ask focused questions
or make targeted changes to that part of the system.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path
import json
import copy

from .mental_model import (
    MentalModel,
    MentalModelNode,
    MentalModelEdge,
    ModelUpdate,
)
from .core import Prompt


@dataclass
class ModelSnapshot:
    """A point-in-time snapshot of the mental model."""

    version: int
    timestamp: datetime
    model_state: MentalModel
    trigger_prompt_id: Optional[str] = None
    trigger_prompt_text: Optional[str] = None
    change_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "trigger_prompt_id": self.trigger_prompt_id,
            "trigger_prompt_text": self.trigger_prompt_text,
            "change_summary": self.change_summary,
            "model_state": json.loads(self.model_state.to_json()),
        }


@dataclass
class FocusedPrompt:
    """A prompt generated from interacting with the diagram."""

    prompt_text: str
    context: str  # What part of the diagram this relates to
    suggested_files: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)  # Related nodes


class InteractiveMentalModel:
    """
    A mental model with time travel and interactive prompting capabilities.

    This transforms the diagram from a passive visualization into an
    active interface for directing an AI coding agent.
    """

    def __init__(self, project_name: str = ""):
        self.current_model = MentalModel(project_name=project_name)
        self.history: list[ModelSnapshot] = []
        self.current_version: int = 0

        # Map from node IDs to associated files (for focused context)
        self.node_files: dict[str, set[str]] = {}

        # Map from node IDs to prompts that modified them (for provenance)
        self.node_prompt_history: dict[str, list[str]] = {}

    def apply_update(
        self,
        update: ModelUpdate,
        prompt: Optional[Prompt] = None,
        affected_files: Optional[list[str]] = None,
    ) -> None:
        """Apply an update and save a snapshot for history."""

        # Save snapshot BEFORE applying update
        self._save_snapshot(prompt, update.reasoning)

        # Apply the update
        for node_id in update.remove_node_ids:
            self.current_model.remove_node(node_id)

        for source_id, target_id, relationship in update.remove_edges:
            self.current_model.edges = [
                e for e in self.current_model.edges
                if not (e.source_id == source_id and
                       e.target_id == target_id and
                       e.relationship == relationship)
            ]

        for node in update.add_nodes:
            self.current_model.add_node(node, prompt.prompt_id if prompt else None)

            # Track file associations
            if affected_files:
                if node.id not in self.node_files:
                    self.node_files[node.id] = set()
                self.node_files[node.id].update(affected_files)

            # Track prompt history
            if prompt:
                if node.id not in self.node_prompt_history:
                    self.node_prompt_history[node.id] = []
                self.node_prompt_history[node.id].append(prompt.prompt_id)

        for edge in update.add_edges:
            edge.introduced_by_prompt_id = prompt.prompt_id if prompt else None
            self.current_model.add_edge(edge)

        self.current_model.version += 1
        self.current_model.last_updated = datetime.now()
        self.current_version = self.current_model.version

    def _save_snapshot(self, prompt: Optional[Prompt], change_summary: str) -> None:
        """Save current state as a snapshot."""
        snapshot = ModelSnapshot(
            version=self.current_version,
            timestamp=datetime.now(),
            model_state=copy.deepcopy(self.current_model),
            trigger_prompt_id=prompt.prompt_id if prompt else None,
            trigger_prompt_text=prompt.text if prompt else None,
            change_summary=change_summary,
        )
        self.history.append(snapshot)

    # ===== TIME TRAVEL =====

    def get_version(self, version: int) -> Optional[MentalModel]:
        """Get the model state at a specific version."""
        for snapshot in self.history:
            if snapshot.version == version:
                return copy.deepcopy(snapshot.model_state)
        if version == self.current_version:
            return copy.deepcopy(self.current_model)
        return None

    def get_timeline(self) -> list[dict]:
        """Get a summary of all versions for timeline display."""
        timeline = []
        for snapshot in self.history:
            timeline.append({
                "version": snapshot.version,
                "timestamp": snapshot.timestamp.isoformat(),
                "prompt_preview": (
                    snapshot.trigger_prompt_text[:50] + "..."
                    if snapshot.trigger_prompt_text and len(snapshot.trigger_prompt_text) > 50
                    else snapshot.trigger_prompt_text
                ),
                "change_summary": snapshot.change_summary,
                "node_count": len(snapshot.model_state.nodes),
                "edge_count": len(snapshot.model_state.edges),
            })

        # Add current state
        timeline.append({
            "version": self.current_version,
            "timestamp": self.current_model.last_updated.isoformat() if self.current_model.last_updated else None,
            "prompt_preview": None,
            "change_summary": "Current state",
            "node_count": len(self.current_model.nodes),
            "edge_count": len(self.current_model.edges),
        })

        return timeline

    def diff_versions(self, from_version: int, to_version: int) -> dict:
        """Get the diff between two versions."""
        from_model = self.get_version(from_version)
        to_model = self.get_version(to_version)

        if not from_model or not to_model:
            return {"error": "Version not found"}

        from_nodes = set(from_model.nodes.keys())
        to_nodes = set(to_model.nodes.keys())

        added_nodes = to_nodes - from_nodes
        removed_nodes = from_nodes - to_nodes

        # Find modified nodes (same ID but different content)
        modified_nodes = []
        for node_id in from_nodes & to_nodes:
            from_node = from_model.nodes[node_id]
            to_node = to_model.nodes[node_id]
            if from_node.label != to_node.label or from_node.description != to_node.description:
                modified_nodes.append(node_id)

        return {
            "from_version": from_version,
            "to_version": to_version,
            "added_nodes": list(added_nodes),
            "removed_nodes": list(removed_nodes),
            "modified_nodes": modified_nodes,
        }

    # ===== INTERACTIVE PROMPTING =====

    def generate_node_prompt(self, node_id: str, intent: str = "explore") -> FocusedPrompt:
        """
        Generate a focused prompt for a specific node in the diagram.

        This is called when a user clicks on a node and wants to interact
        with the AI about that specific part of the system.

        Intent options:
        - "explore": Learn more about this component
        - "modify": Make changes to this component
        - "connect": Add connections to/from this component
        - "debug": Investigate issues with this component
        - "test": Add tests for this component
        """
        node = self.current_model.nodes.get(node_id)
        if not node:
            return FocusedPrompt(
                prompt_text=f"I don't see a component with ID '{node_id}' in the current model.",
                context="Unknown node",
            )

        # Get related context
        connected_nodes = self._get_connected_nodes(node_id)
        associated_files = list(self.node_files.get(node_id, []))
        prompt_history = self.node_prompt_history.get(node_id, [])

        # Build context string
        context_parts = [f"**{node.label}** ({node.node_type})"]
        if node.description:
            context_parts.append(f"Description: {node.description}")
        if connected_nodes:
            context_parts.append(f"Connected to: {', '.join(connected_nodes)}")
        if associated_files:
            context_parts.append(f"Related files: {', '.join(associated_files[:5])}")

        context = "\n".join(context_parts)

        # Generate prompt based on intent
        prompts = {
            "explore": f"Tell me more about the {node.label} component. What does it do, how does it work, and how does it fit into the overall system?",

            "modify": f"I want to make changes to the {node.label} component. What are the key files I should look at, and what should I be careful about when modifying it?",

            "connect": f"I want to add a new connection to/from the {node.label} component. What other parts of the system might it need to interact with?",

            "debug": f"I'm having issues with the {node.label} component. Can you help me understand how it works and identify potential problem areas?",

            "test": f"I want to add tests for the {node.label} component. What are the key behaviors I should test, and what edge cases should I consider?",

            "explain": f"Explain the {node.label} component to me as if I'm new to this codebase. What's its purpose and how does it relate to the rest of the system?",

            "refactor": f"I'm considering refactoring the {node.label} component. What improvements could be made, and what would be the impact on connected components?",
        }

        prompt_text = prompts.get(intent, prompts["explore"])

        # Add context to make it more specific
        if associated_files:
            prompt_text += f"\n\nRelevant files: {', '.join(associated_files[:3])}"

        return FocusedPrompt(
            prompt_text=prompt_text,
            context=context,
            suggested_files=associated_files,
            node_ids=[node_id] + list(connected_nodes),
        )

    def generate_edge_prompt(self, source_id: str, target_id: str) -> FocusedPrompt:
        """Generate a prompt focused on a relationship between two nodes."""
        source = self.current_model.nodes.get(source_id)
        target = self.current_model.nodes.get(target_id)

        if not source or not target:
            return FocusedPrompt(
                prompt_text="Could not find one or both nodes.",
                context="Unknown relationship",
            )

        # Find the edge
        edge = None
        for e in self.current_model.edges:
            if e.source_id == source_id and e.target_id == target_id:
                edge = e
                break

        relationship_desc = edge.relationship if edge else "relates to"

        prompt_text = f"Explain how {source.label} {relationship_desc} {target.label}. How do these components interact, and what data or control flows between them?"

        return FocusedPrompt(
            prompt_text=prompt_text,
            context=f"Relationship: {source.label} → {target.label}",
            suggested_files=list(
                self.node_files.get(source_id, set()) |
                self.node_files.get(target_id, set())
            ),
            node_ids=[source_id, target_id],
        )

    def generate_area_prompt(self, node_ids: list[str]) -> FocusedPrompt:
        """Generate a prompt for a selected area of the diagram (multiple nodes)."""
        nodes = [self.current_model.nodes[nid] for nid in node_ids if nid in self.current_model.nodes]

        if not nodes:
            return FocusedPrompt(
                prompt_text="No valid nodes selected.",
                context="Empty selection",
            )

        node_names = [n.label for n in nodes]
        all_files = set()
        for nid in node_ids:
            all_files.update(self.node_files.get(nid, set()))

        prompt_text = f"I'm looking at this area of the system: {', '.join(node_names)}. How do these components work together? What's the data flow and control flow between them?"

        return FocusedPrompt(
            prompt_text=prompt_text,
            context=f"Selected components: {', '.join(node_names)}",
            suggested_files=list(all_files),
            node_ids=node_ids,
        )

    def _get_connected_nodes(self, node_id: str) -> set[str]:
        """Get all nodes connected to the given node."""
        connected = set()
        for edge in self.current_model.edges:
            if edge.source_id == node_id:
                if edge.target_id in self.current_model.nodes:
                    connected.add(self.current_model.nodes[edge.target_id].label)
            elif edge.target_id == node_id:
                if edge.source_id in self.current_model.nodes:
                    connected.add(self.current_model.nodes[edge.source_id].label)
        return connected

    # ===== EXPORT =====

    def to_interactive_html(self) -> str:
        """
        Export the model as an interactive HTML page.

        This would render the Mermaid diagram with click handlers that
        trigger the focused prompt generation.
        """
        mermaid_code = self.current_model.to_mermaid()
        timeline_data = json.dumps(self.get_timeline())

        # This is a simplified template - a real implementation would use
        # a proper frontend framework and Mermaid's click callbacks
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Living Mental Model</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; }}
        .container {{ display: flex; gap: 20px; }}
        .diagram {{ flex: 2; }}
        .sidebar {{ flex: 1; }}
        .timeline {{ border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }}
        .timeline-item {{ padding: 5px; cursor: pointer; border-bottom: 1px solid #eee; }}
        .timeline-item:hover {{ background: #f5f5f5; }}
        .prompt-panel {{ border: 1px solid #ddd; padding: 10px; }}
        .prompt-panel h3 {{ margin-top: 0; }}
        .prompt-text {{ background: #f9f9f9; padding: 10px; border-radius: 4px; }}
        .node {{ cursor: pointer; }}
        .version-slider {{ width: 100%; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🧠 Living Mental Model</h1>

    <div class="container">
        <div class="diagram">
            <h2>System Architecture</h2>
            <div class="mermaid" id="diagram">
{mermaid_code}
            </div>
            <p><em>Click on any node to generate a focused prompt</em></p>

            <h3>Time Travel</h3>
            <input type="range" class="version-slider" id="versionSlider"
                   min="0" max="{self.current_version}" value="{self.current_version}">
            <span id="versionLabel">Version {self.current_version}</span>
        </div>

        <div class="sidebar">
            <div class="timeline">
                <h3>📜 History</h3>
                <div id="timeline"></div>
            </div>

            <div class="prompt-panel">
                <h3>💬 Focused Prompt</h3>
                <p>Click a node in the diagram to generate a context-aware prompt.</p>
                <div id="promptArea"></div>
            </div>
        </div>
    </div>

    <script>
        const timelineData = {timeline_data};

        // Render timeline
        const timelineEl = document.getElementById('timeline');
        timelineData.forEach(item => {{
            const div = document.createElement('div');
            div.className = 'timeline-item';
            div.innerHTML = `
                <strong>v${{item.version}}</strong>
                <br><small>${{item.change_summary || 'Initial state'}}</small>
                <br><small style="color: #666">${{item.node_count}} nodes, ${{item.edge_count}} edges</small>
            `;
            div.onclick = () => loadVersion(item.version);
            timelineEl.appendChild(div);
        }});

        // Initialize Mermaid with click callbacks
        mermaid.initialize({{
            startOnLoad: true,
            securityLevel: 'loose',
        }});

        // Version slider
        document.getElementById('versionSlider').oninput = (e) => {{
            document.getElementById('versionLabel').textContent = 'Version ' + e.target.value;
            loadVersion(parseInt(e.target.value));
        }};

        function loadVersion(version) {{
            // In a real implementation, this would fetch the diagram for that version
            console.log('Load version:', version);
        }}

        // Node click handling (would be wired up by Mermaid in real implementation)
        function onNodeClick(nodeId) {{
            const promptArea = document.getElementById('promptArea');
            promptArea.innerHTML = `
                <div class="prompt-text">
                    <strong>Generated prompt for: ${{nodeId}}</strong>
                    <p>Tell me more about the ${{nodeId}} component. What does it do,
                    how does it work, and how does it fit into the overall system?</p>
                    <button onclick="copyPrompt()">📋 Copy to Clipboard</button>
                </div>
            `;
        }}
    </script>
</body>
</html>"""
        return html

    def save(self, path: Path) -> None:
        """Save the complete interactive model state."""
        state = {
            "current_model": json.loads(self.current_model.to_json()),
            "history": [s.to_dict() for s in self.history],
            "current_version": self.current_version,
            "node_files": {k: list(v) for k, v in self.node_files.items()},
            "node_prompt_history": self.node_prompt_history,
        }
        Path(path).write_text(json.dumps(state, indent=2))


# Example usage
if __name__ == "__main__":
    from .mental_model import MentalModelNode, MentalModelEdge, ModelUpdate
    from .core import Prompt

    # Create an interactive model
    model = InteractiveMentalModel(project_name="E-commerce Platform")

    # Simulate evolution through multiple prompts
    prompts = [
        Prompt(text="Create a basic product catalog", timestamp=datetime.now()),
        Prompt(text="Add shopping cart functionality", timestamp=datetime.now()),
        Prompt(text="Implement checkout with Stripe", timestamp=datetime.now()),
    ]

    updates = [
        ModelUpdate(
            reasoning="Adding product catalog components",
            add_nodes=[
                MentalModelNode(id="products", label="Product Catalog", node_type="component"),
                MentalModelNode(id="db", label="Product Database", node_type="data"),
            ],
            add_edges=[
                MentalModelEdge(source_id="products", target_id="db", relationship="uses"),
            ],
        ),
        ModelUpdate(
            reasoning="Adding shopping cart",
            add_nodes=[
                MentalModelNode(id="cart", label="Shopping Cart", node_type="component"),
            ],
            add_edges=[
                MentalModelEdge(source_id="cart", target_id="products", relationship="uses"),
            ],
        ),
        ModelUpdate(
            reasoning="Adding checkout and payment processing",
            add_nodes=[
                MentalModelNode(id="checkout", label="Checkout Flow", node_type="process"),
                MentalModelNode(id="stripe", label="Stripe Integration", node_type="interface"),
            ],
            add_edges=[
                MentalModelEdge(source_id="checkout", target_id="cart", relationship="uses"),
                MentalModelEdge(source_id="checkout", target_id="stripe", relationship="uses"),
            ],
        ),
    ]

    # Track file associations
    file_associations = [
        ["src/products/catalog.py", "src/products/models.py"],
        ["src/cart/cart.py", "src/cart/session.py"],
        ["src/checkout/flow.py", "src/payments/stripe.py"],
    ]

    # Apply updates
    for prompt, update, files in zip(prompts, updates, file_associations):
        model.apply_update(update, prompt, files)
        print(f"Applied: {update.reasoning}")

    # Show timeline
    print("\n=== Timeline ===")
    for item in model.get_timeline():
        print(f"  v{item['version']}: {item['change_summary']} ({item['node_count']} nodes)")

    # Show diff
    print("\n=== Diff v0 → v3 ===")
    diff = model.diff_versions(0, 3)
    print(f"  Added: {diff['added_nodes']}")

    # Generate focused prompts
    print("\n=== Focused Prompts ===")

    prompt = model.generate_node_prompt("checkout", "explore")
    print(f"\n[Explore Checkout]")
    print(f"  Context: {prompt.context}")
    print(f"  Prompt: {prompt.prompt_text[:100]}...")

    prompt = model.generate_node_prompt("cart", "modify")
    print(f"\n[Modify Cart]")
    print(f"  Context: {prompt.context}")
    print(f"  Prompt: {prompt.prompt_text[:100]}...")

    prompt = model.generate_edge_prompt("checkout", "stripe")
    print(f"\n[Checkout → Stripe relationship]")
    print(f"  Prompt: {prompt.prompt_text}")

    # Final diagram
    print("\n=== Current Diagram ===")
    print(model.current_model.to_mermaid())
