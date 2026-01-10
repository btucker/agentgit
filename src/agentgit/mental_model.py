"""
Living Mental Model - A visualization that evolves with your codebase.

This module provides a "mental model" diagram that automatically adapts as code changes,
capturing the semantic structure of a system rather than just its file layout.

The key insight: when working with AI coding agents, what matters isn't the exact code
structure, but understanding *how the system works conceptually* and how that evolves.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json
import hashlib

from .core import FileOperation, Prompt, AssistantContext


@dataclass
class MentalModelNode:
    """A node in the mental model - represents a concept, component, or capability."""

    id: str  # Stable identifier
    label: str  # Human-readable name
    node_type: str  # e.g., "component", "service", "concept", "data", "interface"
    description: str = ""  # What this node represents

    # Provenance - where did this understanding come from?
    introduced_by_prompt_id: Optional[str] = None
    last_updated_prompt_id: Optional[str] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class MentalModelEdge:
    """A relationship between nodes in the mental model."""

    source_id: str
    target_id: str
    relationship: str  # e.g., "uses", "contains", "transforms", "depends_on"
    label: str = ""  # Optional label for the edge

    # Provenance
    introduced_by_prompt_id: Optional[str] = None


@dataclass
class MentalModel:
    """
    A living representation of the system's conceptual structure.

    This is NOT a file tree or class diagram - it's a semantic model of
    what the system *does* and how its parts relate conceptually.
    """

    nodes: dict[str, MentalModelNode] = field(default_factory=dict)
    edges: list[MentalModelEdge] = field(default_factory=list)

    # Metadata
    project_name: str = ""
    version: int = 0  # Incremented on each update
    last_updated: Optional[datetime] = None

    # History of what changed
    changelog: list[dict] = field(default_factory=list)

    def add_node(self, node: MentalModelNode, prompt_id: Optional[str] = None) -> None:
        """Add or update a node in the model."""
        if node.id in self.nodes:
            # Update existing
            existing = self.nodes[node.id]
            existing.label = node.label
            existing.description = node.description
            existing.last_updated_prompt_id = prompt_id
        else:
            # Add new
            node.introduced_by_prompt_id = prompt_id
            self.nodes[node.id] = node

    def add_edge(self, edge: MentalModelEdge) -> None:
        """Add an edge if it doesn't already exist."""
        for existing in self.edges:
            if (existing.source_id == edge.source_id and
                existing.target_id == edge.target_id and
                existing.relationship == edge.relationship):
                return  # Already exists
        self.edges.append(edge)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [e for e in self.edges
                         if e.source_id != node_id and e.target_id != node_id]

    def to_mermaid(self) -> str:
        """Export the mental model as a Mermaid diagram."""
        lines = ["graph TD"]

        # Group nodes by type for styling
        node_types = {}
        for node in self.nodes.values():
            if node.node_type not in node_types:
                node_types[node.node_type] = []
            node_types[node.node_type].append(node)

        # Add nodes with type-specific shapes
        shape_map = {
            "component": ("[", "]"),      # Rectangle
            "service": ("[[", "]]"),      # Subroutine shape
            "concept": ("(", ")"),        # Rounded
            "data": ("[(", ")]"),         # Cylinder (database)
            "interface": ("{{", "}}"),    # Hexagon
            "process": ("([", "])"),      # Stadium
        }

        for node in self.nodes.values():
            left, right = shape_map.get(node.node_type, ("[", "]"))
            # Escape quotes in label
            safe_label = node.label.replace('"', "'")
            lines.append(f'    {node.id}{left}"{safe_label}"{right}')

        # Add edges
        arrow_map = {
            "uses": "-->",
            "contains": "-->",
            "transforms": "==>",
            "depends_on": "-.->",
            "implements": "-->",
            "extends": "-->",
        }

        for edge in self.edges:
            arrow = arrow_map.get(edge.relationship, "-->")
            if edge.label:
                lines.append(f'    {edge.source_id} {arrow}|"{edge.label}"| {edge.target_id}')
            else:
                lines.append(f'    {edge.source_id} {arrow} {edge.target_id}')

        # Add styling by node type
        style_map = {
            "component": "fill:#e1f5fe",
            "service": "fill:#fff3e0",
            "concept": "fill:#f3e5f5",
            "data": "fill:#e8f5e9",
            "interface": "fill:#fce4ec",
            "process": "fill:#fff8e1",
        }

        for node_type, nodes in node_types.items():
            if node_type in style_map:
                node_ids = ",".join(n.id for n in nodes)
                if node_ids:
                    lines.append(f'    style {node_ids} {style_map[node_type]}')

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize the mental model to JSON."""
        return json.dumps({
            "project_name": self.project_name,
            "version": self.version,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "nodes": {
                nid: {
                    "id": n.id,
                    "label": n.label,
                    "node_type": n.node_type,
                    "description": n.description,
                    "introduced_by_prompt_id": n.introduced_by_prompt_id,
                    "last_updated_prompt_id": n.last_updated_prompt_id,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relationship": e.relationship,
                    "label": e.label,
                    "introduced_by_prompt_id": e.introduced_by_prompt_id,
                }
                for e in self.edges
            ],
            "changelog": self.changelog,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MentalModel":
        """Deserialize a mental model from JSON."""
        data = json.loads(json_str)
        model = cls(
            project_name=data.get("project_name", ""),
            version=data.get("version", 0),
            changelog=data.get("changelog", []),
        )
        if data.get("last_updated"):
            model.last_updated = datetime.fromisoformat(data["last_updated"])

        for nid, ndata in data.get("nodes", {}).items():
            model.nodes[nid] = MentalModelNode(
                id=ndata["id"],
                label=ndata["label"],
                node_type=ndata["node_type"],
                description=ndata.get("description", ""),
                introduced_by_prompt_id=ndata.get("introduced_by_prompt_id"),
                last_updated_prompt_id=ndata.get("last_updated_prompt_id"),
            )

        for edata in data.get("edges", []):
            model.edges.append(MentalModelEdge(
                source_id=edata["source_id"],
                target_id=edata["target_id"],
                relationship=edata["relationship"],
                label=edata.get("label", ""),
                introduced_by_prompt_id=edata.get("introduced_by_prompt_id"),
            ))

        return model


@dataclass
class ModelUpdate:
    """Describes a proposed update to the mental model."""

    add_nodes: list[MentalModelNode] = field(default_factory=list)
    remove_node_ids: list[str] = field(default_factory=list)
    add_edges: list[MentalModelEdge] = field(default_factory=list)
    remove_edges: list[tuple[str, str, str]] = field(default_factory=list)  # (source, target, relationship)

    # Explanation of why this update was made
    reasoning: str = ""

    def is_empty(self) -> bool:
        return (not self.add_nodes and not self.remove_node_ids and
                not self.add_edges and not self.remove_edges)


def apply_update(model: MentalModel, update: ModelUpdate, prompt_id: Optional[str] = None) -> MentalModel:
    """Apply an update to the mental model."""

    # Remove nodes first
    for node_id in update.remove_node_ids:
        model.remove_node(node_id)

    # Remove edges
    for source_id, target_id, relationship in update.remove_edges:
        model.edges = [e for e in model.edges
                      if not (e.source_id == source_id and
                             e.target_id == target_id and
                             e.relationship == relationship)]

    # Add nodes
    for node in update.add_nodes:
        model.add_node(node, prompt_id)

    # Add edges
    for edge in update.add_edges:
        edge.introduced_by_prompt_id = prompt_id
        model.add_edge(edge)

    # Update metadata
    model.version += 1
    model.last_updated = datetime.now()

    if update.reasoning:
        model.changelog.append({
            "version": model.version,
            "prompt_id": prompt_id,
            "reasoning": update.reasoning,
            "timestamp": model.last_updated.isoformat(),
        })

    return model


def generate_update_prompt(
    current_model: MentalModel,
    operations: list[FileOperation],
) -> str:
    """
    Generate a prompt for an AI to analyze changes and propose mental model updates.

    This is the core of the "living mental model" concept - we're asking the AI
    to interpret code changes in terms of conceptual/architectural impact.
    """

    # Build context about what changed
    changes_summary = []
    for op in operations:
        change_desc = f"- {op.operation_type.value}: {op.file_path}"
        if op.prompt:
            change_desc += f"\n  Intent: {op.prompt.text[:200]}..."
        if op.assistant_context and op.assistant_context.summary:
            change_desc += f"\n  Reasoning: {op.assistant_context.summary[:200]}..."
        changes_summary.append(change_desc)

    current_diagram = current_model.to_mermaid() if current_model.nodes else "(empty - no existing model)"

    prompt = f"""You are analyzing code changes to update a "mental model" diagram.

The mental model represents the CONCEPTUAL structure of the system - not file paths or
class hierarchies, but the semantic components, capabilities, and how they relate.

## Current Mental Model
```mermaid
{current_diagram}
```

## Recent Code Changes
{chr(10).join(changes_summary)}

## Your Task
Analyze these changes and determine if/how the mental model should be updated.

Consider:
1. Are new concepts or components being introduced?
2. Are existing relationships changing?
3. Is there a shift in how parts of the system interact?
4. Should any concepts be removed or renamed?

Respond with a JSON object:
{{
    "reasoning": "Explain your interpretation of how these changes affect the system's conceptual structure",
    "updates": {{
        "add_nodes": [
            {{"id": "unique_id", "label": "Human Name", "node_type": "component|service|concept|data|interface|process", "description": "What this represents"}}
        ],
        "remove_node_ids": ["node_id_to_remove"],
        "add_edges": [
            {{"source_id": "from_node", "target_id": "to_node", "relationship": "uses|contains|transforms|depends_on", "label": "optional edge label"}}
        ],
        "remove_edges": [["source_id", "target_id", "relationship"]]
    }}
}}

If no updates are needed, return: {{"reasoning": "...", "updates": null}}
"""

    return prompt


def parse_update_response(response: str) -> Optional[ModelUpdate]:
    """Parse an AI response into a ModelUpdate."""
    import re

    # Try to extract JSON from the response
    # Look for JSON block or the whole response
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if data.get("updates") is None:
        return ModelUpdate(reasoning=data.get("reasoning", "No changes needed"))

    updates = data["updates"]

    update = ModelUpdate(reasoning=data.get("reasoning", ""))

    for node_data in updates.get("add_nodes", []):
        update.add_nodes.append(MentalModelNode(
            id=node_data["id"],
            label=node_data["label"],
            node_type=node_data.get("node_type", "component"),
            description=node_data.get("description", ""),
        ))

    update.remove_node_ids = updates.get("remove_node_ids", [])

    for edge_data in updates.get("add_edges", []):
        update.add_edges.append(MentalModelEdge(
            source_id=edge_data["source_id"],
            target_id=edge_data["target_id"],
            relationship=edge_data.get("relationship", "uses"),
            label=edge_data.get("label", ""),
        ))

    update.remove_edges = [tuple(e) for e in updates.get("remove_edges", [])]

    return update


# Example usage and testing
if __name__ == "__main__":
    # Create a simple mental model
    model = MentalModel(project_name="Example System")

    # Add some nodes
    model.add_node(MentalModelNode(
        id="api",
        label="REST API",
        node_type="interface",
        description="External API for client applications"
    ))

    model.add_node(MentalModelNode(
        id="processor",
        label="Data Processor",
        node_type="service",
        description="Transforms and validates incoming data"
    ))

    model.add_node(MentalModelNode(
        id="storage",
        label="Data Store",
        node_type="data",
        description="Persistent storage for processed data"
    ))

    # Add relationships
    model.add_edge(MentalModelEdge(
        source_id="api",
        target_id="processor",
        relationship="uses",
        label="sends requests"
    ))

    model.add_edge(MentalModelEdge(
        source_id="processor",
        target_id="storage",
        relationship="transforms",
        label="persists"
    ))

    print("=== Mental Model (Mermaid) ===")
    print(model.to_mermaid())
    print()
    print("=== Mental Model (JSON) ===")
    print(model.to_json())
