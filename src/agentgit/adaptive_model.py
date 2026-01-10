"""
Adaptive Mental Model - AI-determined structure with direct manipulation.

Key changes from previous approach:
1. NO prescribed node types or relationships - the AI decides what structure
   makes sense based on what it observes in the codebase
2. Selection + instruction interface - draw a box around any part of the
   diagram, describe what you want, and the model adapts

The mental model becomes a TWO-WAY interface:
- AI → Human: Shows what the AI thinks the system is
- Human → AI: You reshape the model, which guides future AI understanding
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime
import json
import copy


@dataclass
class ModelElement:
    """
    A generic element in the mental model.

    Unlike the previous approach with fixed types (component, service, etc.),
    elements are freeform. The AI decides what they represent and how to
    visualize them.
    """
    id: str
    label: str

    # Freeform properties - AI decides what's relevant
    properties: dict[str, Any] = field(default_factory=dict)

    # Visual hints (AI-suggested, can be overridden)
    shape: str = "box"  # box, circle, diamond, cylinder, etc.
    color: Optional[str] = None

    # Provenance
    created_by: str = ""  # "ai" or "human"
    reasoning: str = ""   # Why this element exists

    def __hash__(self):
        return hash(self.id)


@dataclass
class ModelRelation:
    """A relationship between elements - also freeform."""
    source_id: str
    target_id: str

    # AI decides what the relationship means
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    # Visual hints
    style: str = "solid"  # solid, dashed, dotted
    arrow: str = "normal"  # normal, none, both

    created_by: str = ""
    reasoning: str = ""


@dataclass
class SelectionInstruction:
    """
    An instruction from the user about a selected region of the diagram.

    This is the core of the "draw a box and describe" interaction.
    """
    selected_element_ids: list[str]
    instruction: str  # What the user wants to change
    timestamp: datetime = field(default_factory=datetime.now)

    # After AI processes this
    applied: bool = False
    ai_interpretation: str = ""


@dataclass
class AdaptiveModel:
    """
    A mental model where structure emerges from AI observation + human guidance.
    """
    elements: dict[str, ModelElement] = field(default_factory=dict)
    relations: list[ModelRelation] = field(default_factory=list)

    # History
    version: int = 0
    snapshots: list[dict] = field(default_factory=list)

    # Pending human instructions
    pending_instructions: list[SelectionInstruction] = field(default_factory=list)

    # The AI's current "understanding" - a freeform description
    ai_summary: str = ""

    def snapshot(self) -> None:
        """Save current state for time travel."""
        self.snapshots.append({
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "elements": {eid: self._element_to_dict(e) for eid, e in self.elements.items()},
            "relations": [self._relation_to_dict(r) for r in self.relations],
            "ai_summary": self.ai_summary,
        })

    def _element_to_dict(self, e: ModelElement) -> dict:
        return {
            "id": e.id, "label": e.label, "properties": e.properties,
            "shape": e.shape, "color": e.color,
            "created_by": e.created_by, "reasoning": e.reasoning,
        }

    def _relation_to_dict(self, r: ModelRelation) -> dict:
        return {
            "source_id": r.source_id, "target_id": r.target_id,
            "label": r.label, "properties": r.properties,
            "style": r.style, "arrow": r.arrow,
            "created_by": r.created_by, "reasoning": r.reasoning,
        }

    def get_version(self, version: int) -> Optional[dict]:
        """Get state at a specific version."""
        for snap in self.snapshots:
            if snap["version"] == version:
                return snap
        return None

    def to_mermaid(self) -> str:
        """Export as Mermaid - let the structure speak for itself."""
        lines = ["graph TD"]

        # Shape mapping
        shapes = {
            "box": ("[", "]"),
            "rounded": ("(", ")"),
            "circle": ("((", "))"),
            "diamond": ("{", "}"),
            "cylinder": ("[(", ")]"),
            "hexagon": ("{{", "}}"),
            "stadium": ("([", "])"),
        }

        for elem in self.elements.values():
            left, right = shapes.get(elem.shape, ("[", "]"))
            safe_label = elem.label.replace('"', "'")
            lines.append(f'    {elem.id}{left}"{safe_label}"{right}')

        # Relations
        arrow_styles = {
            ("solid", "normal"): "-->",
            ("solid", "none"): "---",
            ("solid", "both"): "<-->",
            ("dashed", "normal"): "-.->",
            ("dashed", "none"): "-.-",
            ("dotted", "normal"): "..>",
        }

        for rel in self.relations:
            arrow = arrow_styles.get((rel.style, rel.arrow), "-->")
            if rel.label:
                lines.append(f'    {rel.source_id} {arrow}|"{rel.label}"| {rel.target_id}')
            else:
                lines.append(f'    {rel.source_id} {arrow} {rel.target_id}')

        # Colors
        for elem in self.elements.values():
            if elem.color:
                lines.append(f'    style {elem.id} fill:{elem.color}')

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "elements": {eid: self._element_to_dict(e) for eid, e in self.elements.items()},
            "relations": [self._relation_to_dict(r) for r in self.relations],
            "version": self.version,
            "ai_summary": self.ai_summary,
            "snapshots": self.snapshots,
        }, indent=2)


def generate_observation_prompt(
    current_model: AdaptiveModel,
    context: dict,  # Freeform context about what changed
) -> str:
    """
    Generate a prompt for the AI to observe changes and decide how to update
    the mental model. No prescribed structure - AI decides.
    """

    current_diagram = current_model.to_mermaid() if current_model.elements else "(empty)"

    # Include any pending human instructions
    human_guidance = ""
    if current_model.pending_instructions:
        instructions = current_model.pending_instructions
        human_guidance = "\n\n## Human Guidance\nThe user has provided these instructions about the model:\n"
        for inst in instructions:
            selected = ", ".join(inst.selected_element_ids) if inst.selected_element_ids else "general"
            human_guidance += f"\n- Selection: [{selected}]\n  Instruction: {inst.instruction}\n"

    prompt = f"""You are observing a codebase and maintaining a "mental model" - a diagram that
captures how the system works conceptually.

## Current Mental Model
```mermaid
{current_diagram}
```

{f"Current AI understanding: {current_model.ai_summary}" if current_model.ai_summary else ""}
{human_guidance}

## Recent Changes
{json.dumps(context, indent=2)}

## Your Task

Decide how (if at all) to update the mental model. You have complete freedom in how you
structure it - there are no required categories or relationships. Choose whatever
representation best captures how this system works.

Consider:
- What are the key concepts/capabilities/flows?
- How do they relate to each other?
- What level of abstraction is most useful?
- Has the user provided guidance that should reshape your understanding?

Respond with JSON:
```json
{{
    "thinking": "Your reasoning about what this system is and how to represent it",
    "summary": "A 1-2 sentence description of what this system does (update your understanding)",
    "updates": {{
        "add_elements": [
            {{
                "id": "unique_id",
                "label": "Human readable name",
                "shape": "box|rounded|circle|diamond|cylinder|hexagon|stadium",
                "color": "#hex or null",
                "reasoning": "Why this element exists in the model"
            }}
        ],
        "remove_element_ids": ["id1", "id2"],
        "modify_elements": [
            {{"id": "existing_id", "label": "new label", "shape": "new_shape"}}
        ],
        "add_relations": [
            {{
                "source_id": "from",
                "target_id": "to",
                "label": "relationship description",
                "style": "solid|dashed|dotted",
                "reasoning": "Why this relationship matters"
            }}
        ],
        "remove_relations": [["source_id", "target_id"]]
    }}
}}
```

If no updates needed: {{"thinking": "...", "summary": "...", "updates": null}}
"""
    return prompt


def generate_instruction_prompt(
    model: AdaptiveModel,
    instruction: SelectionInstruction,
) -> str:
    """
    Generate a prompt to process a human instruction about a selected region.
    """

    # Get the selected elements
    selected_elements = [model.elements[eid] for eid in instruction.selected_element_ids
                         if eid in model.elements]

    # Get relations involving selected elements
    selected_ids = set(instruction.selected_element_ids)
    related_relations = [r for r in model.relations
                        if r.source_id in selected_ids or r.target_id in selected_ids]

    selection_desc = ""
    if selected_elements:
        selection_desc = "Selected elements:\n"
        for elem in selected_elements:
            selection_desc += f"- {elem.label} ({elem.id}): {elem.reasoning}\n"
        if related_relations:
            selection_desc += "\nRelated connections:\n"
            for rel in related_relations:
                selection_desc += f"- {rel.source_id} --{rel.label}--> {rel.target_id}\n"
    else:
        selection_desc = "(No specific elements selected - instruction applies to whole model)"

    prompt = f"""The user has selected part of the mental model and provided an instruction.

## Current Model
```mermaid
{model.to_mermaid()}
```

## Selection
{selection_desc}

## User Instruction
"{instruction.instruction}"

## Your Task

Interpret the user's instruction and determine how to update the model. The user is
telling you how they think about this part of the system - incorporate their perspective.

Respond with JSON:
```json
{{
    "interpretation": "How you understand the user's instruction",
    "updates": {{
        "add_elements": [...],
        "remove_element_ids": [...],
        "modify_elements": [...],
        "add_relations": [...],
        "remove_relations": [...]
    }}
}}
```
"""
    return prompt


class AdaptiveModelController:
    """
    Controller that mediates between the model, AI, and human.
    """

    def __init__(
        self,
        ai_callback: Callable[[str], str],
        model: Optional[AdaptiveModel] = None,
    ):
        self.model = model or AdaptiveModel()
        self.ai_callback = ai_callback

    def observe(self, context: dict) -> dict:
        """
        Have the AI observe changes and update the model.
        Returns the AI's response.
        """
        self.model.snapshot()

        prompt = generate_observation_prompt(self.model, context)
        response = self.ai_callback(prompt)

        # Parse and apply
        result = self._parse_response(response)
        if result and result.get("updates"):
            self._apply_updates(result["updates"], created_by="ai")

        if result and result.get("summary"):
            self.model.ai_summary = result["summary"]

        # Clear processed instructions
        self.model.pending_instructions = []
        self.model.version += 1

        return result or {}

    def instruct(self, selected_ids: list[str], instruction: str) -> dict:
        """
        Process a human instruction about selected elements.
        This is the "draw a box and describe" interaction.
        """
        inst = SelectionInstruction(
            selected_element_ids=selected_ids,
            instruction=instruction,
        )

        self.model.snapshot()

        prompt = generate_instruction_prompt(self.model, inst)
        response = self.ai_callback(prompt)

        result = self._parse_response(response)
        if result and result.get("updates"):
            self._apply_updates(result["updates"], created_by="human")

        inst.applied = True
        inst.ai_interpretation = result.get("interpretation", "") if result else ""

        self.model.version += 1

        return result or {}

    def queue_instruction(self, selected_ids: list[str], instruction: str) -> None:
        """
        Queue a human instruction to be considered on next observation.
        Use this when you want to guide future AI updates without immediately applying.
        """
        self.model.pending_instructions.append(SelectionInstruction(
            selected_element_ids=selected_ids,
            instruction=instruction,
        ))

    def _parse_response(self, response: str) -> Optional[dict]:
        """Extract JSON from AI response."""
        import re

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def _apply_updates(self, updates: dict, created_by: str) -> None:
        """Apply updates to the model."""

        # Remove elements first
        for eid in updates.get("remove_element_ids", []):
            if eid in self.model.elements:
                del self.model.elements[eid]
                # Also remove related relations
                self.model.relations = [
                    r for r in self.model.relations
                    if r.source_id != eid and r.target_id != eid
                ]

        # Remove relations
        for source_id, target_id in updates.get("remove_relations", []):
            self.model.relations = [
                r for r in self.model.relations
                if not (r.source_id == source_id and r.target_id == target_id)
            ]

        # Add elements
        for elem_data in updates.get("add_elements", []):
            elem = ModelElement(
                id=elem_data["id"],
                label=elem_data["label"],
                shape=elem_data.get("shape", "box"),
                color=elem_data.get("color"),
                properties=elem_data.get("properties", {}),
                created_by=created_by,
                reasoning=elem_data.get("reasoning", ""),
            )
            self.model.elements[elem.id] = elem

        # Modify elements
        for mod in updates.get("modify_elements", []):
            if mod["id"] in self.model.elements:
                elem = self.model.elements[mod["id"]]
                if "label" in mod:
                    elem.label = mod["label"]
                if "shape" in mod:
                    elem.shape = mod["shape"]
                if "color" in mod:
                    elem.color = mod["color"]
                if "properties" in mod:
                    elem.properties.update(mod["properties"])

        # Add relations
        for rel_data in updates.get("add_relations", []):
            rel = ModelRelation(
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                label=rel_data.get("label", ""),
                style=rel_data.get("style", "solid"),
                arrow=rel_data.get("arrow", "normal"),
                properties=rel_data.get("properties", {}),
                created_by=created_by,
                reasoning=rel_data.get("reasoning", ""),
            )
            self.model.relations.append(rel)


# Demo with simulated AI
if __name__ == "__main__":
    # Simulated AI that responds to different contexts
    def mock_ai(prompt: str) -> str:
        if "e-commerce" in prompt.lower() or "product" in prompt.lower():
            return """```json
{
    "thinking": "This looks like an e-commerce system. I see product-related code and what appears to be order processing. Rather than using generic 'component' types, I'll model this as a flow: browsing -> cart -> checkout.",
    "summary": "An e-commerce platform where users browse products, add to cart, and checkout",
    "updates": {
        "add_elements": [
            {"id": "browse", "label": "Browse & Search", "shape": "stadium", "color": "#e3f2fd", "reasoning": "Entry point - users discovering products"},
            {"id": "cart", "label": "Shopping Cart", "shape": "rounded", "color": "#fff3e0", "reasoning": "Accumulation state before purchase"},
            {"id": "checkout", "label": "Purchase Flow", "shape": "hexagon", "color": "#e8f5e9", "reasoning": "Critical path - where money changes hands"}
        ],
        "add_relations": [
            {"source_id": "browse", "target_id": "cart", "label": "add items", "style": "solid"},
            {"source_id": "cart", "target_id": "checkout", "label": "proceed", "style": "solid"}
        ]
    }
}
```"""
        elif "actually three separate" in prompt.lower() or "split" in prompt.lower():
            # Response to human instruction to split something
            return """```json
{
    "interpretation": "The user sees the checkout as three distinct phases that should be modeled separately",
    "updates": {
        "remove_element_ids": ["checkout"],
        "add_elements": [
            {"id": "shipping", "label": "Shipping Info", "shape": "rounded", "color": "#e8f5e9", "reasoning": "User specified: first phase of checkout"},
            {"id": "payment", "label": "Payment", "shape": "hexagon", "color": "#ffebee", "reasoning": "User specified: critical payment phase"},
            {"id": "confirm", "label": "Confirmation", "shape": "rounded", "color": "#e8f5e9", "reasoning": "User specified: final confirmation"}
        ],
        "add_relations": [
            {"source_id": "cart", "target_id": "shipping", "label": "begin checkout", "style": "solid"},
            {"source_id": "shipping", "target_id": "payment", "label": "next", "style": "solid"},
            {"source_id": "payment", "target_id": "confirm", "label": "complete", "style": "solid"}
        ]
    }
}
```"""
        else:
            return """```json
{"thinking": "No significant changes needed", "summary": "System unchanged", "updates": null}
```"""

    # Create controller
    controller = AdaptiveModelController(ai_callback=mock_ai)

    print("=== Initial Observation ===")
    print("AI observes: 'e-commerce codebase with products and orders'\n")

    result = controller.observe({
        "files_changed": ["src/products/catalog.py", "src/orders/checkout.py"],
        "intent": "Building an e-commerce platform",
    })

    print(f"AI thinking: {result.get('thinking', '')[:100]}...")
    print(f"AI summary: {result.get('summary', '')}")
    print(f"\nDiagram:\n{controller.model.to_mermaid()}")

    print("\n" + "="*50)
    print("=== Human Instruction ===")
    print("User selects 'checkout' and says: 'This is actually three separate steps'")
    print()

    result = controller.instruct(
        selected_ids=["checkout"],
        instruction="This is actually three separate steps: shipping info, payment, and confirmation. Split it up."
    )

    print(f"AI interpretation: {result.get('interpretation', '')}")
    print(f"\nUpdated diagram:\n{controller.model.to_mermaid()}")

    print("\n" + "="*50)
    print("=== Version History ===")
    for i, snap in enumerate(controller.model.snapshots):
        print(f"v{snap['version']}: {len(snap['elements'])} elements - {snap['ai_summary'][:50]}...")
