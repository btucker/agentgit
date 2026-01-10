"""Reframe - A living mental model that evolves with your codebase.

Reframe maintains a semantic visualization of your system that automatically
adapts as code changes. It's not a file tree or class diagram - it's how
you and an AI collaboratively understand what the system *does*.

Core concepts:
- AI observes code changes and proposes model updates
- Human can "draw a box and describe" to reshape understanding
- Insights accumulate over time, creating shared context
- Time travel through how understanding evolved

The model structure is NOT prescribed - the AI decides what abstraction
level and relationships best capture the system.

Name origin: "Reframe" - to see something through a different frame/lens.
From cognitive psychology, where reframing means shifting your mental model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agentgit.plugins import hookimpl, hookspec

if TYPE_CHECKING:
    from agentgit.core import AssistantTurn, FileOperation, Prompt, PromptResponse

logger = logging.getLogger(__name__)

ENHANCER_NAME = "mental_model"


# ============================================================================
# Data Structures (freeform - AI decides structure)
# ============================================================================


@dataclass
class ModelElement:
    """A freeform element in the mental model. AI decides what it represents."""

    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    shape: str = "box"
    color: str | None = None
    created_by: str = ""  # "ai" or "human"
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "properties": self.properties,
            "shape": self.shape,
            "color": self.color,
            "created_by": self.created_by,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelElement":
        return cls(
            id=data["id"],
            label=data["label"],
            properties=data.get("properties", {}),
            shape=data.get("shape", "box"),
            color=data.get("color"),
            created_by=data.get("created_by", ""),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class ModelRelation:
    """A freeform relationship. AI decides what it means."""

    source_id: str
    target_id: str
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    style: str = "solid"
    created_by: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "properties": self.properties,
            "style": self.style,
            "created_by": self.created_by,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelRelation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            label=data.get("label", ""),
            properties=data.get("properties", {}),
            style=data.get("style", "solid"),
            created_by=data.get("created_by", ""),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class SequenceStep:
    """A step in a sequence diagram."""

    from_participant: str
    to_participant: str
    message: str
    step_type: str = "sync"  # sync, async, reply, note
    note: str | None = None


@dataclass
class SequenceDiagram:
    """A sequence diagram showing interactions over time.

    Useful for showing:
    - Request/response flows
    - Event propagation
    - User journeys through the system
    """

    id: str
    title: str
    participants: list[str] = field(default_factory=list)
    steps: list[SequenceStep] = field(default_factory=list)
    description: str = ""
    created_by: str = ""
    trigger: str = ""  # What flow this represents (e.g., "user login")

    def to_mermaid(self) -> str:
        """Export as Mermaid sequence diagram."""
        lines = ["sequenceDiagram"]

        if self.title:
            lines.append(f"    title {self.title}")

        # Declare participants in order
        for p in self.participants:
            # Sanitize participant names for mermaid
            safe_p = p.replace(" ", "_")
            if p != safe_p:
                lines.append(f"    participant {safe_p} as {p}")
            else:
                lines.append(f"    participant {p}")

        # Add steps
        for step in self.steps:
            arrow = {
                "sync": "->>",
                "async": "-->>",
                "reply": "-->>",
                "create": "->>+",
                "destroy": "->>-",
            }.get(step.step_type, "->>")

            from_p = step.from_participant.replace(" ", "_")
            to_p = step.to_participant.replace(" ", "_")
            lines.append(f"    {from_p}{arrow}{to_p}: {step.message}")

            if step.note:
                lines.append(f"    Note over {from_p},{to_p}: {step.note}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "participants": self.participants,
            "steps": [
                {
                    "from": s.from_participant,
                    "to": s.to_participant,
                    "message": s.message,
                    "type": s.step_type,
                    "note": s.note,
                }
                for s in self.steps
            ],
            "description": self.description,
            "created_by": self.created_by,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceDiagram":
        diagram = cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            participants=data.get("participants", []),
            description=data.get("description", ""),
            created_by=data.get("created_by", ""),
            trigger=data.get("trigger", ""),
        )
        for step_data in data.get("steps", []):
            diagram.steps.append(SequenceStep(
                from_participant=step_data["from"],
                to_participant=step_data["to"],
                message=step_data["message"],
                step_type=step_data.get("type", "sync"),
                note=step_data.get("note"),
            ))
        return diagram


@dataclass
class ModelSnapshot:
    """A point-in-time snapshot for time travel."""

    version: int
    timestamp: datetime
    elements: dict[str, ModelElement]
    relations: list[ModelRelation]
    ai_summary: str
    trigger: str  # What caused this snapshot


@dataclass
class MentalModel:
    """The living mental model - structure emerges from AI observation.

    Contains two types of diagrams:
    1. Component diagram (elements + relations) - the "what"
    2. Sequence diagrams - the "how" (flows, interactions)
    """

    # Component diagram
    elements: dict[str, ModelElement] = field(default_factory=dict)
    relations: list[ModelRelation] = field(default_factory=list)

    # Sequence diagrams for showing flows
    sequences: dict[str, SequenceDiagram] = field(default_factory=dict)

    version: int = 0
    ai_summary: str = ""
    snapshots: list[ModelSnapshot] = field(default_factory=list)

    # Files associated with each element (for focused prompting)
    element_files: dict[str, set[str]] = field(default_factory=dict)

    def snapshot(self, trigger: str = "") -> None:
        """Save current state for time travel."""
        import copy

        self.snapshots.append(
            ModelSnapshot(
                version=self.version,
                timestamp=datetime.now(),
                elements=copy.deepcopy(self.elements),
                relations=copy.deepcopy(self.relations),
                ai_summary=self.ai_summary,
                trigger=trigger,
            )
        )

    def to_mermaid(self) -> str:
        """Export as Mermaid diagram."""
        lines = ["graph TD"]

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

        arrow_styles = {
            "solid": "-->",
            "dashed": "-.->",
            "dotted": "..>",
            "thick": "==>",
        }

        for rel in self.relations:
            arrow = arrow_styles.get(rel.style, "-->")
            if rel.label:
                lines.append(f'    {rel.source_id} {arrow}|"{rel.label}"| {rel.target_id}')
            else:
                lines.append(f'    {rel.source_id} {arrow} {rel.target_id}')

        for elem in self.elements.values():
            if elem.color:
                lines.append(f"    style {elem.id} fill:{elem.color}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "elements": {eid: e.to_dict() for eid, e in self.elements.items()},
            "relations": [r.to_dict() for r in self.relations],
            "sequences": {sid: s.to_dict() for sid, s in self.sequences.items()},
            "version": self.version,
            "ai_summary": self.ai_summary,
            "element_files": {k: list(v) for k, v in self.element_files.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_full_mermaid(self) -> str:
        """Export both component diagram and sequence diagrams."""
        parts = ["## Component Diagram\n", "```mermaid", self.to_mermaid(), "```"]

        for seq_id, seq in self.sequences.items():
            parts.append(f"\n## {seq.title or seq_id}\n")
            if seq.description:
                parts.append(f"{seq.description}\n")
            parts.append("```mermaid")
            parts.append(seq.to_mermaid())
            parts.append("```")

        return "\n".join(parts)

    @classmethod
    def from_dict(cls, data: dict) -> "MentalModel":
        model = cls(
            version=data.get("version", 0),
            ai_summary=data.get("ai_summary", ""),
        )
        for eid, edata in data.get("elements", {}).items():
            model.elements[eid] = ModelElement.from_dict(edata)
        for rdata in data.get("relations", []):
            model.relations.append(ModelRelation.from_dict(rdata))
        for sid, sdata in data.get("sequences", {}).items():
            model.sequences[sid] = SequenceDiagram.from_dict(sdata)
        for eid, files in data.get("element_files", {}).items():
            model.element_files[eid] = set(files)
        return model


# ============================================================================
# LLM Integration (uses same pattern as llm.py)
# ============================================================================


def _get_llm_model(model: str = "claude-cli-haiku"):
    """Get LLM model instance."""
    try:
        import llm

        return llm.get_model(model)
    except ImportError:
        logger.warning("llm not installed")
        return None
    except Exception as e:
        logger.warning("Failed to get model: %s", e)
        return None


def _run_llm(prompt: str, model: str = "claude-cli-haiku") -> str | None:
    """Run prompt through LLM."""
    llm_model = _get_llm_model(model)
    if not llm_model:
        return None
    try:
        return llm_model.prompt(prompt).text()
    except Exception as e:
        logger.warning("LLM request failed: %s", e)
        return None


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ============================================================================
# AI Prompts (AI decides structure - no prescribed types)
# ============================================================================


def build_observation_prompt(
    model: MentalModel,
    context: dict,
    accumulated_insights: str = "",
) -> str:
    """Build prompt for AI to observe changes and update model.

    The AI receives:
    - Current model structure (JSON/Mermaid)
    - Code changes (files, intents, reasoning)
    - Accumulated insights (previous AI + human observations)

    And outputs:
    - Updated model structure
    - New insight to append to the insights document
    """

    current_diagram = model.to_mermaid() if model.elements else "(empty)"

    insights_section = ""
    if accumulated_insights:
        # Truncate if too long, keeping most recent
        if len(accumulated_insights) > 3000:
            accumulated_insights = "...(earlier insights truncated)...\n" + accumulated_insights[-3000:]
        insights_section = f"""
## Accumulated Insights
These are previous observations and refinements from both AI and human:

{accumulated_insights}
"""

    return f"""You are maintaining a "mental model" of a codebase - a conceptual diagram
of how the system works, not its file structure.

## Current Mental Model
```mermaid
{current_diagram}
```

{f"Current understanding: {model.ai_summary}" if model.ai_summary else ""}
{insights_section}

## Recent Code Changes
{json.dumps(context, indent=2)}

## Your Task

1. **Review** the accumulated insights - they contain important context from
   previous observations and human refinements.

2. **Analyze** how these code changes affect the conceptual structure.

3. **Update** the model if needed. You have complete freedom in structure:
   - Choose whatever elements/relationships capture the system best
   - Pick shapes (box, rounded, circle, diamond, cylinder, hexagon, stadium)
   - Use colors to indicate groupings or importance
   - Let the right abstraction level emerge from what you observe

4. **Document** your insight - what did you learn about the system?
   This will be added to the insights document for future reference.

Respond with JSON:
```json
{{
    "insight": "What you learned or observed about the system (1-3 sentences). This gets appended to the insights document.",
    "summary": "1-2 sentence description of what this system does (updated if needed)",
    "updates": {{
        "add_elements": [
            {{"id": "unique_id", "label": "Name", "shape": "box", "color": "#hex", "reasoning": "why"}}
        ],
        "remove_element_ids": ["id"],
        "modify_elements": [{{"id": "x", "label": "new", "shape": "new"}}],
        "add_relations": [
            {{"source_id": "a", "target_id": "b", "label": "relationship", "style": "solid", "reasoning": "why"}}
        ],
        "remove_relations": [["source_id", "target_id"]]
    }}
}}
```

If no structural changes needed, still provide an insight:
{{"insight": "...", "summary": "...", "updates": null}}
"""


def build_instruction_prompt(model: MentalModel, selected_ids: list[str], instruction: str) -> str:
    """Build prompt for AI to process human instruction about selected region."""

    selected_elements = [model.elements[eid] for eid in selected_ids if eid in model.elements]

    selection_desc = ""
    if selected_elements:
        selection_desc = "Selected elements:\n"
        for elem in selected_elements:
            selection_desc += f"- {elem.label} ({elem.id}): {elem.reasoning}\n"
    else:
        selection_desc = "(No specific elements - instruction applies to whole model)"

    return f"""The user has selected part of the mental model and given an instruction.

## Current Model
```mermaid
{model.to_mermaid()}
```

## Selection
{selection_desc}

## User Instruction
"{instruction}"

Interpret this instruction and update the model accordingly.
The user is reshaping how they think about this part of the system.

Respond with JSON:
```json
{{
    "interpretation": "How you understand the instruction",
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


def build_focused_prompt(model: MentalModel, element_id: str, intent: str) -> str:
    """Generate a focused prompt for interacting with a specific element."""

    elem = model.elements.get(element_id)
    if not elem:
        return f"Element {element_id} not found in model."

    # Find connected elements
    connected = []
    for rel in model.relations:
        if rel.source_id == element_id and rel.target_id in model.elements:
            connected.append(f"{model.elements[rel.target_id].label} ({rel.label})")
        elif rel.target_id == element_id and rel.source_id in model.elements:
            connected.append(f"{model.elements[rel.source_id].label} ({rel.label})")

    files = list(model.element_files.get(element_id, []))

    context = f"**{elem.label}**"
    if elem.reasoning:
        context += f"\n{elem.reasoning}"
    if connected:
        context += f"\nConnected to: {', '.join(connected)}"
    if files:
        context += f"\nRelated files: {', '.join(files[:5])}"

    prompts = {
        "explore": f"Tell me about the {elem.label}. What does it do and how does it fit in?",
        "modify": f"I want to change the {elem.label}. What files should I look at?",
        "debug": f"I'm having issues with {elem.label}. Help me understand how it works.",
        "test": f"I want to add tests for {elem.label}. What should I test?",
        "refactor": f"I'm considering refactoring {elem.label}. What would be the impact?",
    }

    prompt = prompts.get(intent, prompts["explore"])
    if files:
        prompt += f"\n\nRelevant files: {', '.join(files[:3])}"

    return prompt


# ============================================================================
# Enhancer Plugin
# ============================================================================


class MentalModelEnhancer:
    """Enhancer that maintains a living mental model of the codebase.

    The mental model is stored in the agentgit output repository at:
    .agentgit/mental_model.json

    This keeps it alongside the git history, making it part of the
    artifact that agentgit produces.
    """

    # Standard location within agentgit repos
    MODEL_FILENAME = ".agentgit/mental_model.json"

    def __init__(self, repo_path: Path | None = None):
        """Initialize the enhancer.

        Args:
            repo_path: Path to the agentgit output repo. If provided,
                      loads existing model from .agentgit/mental_model.json
        """
        self.model = MentalModel()
        self._repo_path: Path | None = None

        if repo_path:
            self.set_repo_path(repo_path)

    def set_repo_path(self, repo_path: Path) -> None:
        """Set the agentgit repo path and load existing model if present."""
        self._repo_path = repo_path
        model_path = repo_path / self.MODEL_FILENAME
        if model_path.exists():
            self.model = MentalModel.from_dict(json.loads(model_path.read_text()))

    @property
    def model_path(self) -> Path | None:
        """Get the full path to the mental model file."""
        if self._repo_path:
            return self._repo_path / self.MODEL_FILENAME
        return None

    def save(self) -> None:
        """Save model to the agentgit repo."""
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_path.write_text(self.model.to_json())

    def save_insights(self, new_insight: str | None = None) -> Path | None:
        """Append insights to the collaborative insights document.

        The insights file is a living document where both AI and human
        contribute understanding. On each update, the AI reads this file
        along with the JSON and code diff to inform its interpretation.

        Args:
            new_insight: New insight to append (from AI or human).

        Returns:
            Path to the insights file, or None if no repo configured.
        """
        if not self._repo_path:
            return None

        insights_path = self._repo_path / ".agentgit" / "mental_model.md"
        insights_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file if it doesn't exist
        if not insights_path.exists():
            initial_content = """# Reframe Insights

This document captures the evolving understanding of this system's architecture.
Both AI observations and human refinements accumulate here, creating a shared
mental model that improves over time.

Edit this file directly to add your own insights - they'll be read by the AI
on the next update to inform its understanding.

---

"""
            insights_path.write_text(initial_content)

        # Append new insight if provided
        if new_insight:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(insights_path, "a") as f:
                f.write(f"\n## {timestamp}\n\n{new_insight}\n")

        return insights_path

    def load_insights(self) -> str:
        """Load the accumulated insights for context.

        Returns:
            The contents of the insights file, or empty string if not found.
        """
        if not self._repo_path:
            return ""

        insights_path = self._repo_path / ".agentgit" / "mental_model.md"
        if insights_path.exists():
            return insights_path.read_text()
        return ""

    def add_human_insight(self, insight: str) -> None:
        """Add a human-provided insight to the document.

        Use this when the human wants to correct or refine the model's
        understanding without making structural changes.
        """
        self.save_insights(f"**Human insight:**\n\n{insight}")

    @hookimpl
    def agentgit_get_enhancer_info(self) -> dict[str, str]:
        return {
            "name": ENHANCER_NAME,
            "description": "Maintain a living mental model diagram of the codebase",
        }

    def observe_changes(
        self,
        prompt_responses: list["PromptResponse"],
        model: str = "claude-cli-haiku",
    ) -> dict | None:
        """
        Observe code changes and update the mental model.

        This is the main entry point - call after processing a transcript.

        The AI receives:
        - Current model structure
        - Code changes (files, intents, reasoning)
        - Accumulated insights from previous observations + human input

        The AI outputs:
        - Updated model structure
        - New insight to append to the insights document
        """
        # Build context from changes
        context = {"prompts": [], "files_changed": [], "reasoning": []}

        for pr in prompt_responses[:5]:  # Look at recent prompts
            context["prompts"].append(_truncate(pr.prompt.text, 200))
            for turn in pr.turns:
                for op in turn.operations:
                    context["files_changed"].append(op.file_path)
                if turn.context and turn.context.summary:
                    context["reasoning"].append(_truncate(turn.context.summary, 200))

        context["files_changed"] = list(set(context["files_changed"]))[:20]

        # Load accumulated insights for context
        accumulated_insights = self.load_insights()

        # Snapshot before update
        self.model.snapshot(trigger="observation")

        # Ask AI to interpret changes (with insights as context)
        prompt = build_observation_prompt(self.model, context, accumulated_insights)
        response = _run_llm(prompt, model)

        if not response:
            return None

        result = self._parse_response(response)

        # Apply structural updates
        if result and result.get("updates"):
            self._apply_updates(result["updates"], "ai", context["files_changed"])

        if result and result.get("summary"):
            self.model.ai_summary = result["summary"]

        # Save new insight to the insights document
        if result and result.get("insight"):
            self.save_insights(f"**AI observation:**\n\n{result['insight']}")

        self.model.version += 1
        self.save()

        return result

    def process_instruction(
        self,
        selected_ids: list[str],
        instruction: str,
        model: str = "claude-cli-haiku",
    ) -> dict | None:
        """
        Process a human instruction about selected elements.

        This is the "draw a box and describe" interaction.
        """
        self.model.snapshot(trigger=f"instruction: {instruction[:50]}")

        prompt = build_instruction_prompt(self.model, selected_ids, instruction)
        response = _run_llm(prompt, model)

        if not response:
            return None

        result = self._parse_response(response)
        if result and result.get("updates"):
            self._apply_updates(result["updates"], "human", [])

        self.model.version += 1
        self.save()

        return result

    def generate_focused_prompt(self, element_id: str, intent: str = "explore") -> str:
        """Generate a focused prompt for a specific element."""
        return build_focused_prompt(self.model, element_id, intent)

    def get_timeline(self) -> list[dict]:
        """Get version history for time travel UI."""
        timeline = []
        for snap in self.model.snapshots:
            timeline.append({
                "version": snap.version,
                "timestamp": snap.timestamp.isoformat(),
                "trigger": snap.trigger,
                "element_count": len(snap.elements),
                "summary": snap.ai_summary[:50] if snap.ai_summary else "",
            })
        timeline.append({
            "version": self.model.version,
            "timestamp": datetime.now().isoformat(),
            "trigger": "current",
            "element_count": len(self.model.elements),
            "summary": self.model.ai_summary[:50] if self.model.ai_summary else "",
        })
        return timeline

    def get_version(self, version: int) -> MentalModel | None:
        """Get model state at a specific version."""
        import copy

        for snap in self.model.snapshots:
            if snap.version == version:
                model = MentalModel(
                    version=snap.version,
                    ai_summary=snap.ai_summary,
                )
                model.elements = copy.deepcopy(snap.elements)
                model.relations = copy.deepcopy(snap.relations)
                return model
        if version == self.model.version:
            return copy.deepcopy(self.model)
        return None

    def _parse_response(self, response: str) -> dict | None:
        """Parse JSON from AI response."""
        import re

        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def _apply_updates(self, updates: dict, created_by: str, files: list[str]) -> None:
        """Apply updates to the model."""

        # Remove elements
        for eid in updates.get("remove_element_ids", []):
            if eid in self.model.elements:
                del self.model.elements[eid]
                self.model.relations = [
                    r
                    for r in self.model.relations
                    if r.source_id != eid and r.target_id != eid
                ]

        # Remove relations
        for source_id, target_id in updates.get("remove_relations", []):
            self.model.relations = [
                r
                for r in self.model.relations
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

            # Associate files with new elements
            if files:
                if elem.id not in self.model.element_files:
                    self.model.element_files[elem.id] = set()
                self.model.element_files[elem.id].update(files)

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

        # Add relations
        for rel_data in updates.get("add_relations", []):
            rel = ModelRelation(
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                label=rel_data.get("label", ""),
                style=rel_data.get("style", "solid"),
                properties=rel_data.get("properties", {}),
                created_by=created_by,
                reasoning=rel_data.get("reasoning", ""),
            )
            self.model.relations.append(rel)


# Global instance (for integration with agentgit flow)
_enhancer_instance: MentalModelEnhancer | None = None


def get_mental_model_enhancer(repo_path: Path | None = None) -> MentalModelEnhancer:
    """Get the mental model enhancer, optionally bound to a repo.

    Args:
        repo_path: Path to agentgit output repo. If provided and different
                  from current instance, creates new instance bound to that repo.

    Returns:
        MentalModelEnhancer instance.
    """
    global _enhancer_instance

    if repo_path:
        # Create new instance for this repo
        if _enhancer_instance is None or _enhancer_instance._repo_path != repo_path:
            _enhancer_instance = MentalModelEnhancer(repo_path)
    elif _enhancer_instance is None:
        _enhancer_instance = MentalModelEnhancer()

    return _enhancer_instance


def reframe(
    repo_path: Path,
    prompt_responses: list["PromptResponse"],
    model: str = "claude-cli-haiku",
    verbose: bool = False,
) -> MentalModel | None:
    """Reframe the mental model based on recent code changes.

    This is the main entry point - call after processing a transcript
    to evolve the mental model.

    The result is stored in the repo at:
    - .agentgit/mental_model.json  (structural data)
    - .agentgit/mental_model.md    (accumulated insights)

    Args:
        repo_path: Path to the agentgit output repo.
        prompt_responses: The prompt responses that were processed.
        model: LLM model to use for interpretation.
        verbose: If True, print progress.

    Returns:
        The updated MentalModel, or None if update failed.
    """
    enhancer = get_mental_model_enhancer(repo_path)

    if verbose:
        print("Reframing...", flush=True)

    result = enhancer.observe_changes(prompt_responses, model)
    if result:
        # Note: observe_changes already saves the model and insights
        if verbose:
            print(f"  Model: v{enhancer.model.version} ({len(enhancer.model.elements)} elements)")
            if result.get("insight"):
                print(f"  Insight: {result['insight'][:60]}...")

        logger.info(
            "Reframed: v%d with %d elements",
            enhancer.model.version,
            len(enhancer.model.elements),
        )
        return enhancer.model

    return None


# Alias for backwards compatibility
update_mental_model_after_build = reframe


def load_mental_model(repo_path: Path) -> MentalModel | None:
    """Load a mental model from an agentgit repo.

    Args:
        repo_path: Path to the agentgit output repo.

    Returns:
        The MentalModel if found, None otherwise.
    """
    model_path = repo_path / MentalModelEnhancer.MODEL_FILENAME
    if model_path.exists():
        return MentalModel.from_dict(json.loads(model_path.read_text()))
    return None


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    # Demo: Direct manipulation of the model (no LLM needed)
    print("=== Reframe Demo ===\n")

    model = MentalModel()

    # Simulate what AI would return for an e-commerce observation
    print("Step 1: AI observes e-commerce codebase")
    model.snapshot("initial observation")

    model.elements["browse"] = ModelElement(
        id="browse",
        label="Product Discovery",
        shape="stadium",
        color="#e3f2fd",
        reasoning="How users find products",
        created_by="ai",
    )
    model.elements["cart"] = ModelElement(
        id="cart",
        label="Cart",
        shape="rounded",
        color="#fff3e0",
        reasoning="Accumulates items before purchase",
        created_by="ai",
    )
    model.elements["checkout"] = ModelElement(
        id="checkout",
        label="Purchase",
        shape="hexagon",
        color="#e8f5e9",
        reasoning="Where transactions happen",
        created_by="ai",
    )
    model.relations.append(
        ModelRelation(source_id="browse", target_id="cart", label="add to")
    )
    model.relations.append(
        ModelRelation(source_id="cart", target_id="checkout", label="proceed")
    )
    model.version = 1
    model.ai_summary = "E-commerce platform for browsing and buying products"

    print(f"AI summary: {model.ai_summary}")
    print(f"\n{model.to_mermaid()}")

    # Simulate human instruction: split checkout
    print("\n" + "=" * 50)
    print("Step 2: Human draws box around 'checkout' and says:")
    print('         "This is actually three steps: shipping, payment, confirmation"')

    model.snapshot("human instruction")

    # Remove checkout
    del model.elements["checkout"]
    model.relations = [r for r in model.relations if r.target_id != "checkout"]

    # Add the three steps
    model.elements["shipping"] = ModelElement(
        id="shipping",
        label="Shipping",
        shape="rounded",
        color="#e8f5e9",
        created_by="human",
    )
    model.elements["payment"] = ModelElement(
        id="payment",
        label="Payment",
        shape="hexagon",
        color="#ffebee",
        created_by="human",
    )
    model.elements["confirm"] = ModelElement(
        id="confirm",
        label="Confirmation",
        shape="rounded",
        color="#e8f5e9",
        created_by="human",
    )
    model.relations.append(
        ModelRelation(source_id="cart", target_id="shipping", label="begin")
    )
    model.relations.append(
        ModelRelation(source_id="shipping", target_id="payment", label="next")
    )
    model.relations.append(
        ModelRelation(source_id="payment", target_id="confirm", label="complete")
    )
    model.version = 2

    print(f"\n{model.to_mermaid()}")

    # Show timeline
    print("\n" + "=" * 50)
    print("Timeline (for time travel):")
    model.snapshot("current")
    for snap in model.snapshots:
        print(f"  v{snap.version}: {snap.trigger} ({len(snap.elements)} elements)")

    # Show focused prompt
    print("\n" + "=" * 50)
    print("Focused prompt for 'payment' element:")
    print(build_focused_prompt(model, "payment", "debug"))
