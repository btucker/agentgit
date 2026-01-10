"""Tests for Reframe (mental model enhancer)."""

import json
import pytest
from datetime import datetime
from pathlib import Path

from agentgit.enhancers.mental_model import (
    MentalModel,
    ModelElement,
    ModelRelation,
    ModelSnapshot,
    SequenceStep,
    SequenceDiagram,
    MentalModelEnhancer,
    build_observation_prompt,
    build_instruction_prompt,
    build_focused_prompt,
    reframe,
    load_mental_model,
    get_mental_model_enhancer,
    _truncate,
)


class TestModelElement:
    """Tests for ModelElement dataclass."""

    def test_element_creation(self):
        """Should create element with required fields."""
        elem = ModelElement(id="auth", label="Authentication")
        assert elem.id == "auth"
        assert elem.label == "Authentication"
        assert elem.shape == "box"  # default
        assert elem.color is None
        assert elem.created_by == ""
        assert elem.reasoning == ""

    def test_element_with_all_fields(self):
        """Should create element with all fields."""
        elem = ModelElement(
            id="auth",
            label="Authentication",
            properties={"complexity": "high"},
            shape="hexagon",
            color="#ff0000",
            created_by="ai",
            reasoning="Handles user identity",
        )
        assert elem.properties == {"complexity": "high"}
        assert elem.shape == "hexagon"
        assert elem.color == "#ff0000"
        assert elem.created_by == "ai"
        assert elem.reasoning == "Handles user identity"

    def test_element_to_dict(self):
        """Should serialize to dictionary."""
        elem = ModelElement(
            id="auth",
            label="Auth",
            shape="circle",
            color="#00ff00",
            created_by="human",
            reasoning="test",
        )
        d = elem.to_dict()
        assert d["id"] == "auth"
        assert d["label"] == "Auth"
        assert d["shape"] == "circle"
        assert d["color"] == "#00ff00"
        assert d["created_by"] == "human"
        assert d["reasoning"] == "test"

    def test_element_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "id": "cart",
            "label": "Shopping Cart",
            "properties": {"items": 0},
            "shape": "rounded",
            "color": "#0000ff",
            "created_by": "ai",
            "reasoning": "Stores items",
        }
        elem = ModelElement.from_dict(d)
        assert elem.id == "cart"
        assert elem.label == "Shopping Cart"
        assert elem.properties == {"items": 0}
        assert elem.shape == "rounded"
        assert elem.color == "#0000ff"

    def test_element_equality_by_id(self):
        """Elements with same id should be distinguishable."""
        elem1 = ModelElement(id="test", label="Test")
        elem2 = ModelElement(id="test", label="Different Label")
        # Dataclasses compare by all fields, so these are different
        assert elem1.id == elem2.id
        assert elem1.label != elem2.label


class TestModelRelation:
    """Tests for ModelRelation dataclass."""

    def test_relation_creation(self):
        """Should create relation with required fields."""
        rel = ModelRelation(source_id="a", target_id="b")
        assert rel.source_id == "a"
        assert rel.target_id == "b"
        assert rel.label == ""
        assert rel.style == "solid"

    def test_relation_with_all_fields(self):
        """Should create relation with all fields."""
        rel = ModelRelation(
            source_id="auth",
            target_id="users",
            label="validates",
            properties={"async": True},
            style="dashed",
            created_by="ai",
            reasoning="Auth checks user DB",
        )
        assert rel.label == "validates"
        assert rel.properties == {"async": True}
        assert rel.style == "dashed"

    def test_relation_to_dict(self):
        """Should serialize to dictionary."""
        rel = ModelRelation(
            source_id="a",
            target_id="b",
            label="uses",
            style="dotted",
        )
        d = rel.to_dict()
        assert d["source_id"] == "a"
        assert d["target_id"] == "b"
        assert d["label"] == "uses"
        assert d["style"] == "dotted"

    def test_relation_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "source_id": "x",
            "target_id": "y",
            "label": "depends",
            "style": "solid",
        }
        rel = ModelRelation.from_dict(d)
        assert rel.source_id == "x"
        assert rel.target_id == "y"
        assert rel.label == "depends"


class TestSequenceStep:
    """Tests for SequenceStep dataclass."""

    def test_step_creation(self):
        """Should create step with required fields."""
        step = SequenceStep(
            from_participant="Client",
            to_participant="Server",
            message="GET /api/users",
        )
        assert step.from_participant == "Client"
        assert step.to_participant == "Server"
        assert step.message == "GET /api/users"
        assert step.step_type == "sync"  # default
        assert step.note is None

    def test_step_with_all_fields(self):
        """Should create step with all fields."""
        step = SequenceStep(
            from_participant="API",
            to_participant="Database",
            message="SELECT * FROM users",
            step_type="async",
            note="May take a while",
        )
        assert step.step_type == "async"
        assert step.note == "May take a while"


class TestSequenceDiagram:
    """Tests for SequenceDiagram dataclass."""

    def test_diagram_creation(self):
        """Should create empty diagram."""
        diagram = SequenceDiagram(id="login", title="User Login Flow")
        assert diagram.id == "login"
        assert diagram.title == "User Login Flow"
        assert diagram.participants == []
        assert diagram.steps == []

    def test_diagram_with_participants(self):
        """Should create diagram with participants."""
        diagram = SequenceDiagram(
            id="api-flow",
            title="API Request",
            participants=["Client", "API", "Database"],
        )
        assert len(diagram.participants) == 3
        assert "API" in diagram.participants

    def test_to_mermaid_basic(self):
        """Should generate valid mermaid sequence diagram."""
        diagram = SequenceDiagram(
            id="test",
            title="Test Flow",
            participants=["Client", "Server"],
        )
        diagram.steps.append(SequenceStep(
            from_participant="Client",
            to_participant="Server",
            message="Request",
        ))

        mermaid = diagram.to_mermaid()

        assert "sequenceDiagram" in mermaid
        assert "title Test Flow" in mermaid
        assert "participant Client" in mermaid
        assert "participant Server" in mermaid
        assert "Client->>Server: Request" in mermaid

    def test_to_mermaid_arrow_styles(self):
        """Should use correct arrow styles for step types."""
        diagram = SequenceDiagram(
            id="test",
            title="",
            participants=["A", "B"],
        )
        diagram.steps.append(SequenceStep(
            from_participant="A",
            to_participant="B",
            message="sync",
            step_type="sync",
        ))
        diagram.steps.append(SequenceStep(
            from_participant="B",
            to_participant="A",
            message="async",
            step_type="async",
        ))
        diagram.steps.append(SequenceStep(
            from_participant="A",
            to_participant="B",
            message="reply",
            step_type="reply",
        ))

        mermaid = diagram.to_mermaid()

        assert "A->>B: sync" in mermaid
        assert "B-->>A: async" in mermaid
        assert "A-->>B: reply" in mermaid

    def test_to_mermaid_with_notes(self):
        """Should include notes in mermaid output."""
        diagram = SequenceDiagram(
            id="test",
            title="",
            participants=["A", "B"],
        )
        diagram.steps.append(SequenceStep(
            from_participant="A",
            to_participant="B",
            message="Request",
            note="Important step",
        ))

        mermaid = diagram.to_mermaid()

        assert "Note over A,B: Important step" in mermaid

    def test_to_mermaid_sanitizes_spaces(self):
        """Should sanitize participant names with spaces."""
        diagram = SequenceDiagram(
            id="test",
            title="",
            participants=["API Gateway"],
        )
        diagram.steps.append(SequenceStep(
            from_participant="API Gateway",
            to_participant="API Gateway",
            message="Self call",
        ))

        mermaid = diagram.to_mermaid()

        assert "participant API_Gateway as API Gateway" in mermaid
        assert "API_Gateway->>API_Gateway: Self call" in mermaid

    def test_to_dict(self):
        """Should serialize to dictionary."""
        diagram = SequenceDiagram(
            id="login",
            title="Login",
            participants=["User", "Auth"],
            description="User authentication flow",
            created_by="ai",
            trigger="user login",
        )
        diagram.steps.append(SequenceStep(
            from_participant="User",
            to_participant="Auth",
            message="Login",
            step_type="sync",
            note="With credentials",
        ))

        d = diagram.to_dict()

        assert d["id"] == "login"
        assert d["title"] == "Login"
        assert d["participants"] == ["User", "Auth"]
        assert d["description"] == "User authentication flow"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["from"] == "User"
        assert d["steps"][0]["to"] == "Auth"
        assert d["steps"][0]["message"] == "Login"
        assert d["steps"][0]["type"] == "sync"
        assert d["steps"][0]["note"] == "With credentials"

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "id": "checkout",
            "title": "Checkout Flow",
            "participants": ["Cart", "Payment", "Order"],
            "steps": [
                {"from": "Cart", "to": "Payment", "message": "Pay", "type": "sync"},
                {"from": "Payment", "to": "Order", "message": "Create", "type": "async", "note": "Background"},
            ],
            "description": "Purchase flow",
            "created_by": "human",
            "trigger": "checkout button",
        }

        diagram = SequenceDiagram.from_dict(d)

        assert diagram.id == "checkout"
        assert diagram.title == "Checkout Flow"
        assert len(diagram.participants) == 3
        assert len(diagram.steps) == 2
        assert diagram.steps[0].from_participant == "Cart"
        assert diagram.steps[1].note == "Background"


class TestMentalModel:
    """Tests for MentalModel dataclass."""

    def test_empty_model(self):
        """Should create empty model."""
        model = MentalModel()
        assert len(model.elements) == 0
        assert len(model.relations) == 0
        assert model.version == 0
        assert model.ai_summary == ""

    def test_add_elements(self):
        """Should add elements to model."""
        model = MentalModel()
        model.elements["auth"] = ModelElement(id="auth", label="Auth")
        model.elements["db"] = ModelElement(id="db", label="Database")
        assert len(model.elements) == 2
        assert "auth" in model.elements
        assert "db" in model.elements

    def test_add_relations(self):
        """Should add relations to model."""
        model = MentalModel()
        model.elements["a"] = ModelElement(id="a", label="A")
        model.elements["b"] = ModelElement(id="b", label="B")
        model.relations.append(ModelRelation(source_id="a", target_id="b"))
        assert len(model.relations) == 1

    def test_snapshot_saves_state(self):
        """Should save current state as snapshot."""
        model = MentalModel()
        model.elements["test"] = ModelElement(id="test", label="Test")
        model.version = 1

        model.snapshot("first snapshot")

        assert len(model.snapshots) == 1
        assert model.snapshots[0].version == 1
        assert model.snapshots[0].trigger == "first snapshot"
        assert "test" in model.snapshots[0].elements

    def test_snapshot_is_deep_copy(self):
        """Snapshot should be independent of current state."""
        model = MentalModel()
        model.elements["test"] = ModelElement(id="test", label="Original")
        model.snapshot("before change")

        # Modify current state
        model.elements["test"].label = "Modified"

        # Snapshot should still have original
        assert model.snapshots[0].elements["test"].label == "Original"

    def test_to_mermaid_empty(self):
        """Empty model should produce minimal mermaid."""
        model = MentalModel()
        mermaid = model.to_mermaid()
        assert "graph TD" in mermaid

    def test_to_mermaid_with_elements(self):
        """Should generate valid mermaid with elements."""
        model = MentalModel()
        model.elements["api"] = ModelElement(id="api", label="REST API", shape="hexagon")
        model.elements["db"] = ModelElement(id="db", label="Database", shape="cylinder")

        mermaid = model.to_mermaid()

        assert "graph TD" in mermaid
        assert 'api{{"REST API"}}' in mermaid
        assert 'db[("Database")]' in mermaid

    def test_to_mermaid_with_relations(self):
        """Should generate mermaid with relationships."""
        model = MentalModel()
        model.elements["a"] = ModelElement(id="a", label="A")
        model.elements["b"] = ModelElement(id="b", label="B")
        model.relations.append(ModelRelation(source_id="a", target_id="b", label="uses"))

        mermaid = model.to_mermaid()

        assert 'a -->|"uses"| b' in mermaid

    def test_to_mermaid_with_colors(self):
        """Should include style directives for colors."""
        model = MentalModel()
        model.elements["test"] = ModelElement(id="test", label="Test", color="#ff0000")

        mermaid = model.to_mermaid()

        assert "style test fill:#ff0000" in mermaid

    def test_to_mermaid_shapes(self):
        """Should use correct mermaid shapes."""
        model = MentalModel()
        model.elements["box"] = ModelElement(id="box", label="Box", shape="box")
        model.elements["rounded"] = ModelElement(id="rounded", label="Rounded", shape="rounded")
        model.elements["circle"] = ModelElement(id="circle", label="Circle", shape="circle")
        model.elements["diamond"] = ModelElement(id="diamond", label="Diamond", shape="diamond")
        model.elements["cylinder"] = ModelElement(id="cylinder", label="Cylinder", shape="cylinder")
        model.elements["hexagon"] = ModelElement(id="hexagon", label="Hexagon", shape="hexagon")
        model.elements["stadium"] = ModelElement(id="stadium", label="Stadium", shape="stadium")

        mermaid = model.to_mermaid()

        assert 'box["Box"]' in mermaid
        assert 'rounded("Rounded")' in mermaid
        assert 'circle(("Circle"))' in mermaid
        assert 'diamond{"Diamond"}' in mermaid
        assert 'cylinder[("Cylinder")]' in mermaid
        assert 'hexagon{{"Hexagon"}}' in mermaid
        assert 'stadium(["Stadium"])' in mermaid

    def test_to_mermaid_relation_styles(self):
        """Should use correct arrow styles for relations."""
        model = MentalModel()
        model.elements["a"] = ModelElement(id="a", label="A")
        model.elements["b"] = ModelElement(id="b", label="B")
        model.elements["c"] = ModelElement(id="c", label="C")
        model.elements["d"] = ModelElement(id="d", label="D")

        model.relations.append(ModelRelation(source_id="a", target_id="b", style="solid"))
        model.relations.append(ModelRelation(source_id="b", target_id="c", style="dashed"))
        model.relations.append(ModelRelation(source_id="c", target_id="d", style="thick"))

        mermaid = model.to_mermaid()

        assert "a --> b" in mermaid
        assert "b -.-> c" in mermaid
        assert "c ==> d" in mermaid

    def test_to_json_roundtrip(self):
        """Should serialize and deserialize correctly."""
        model = MentalModel()
        model.elements["test"] = ModelElement(
            id="test", label="Test", shape="hexagon", color="#123456"
        )
        model.relations.append(ModelRelation(source_id="test", target_id="test", label="self"))
        model.version = 5
        model.ai_summary = "A test system"

        json_str = model.to_json()
        restored = MentalModel.from_dict(json.loads(json_str))

        assert len(restored.elements) == 1
        assert restored.elements["test"].label == "Test"
        assert restored.elements["test"].shape == "hexagon"
        assert len(restored.relations) == 1
        assert restored.relations[0].label == "self"
        assert restored.version == 5
        assert restored.ai_summary == "A test system"

    def test_to_dict(self):
        """Should convert to dictionary."""
        model = MentalModel()
        model.elements["x"] = ModelElement(id="x", label="X")
        model.version = 3

        d = model.to_dict()

        assert "elements" in d
        assert "relations" in d
        assert "version" in d
        assert d["version"] == 3

    def test_sequences_in_model(self):
        """Should support sequence diagrams."""
        model = MentalModel()
        diagram = SequenceDiagram(
            id="login",
            title="Login Flow",
            participants=["User", "Auth", "DB"],
        )
        diagram.steps.append(SequenceStep(
            from_participant="User",
            to_participant="Auth",
            message="Login",
        ))
        model.sequences["login"] = diagram

        assert len(model.sequences) == 1
        assert "login" in model.sequences
        assert model.sequences["login"].title == "Login Flow"

    def test_to_dict_includes_sequences(self):
        """Should include sequences in to_dict."""
        model = MentalModel()
        model.sequences["test"] = SequenceDiagram(
            id="test",
            title="Test",
            participants=["A", "B"],
        )

        d = model.to_dict()

        assert "sequences" in d
        assert "test" in d["sequences"]
        assert d["sequences"]["test"]["title"] == "Test"

    def test_from_dict_restores_sequences(self):
        """Should restore sequences from dict."""
        d = {
            "elements": {},
            "relations": [],
            "sequences": {
                "checkout": {
                    "id": "checkout",
                    "title": "Checkout",
                    "participants": ["Cart", "Payment"],
                    "steps": [
                        {"from": "Cart", "to": "Payment", "message": "Pay", "type": "sync"}
                    ],
                }
            },
            "version": 1,
            "ai_summary": "",
        }

        model = MentalModel.from_dict(d)

        assert len(model.sequences) == 1
        assert "checkout" in model.sequences
        assert model.sequences["checkout"].title == "Checkout"
        assert len(model.sequences["checkout"].steps) == 1

    def test_to_full_mermaid(self):
        """Should export both component and sequence diagrams."""
        model = MentalModel()
        model.elements["api"] = ModelElement(id="api", label="API")
        model.sequences["flow"] = SequenceDiagram(
            id="flow",
            title="Request Flow",
            participants=["Client", "Server"],
            description="How requests work",
        )
        model.sequences["flow"].steps.append(SequenceStep(
            from_participant="Client",
            to_participant="Server",
            message="Request",
        ))

        full = model.to_full_mermaid()

        # Should have component diagram
        assert "## Component Diagram" in full
        assert "graph TD" in full
        assert "API" in full

        # Should have sequence diagram
        assert "## Request Flow" in full
        assert "How requests work" in full
        assert "sequenceDiagram" in full
        assert "Client->>Server: Request" in full


class TestMentalModelEnhancer:
    """Tests for MentalModelEnhancer class."""

    def test_init_without_repo(self):
        """Should initialize without repo path."""
        enhancer = MentalModelEnhancer()
        assert enhancer.model is not None
        assert enhancer._repo_path is None

    def test_init_with_nonexistent_repo(self, tmp_path):
        """Should initialize with new repo path."""
        repo_path = tmp_path / "new_repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)

        assert enhancer._repo_path == repo_path
        assert len(enhancer.model.elements) == 0

    def test_init_loads_existing_model(self, tmp_path):
        """Should load existing model from repo."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create existing model file
        model_dir = repo_path / ".agentgit"
        model_dir.mkdir()
        model_file = model_dir / "mental_model.json"

        existing_model = {
            "elements": {
                "existing": {
                    "id": "existing",
                    "label": "Existing Element",
                    "shape": "box",
                }
            },
            "relations": [],
            "version": 10,
            "ai_summary": "Existing model",
        }
        model_file.write_text(json.dumps(existing_model))

        enhancer = MentalModelEnhancer(repo_path)

        assert len(enhancer.model.elements) == 1
        assert "existing" in enhancer.model.elements
        assert enhancer.model.version == 10

    def test_model_path_property(self, tmp_path):
        """Should return correct model path."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)

        assert enhancer.model_path == repo_path / ".agentgit" / "mental_model.json"

    def test_model_path_none_without_repo(self):
        """Should return None if no repo configured."""
        enhancer = MentalModelEnhancer()
        assert enhancer.model_path is None

    def test_save_creates_directory(self, tmp_path):
        """Should create .agentgit directory if needed."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        enhancer.model.elements["test"] = ModelElement(id="test", label="Test")
        enhancer.save()

        assert (repo_path / ".agentgit").exists()
        assert (repo_path / ".agentgit" / "mental_model.json").exists()

    def test_save_writes_model(self, tmp_path):
        """Should write model to file."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        enhancer.model.elements["test"] = ModelElement(id="test", label="Test Element")
        enhancer.model.version = 42
        enhancer.save()

        content = (repo_path / ".agentgit" / "mental_model.json").read_text()
        data = json.loads(content)

        assert "test" in data["elements"]
        assert data["elements"]["test"]["label"] == "Test Element"
        assert data["version"] == 42

    def test_save_insights_creates_file(self, tmp_path):
        """Should create insights file with header."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        path = enhancer.save_insights()

        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "# Reframe Insights" in content

    def test_save_insights_appends(self, tmp_path):
        """Should append insights to file."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        enhancer.save_insights("First insight")
        enhancer.save_insights("Second insight")

        content = (repo_path / ".agentgit" / "mental_model.md").read_text()

        assert "First insight" in content
        assert "Second insight" in content

    def test_save_insights_includes_timestamp(self, tmp_path):
        """Should include timestamp with each insight."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        enhancer.save_insights("Test insight")

        content = (repo_path / ".agentgit" / "mental_model.md").read_text()

        # Should have a timestamp header like "## 2026-01-10 14:30"
        import re
        assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2}", content)

    def test_load_insights_returns_content(self, tmp_path):
        """Should load existing insights."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        enhancer.save_insights("Test insight content")

        loaded = enhancer.load_insights()

        assert "Test insight content" in loaded

    def test_load_insights_empty_if_no_file(self, tmp_path):
        """Should return empty string if no insights file."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        loaded = enhancer.load_insights()

        assert loaded == ""

    def test_add_human_insight(self, tmp_path):
        """Should add human insight with marker."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)
        enhancer.add_human_insight("The checkout is actually three steps")

        content = (repo_path / ".agentgit" / "mental_model.md").read_text()

        assert "**Human insight:**" in content
        assert "The checkout is actually three steps" in content


class TestPromptBuilding:
    """Tests for prompt generation functions."""

    def test_build_observation_prompt_empty_model(self):
        """Should handle empty model."""
        model = MentalModel()
        context = {"prompts": ["Add auth"], "files_changed": ["auth.py"]}

        prompt = build_observation_prompt(model, context)

        assert "mental model" in prompt.lower()
        assert "(empty)" in prompt
        assert "auth.py" in prompt

    def test_build_observation_prompt_with_model(self):
        """Should include current model diagram."""
        model = MentalModel()
        model.elements["test"] = ModelElement(id="test", label="Test Component")
        context = {"prompts": [], "files_changed": []}

        prompt = build_observation_prompt(model, context)

        assert "Test Component" in prompt
        assert "```mermaid" in prompt

    def test_build_observation_prompt_with_insights(self):
        """Should include accumulated insights."""
        model = MentalModel()
        context = {"prompts": [], "files_changed": []}
        insights = "Previous insight: The system uses microservices"

        prompt = build_observation_prompt(model, context, insights)

        assert "Previous insight" in prompt
        assert "microservices" in prompt

    def test_build_observation_prompt_truncates_long_insights(self):
        """Should truncate very long insights."""
        model = MentalModel()
        context = {}
        long_insights = "x" * 5000

        prompt = build_observation_prompt(model, context, long_insights)

        assert "(earlier insights truncated)" in prompt

    def test_build_observation_prompt_includes_summary(self):
        """Should include AI summary if present."""
        model = MentalModel()
        model.ai_summary = "An e-commerce platform"
        context = {}

        prompt = build_observation_prompt(model, context)

        assert "An e-commerce platform" in prompt

    def test_build_instruction_prompt_with_selection(self):
        """Should include selected elements."""
        model = MentalModel()
        model.elements["checkout"] = ModelElement(
            id="checkout",
            label="Checkout",
            reasoning="Where purchases happen",
        )

        prompt = build_instruction_prompt(
            model,
            selected_ids=["checkout"],
            instruction="Split this into three steps",
        )

        assert "Checkout" in prompt
        assert "Split this into three steps" in prompt
        assert "Where purchases happen" in prompt

    def test_build_instruction_prompt_empty_selection(self):
        """Should handle empty selection."""
        model = MentalModel()

        prompt = build_instruction_prompt(
            model,
            selected_ids=[],
            instruction="Add a database component",
        )

        assert "Add a database component" in prompt
        assert "No specific elements" in prompt

    def test_build_focused_prompt_explore(self):
        """Should generate explore prompt."""
        model = MentalModel()
        model.elements["auth"] = ModelElement(id="auth", label="Authentication")

        prompt = build_focused_prompt(model, "auth", "explore")

        assert "Authentication" in prompt
        assert "Tell me" in prompt or "tell me" in prompt

    def test_build_focused_prompt_debug(self):
        """Should generate debug prompt."""
        model = MentalModel()
        model.elements["auth"] = ModelElement(id="auth", label="Authentication")

        prompt = build_focused_prompt(model, "auth", "debug")

        assert "issues" in prompt.lower() or "help" in prompt.lower()

    def test_build_focused_prompt_unknown_element(self):
        """Should handle unknown element."""
        model = MentalModel()

        prompt = build_focused_prompt(model, "nonexistent", "explore")

        assert "not found" in prompt.lower()

    def test_build_focused_prompt_mentions_element(self):
        """Should mention the element in the prompt."""
        model = MentalModel()
        model.elements["auth"] = ModelElement(id="auth", label="Authentication")
        model.elements["users"] = ModelElement(id="users", label="User Store")
        model.relations.append(
            ModelRelation(source_id="auth", target_id="users", label="validates")
        )

        prompt = build_focused_prompt(model, "auth", "explore")

        # Prompt should mention the element being explored
        assert "Authentication" in prompt


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        """Should not truncate short text."""
        result = _truncate("short", 100)
        assert result == "short"

    def test_truncate_long_text(self):
        """Should truncate and add ellipsis."""
        result = _truncate("a" * 100, 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_exact_length(self):
        """Should not truncate at exact length."""
        result = _truncate("exact", 5)
        assert result == "exact"


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_load_mental_model_exists(self, tmp_path):
        """Should load model from repo."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        model_dir = repo_path / ".agentgit"
        model_dir.mkdir()

        model_data = {
            "elements": {"test": {"id": "test", "label": "Test", "shape": "box"}},
            "relations": [],
            "version": 5,
            "ai_summary": "Test model",
        }
        (model_dir / "mental_model.json").write_text(json.dumps(model_data))

        loaded = load_mental_model(repo_path)

        assert loaded is not None
        assert "test" in loaded.elements
        assert loaded.version == 5

    def test_load_mental_model_not_exists(self, tmp_path):
        """Should return None if model doesn't exist."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        loaded = load_mental_model(repo_path)

        assert loaded is None

    def test_get_mental_model_enhancer_creates_instance(self):
        """Should create enhancer instance."""
        # Reset global state
        import agentgit.enhancers.mental_model as mm
        mm._enhancer_instance = None

        enhancer = get_mental_model_enhancer()

        assert enhancer is not None
        assert isinstance(enhancer, MentalModelEnhancer)

    def test_get_mental_model_enhancer_reuses_instance(self):
        """Should reuse existing instance."""
        # Reset global state
        import agentgit.enhancers.mental_model as mm
        mm._enhancer_instance = None

        enhancer1 = get_mental_model_enhancer()
        enhancer2 = get_mental_model_enhancer()

        assert enhancer1 is enhancer2

    def test_get_mental_model_enhancer_new_repo(self, tmp_path):
        """Should create new instance for different repo."""
        import agentgit.enhancers.mental_model as mm
        mm._enhancer_instance = None

        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        repo2 = tmp_path / "repo2"
        repo2.mkdir()

        enhancer1 = get_mental_model_enhancer(repo1)
        enhancer2 = get_mental_model_enhancer(repo2)

        assert enhancer1._repo_path == repo1
        assert enhancer2._repo_path == repo2


class TestIntegration:
    """Integration tests for the full reframe flow."""

    def test_full_model_evolution(self, tmp_path):
        """Test a complete model evolution scenario."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        enhancer = MentalModelEnhancer(repo_path)

        # Step 1: Initial model
        enhancer.model.elements["browse"] = ModelElement(
            id="browse", label="Product Browser", shape="stadium"
        )
        enhancer.model.elements["cart"] = ModelElement(
            id="cart", label="Shopping Cart", shape="rounded"
        )
        enhancer.model.relations.append(
            ModelRelation(source_id="browse", target_id="cart", label="add to")
        )
        enhancer.model.version = 1
        enhancer.model.ai_summary = "E-commerce product browsing"
        enhancer.save()

        # Step 2: Human adds insight
        enhancer.add_human_insight("The cart should support wishlists too")

        # Step 3: Snapshot and evolve
        enhancer.model.snapshot("before wishlist")
        enhancer.model.elements["wishlist"] = ModelElement(
            id="wishlist", label="Wishlist", shape="rounded"
        )
        enhancer.model.relations.append(
            ModelRelation(source_id="browse", target_id="wishlist", label="save for later")
        )
        enhancer.model.version = 2
        enhancer.save()

        # Verify final state
        loaded = load_mental_model(repo_path)
        assert len(loaded.elements) == 3
        assert "wishlist" in loaded.elements
        assert loaded.version == 2

        # Verify insights
        insights = enhancer.load_insights()
        assert "wishlists" in insights

        # Verify snapshots
        assert len(enhancer.model.snapshots) == 1
        assert len(enhancer.model.snapshots[0].elements) == 2  # Before wishlist

    def test_mermaid_output_is_valid(self, tmp_path):
        """Test that generated mermaid is syntactically valid."""
        model = MentalModel()

        # Add various elements and relations
        model.elements["api"] = ModelElement(
            id="api", label="API Gateway", shape="hexagon", color="#e1f5fe"
        )
        model.elements["auth"] = ModelElement(
            id="auth", label="Auth Service", shape="box", color="#fff3e0"
        )
        model.elements["db"] = ModelElement(
            id="db", label="Database", shape="cylinder", color="#e8f5e9"
        )

        model.relations.append(
            ModelRelation(source_id="api", target_id="auth", label="authenticates", style="solid")
        )
        model.relations.append(
            ModelRelation(source_id="auth", target_id="db", label="queries", style="dashed")
        )

        mermaid = model.to_mermaid()

        # Check structure
        lines = mermaid.strip().split("\n")
        assert lines[0].strip() == "graph TD"

        # Check no syntax errors (basic validation)
        assert mermaid.count("(") == mermaid.count(")")
        assert mermaid.count("[") == mermaid.count("]")
        assert mermaid.count("{") == mermaid.count("}")
