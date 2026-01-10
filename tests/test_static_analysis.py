"""Tests for static analysis module."""

import pytest
import textwrap
from pathlib import Path

from agentgit.enhancers.static_analysis import (
    CodeEntity,
    CodeRelation,
    CodeStructure,
    analyze_python_simple,
    analyze_javascript_simple,
    analyze_code,
    structure_to_context,
)


class TestCodeEntity:
    """Tests for CodeEntity dataclass."""

    def test_entity_creation(self):
        """Should create entity with required fields."""
        entity = CodeEntity(
            name="UserService",
            entity_type="class",
            file_path="src/services/user.py",
        )
        assert entity.name == "UserService"
        assert entity.entity_type == "class"
        assert entity.file_path == "src/services/user.py"
        assert entity.line_number is None
        assert entity.docstring is None
        assert entity.properties == {}

    def test_entity_with_all_fields(self):
        """Should create entity with all fields."""
        entity = CodeEntity(
            name="authenticate",
            entity_type="function",
            file_path="auth.py",
            line_number=42,
            docstring="Authenticate a user.",
            properties={"async": True, "public": True},
        )
        assert entity.line_number == 42
        assert entity.docstring == "Authenticate a user."
        assert entity.properties["async"] is True


class TestCodeRelation:
    """Tests for CodeRelation dataclass."""

    def test_relation_creation(self):
        """Should create relation with required fields."""
        rel = CodeRelation(
            source="UserService",
            target="Database",
            relation_type="uses",
            file_path="services.py",
        )
        assert rel.source == "UserService"
        assert rel.target == "Database"
        assert rel.relation_type == "uses"
        assert rel.file_path == "services.py"

    def test_relation_types(self):
        """Should support various relation types."""
        for rel_type in ["calls", "imports", "inherits", "implements", "uses"]:
            rel = CodeRelation(
                source="A",
                target="B",
                relation_type=rel_type,
                file_path="test.py",
            )
            assert rel.relation_type == rel_type


class TestCodeStructure:
    """Tests for CodeStructure dataclass."""

    def test_empty_structure(self):
        """Should create empty structure."""
        structure = CodeStructure()
        assert structure.entities == []
        assert structure.relations == []
        assert structure.language == "unknown"
        assert structure.analysis_method == "unknown"

    def test_structure_with_data(self):
        """Should hold entities and relations."""
        structure = CodeStructure(
            language="python",
            analysis_method="regex",
        )
        structure.entities.append(CodeEntity(
            name="TestClass",
            entity_type="class",
            file_path="test.py",
        ))
        structure.relations.append(CodeRelation(
            source="test.py",
            target="os",
            relation_type="imports",
            file_path="test.py",
        ))

        assert len(structure.entities) == 1
        assert len(structure.relations) == 1
        assert structure.language == "python"


class TestAnalyzePythonSimple:
    """Tests for analyze_python_simple function."""

    def test_finds_classes(self, tmp_path):
        """Should find class definitions."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
class UserService:
    pass

class OrderService(BaseService):
    pass
""")

        structure = analyze_python_simple(tmp_path)

        class_entities = [e for e in structure.entities if e.entity_type == "class"]
        assert len(class_entities) == 2
        names = [e.name for e in class_entities]
        assert "UserService" in names
        assert "OrderService" in names

    def test_finds_functions(self, tmp_path):
        """Should find function definitions."""
        py_file = tmp_path / "test.py"
        py_file.write_text(textwrap.dedent("""\
            def top_level_func():
                pass

            def another_function(arg):
                return arg
            """))

        structure = analyze_python_simple(tmp_path)

        func_entities = [e for e in structure.entities if e.entity_type == "function"]
        assert len(func_entities) == 2
        names = [e.name for e in func_entities]
        assert "top_level_func" in names
        assert "another_function" in names

    def test_finds_methods(self, tmp_path):
        """Should find method definitions."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
class MyClass:
    def my_method(self):
        pass

    def another_method(self):
        pass
""")

        structure = analyze_python_simple(tmp_path)

        method_entities = [e for e in structure.entities if e.entity_type == "method"]
        assert len(method_entities) == 2

    def test_tracks_inheritance(self, tmp_path):
        """Should track inheritance relations."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
class Child(Parent):
    pass

class MultiChild(Parent1, Parent2):
    pass
""")

        structure = analyze_python_simple(tmp_path)

        inherit_rels = [r for r in structure.relations if r.relation_type == "inherits"]
        assert len(inherit_rels) >= 2

        targets = [r.target for r in inherit_rels]
        assert "Parent" in targets
        assert "Parent1" in targets

    def test_tracks_imports(self, tmp_path):
        """Should track import relations."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
import os
from pathlib import Path
from typing import List, Dict
""")

        structure = analyze_python_simple(tmp_path)

        import_rels = [r for r in structure.relations if r.relation_type == "imports"]
        assert len(import_rels) >= 3

        targets = [r.target for r in import_rels]
        assert "os" in targets
        assert "pathlib.Path" in targets

    def test_handles_single_file(self, tmp_path):
        """Should handle single file path."""
        py_file = tmp_path / "single.py"
        py_file.write_text("""
class SingleClass:
    pass
""")

        structure = analyze_python_simple(py_file)

        assert len(structure.entities) >= 1
        assert structure.language == "python"
        assert structure.analysis_method == "regex"

    def test_ignores_object_base(self, tmp_path):
        """Should ignore 'object' as base class."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
class MyClass(object):
    pass
""")

        structure = analyze_python_simple(tmp_path)

        inherit_rels = [r for r in structure.relations if r.relation_type == "inherits"]
        targets = [r.target for r in inherit_rels]
        assert "object" not in targets

    def test_includes_line_numbers(self, tmp_path):
        """Should include line numbers."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
# Line 1 is blank

class MyClass:
    pass
""")

        structure = analyze_python_simple(tmp_path)

        class_entities = [e for e in structure.entities if e.entity_type == "class"]
        assert len(class_entities) == 1
        assert class_entities[0].line_number is not None
        assert class_entities[0].line_number > 0


class TestAnalyzeJavaScriptSimple:
    """Tests for analyze_javascript_simple function."""

    def test_finds_classes(self, tmp_path):
        """Should find class definitions."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
class UserService {
    constructor() {}
}

class OrderService extends BaseService {
    process() {}
}
""")

        structure = analyze_javascript_simple(tmp_path)

        class_entities = [e for e in structure.entities if e.entity_type == "class"]
        assert len(class_entities) == 2
        names = [e.name for e in class_entities]
        assert "UserService" in names
        assert "OrderService" in names

    def test_tracks_extends(self, tmp_path):
        """Should track extends relations."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
class Child extends Parent {
}
""")

        structure = analyze_javascript_simple(tmp_path)

        extend_rels = [r for r in structure.relations if r.relation_type == "extends"]
        assert len(extend_rels) == 1
        assert extend_rels[0].target == "Parent"

    def test_finds_functions(self, tmp_path):
        """Should find function definitions."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function regularFunction() {
    return 1;
}

const arrowFunc = () => {
    return 2;
};

const anotherArrow = (x) => x * 2;
""")

        structure = analyze_javascript_simple(tmp_path)

        func_entities = [e for e in structure.entities if e.entity_type == "function"]
        names = [e.name for e in func_entities]
        assert "regularFunction" in names
        assert "arrowFunc" in names

    def test_finds_react_components(self, tmp_path):
        """Should find React components (PascalCase functions)."""
        tsx_file = tmp_path / "test.tsx"
        tsx_file.write_text(textwrap.dedent("""\
            function UserProfile() {
                return <div>Profile</div>;
            }

            const OrderList = () => {
                return <ul></ul>;
            };
            """))

        structure = analyze_javascript_simple(tmp_path)

        # React components are detected as functions (PascalCase naming)
        names = [e.name for e in structure.entities]
        # Should find the PascalCase function declarations
        assert "UserProfile" in names
        assert "OrderList" in names

    def test_tracks_imports(self, tmp_path):
        """Should track import relations."""
        js_file = tmp_path / "test.js"
        js_file.write_text("""
import React from 'react';
import { useState, useEffect } from 'react';
import axios from './utils/axios';
""")

        structure = analyze_javascript_simple(tmp_path)

        import_rels = [r for r in structure.relations if r.relation_type == "imports"]
        targets = [r.target for r in import_rels]
        assert "react" in targets
        assert "./utils/axios" in targets

    def test_handles_typescript(self, tmp_path):
        """Should handle TypeScript files."""
        ts_file = tmp_path / "test.ts"
        ts_file.write_text("""
interface User {
    name: string;
}

class UserService {
    getUser(): User {
        return { name: 'test' };
    }
}
""")

        structure = analyze_javascript_simple(tmp_path)

        class_entities = [e for e in structure.entities if e.entity_type == "class"]
        assert len(class_entities) == 1
        assert class_entities[0].name == "UserService"

    def test_skips_node_modules(self, tmp_path):
        """Should skip node_modules directory."""
        node_modules = tmp_path / "node_modules" / "some-package"
        node_modules.mkdir(parents=True)
        (node_modules / "index.js").write_text("class ExternalClass {}")

        src = tmp_path / "src"
        src.mkdir()
        (src / "app.js").write_text("class AppClass {}")

        structure = analyze_javascript_simple(tmp_path)

        names = [e.name for e in structure.entities]
        assert "AppClass" in names
        assert "ExternalClass" not in names


class TestAnalyzeCode:
    """Tests for analyze_code function."""

    def test_detects_python_project(self, tmp_path):
        """Should detect and analyze Python project."""
        (tmp_path / "main.py").write_text("class Main: pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        structure = analyze_code(tmp_path)

        assert structure.language == "python"
        assert len(structure.entities) >= 2

    def test_detects_javascript_project(self, tmp_path):
        """Should detect and analyze JavaScript project."""
        (tmp_path / "app.js").write_text("function app() {}")
        (tmp_path / "utils.js").write_text("const helper = () => {}")
        (tmp_path / "index.ts").write_text("class Index {}")

        structure = analyze_code(tmp_path)

        assert structure.language == "javascript"
        assert len(structure.entities) >= 2

    def test_handles_single_python_file(self, tmp_path):
        """Should handle single Python file."""
        py_file = tmp_path / "single.py"
        py_file.write_text("class Single: pass")

        structure = analyze_code(py_file)

        assert structure.language == "python"
        assert len(structure.entities) == 1

    def test_handles_single_javascript_file(self, tmp_path):
        """Should handle single JavaScript file."""
        js_file = tmp_path / "single.js"
        js_file.write_text("class Single {}")

        structure = analyze_code(js_file)

        assert structure.language == "javascript"
        assert len(structure.entities) >= 1

    def test_mixed_project_uses_majority(self, tmp_path):
        """Should use majority language for mixed projects."""
        # More Python files
        (tmp_path / "a.py").write_text("class A: pass")
        (tmp_path / "b.py").write_text("class B: pass")
        (tmp_path / "c.py").write_text("class C: pass")
        (tmp_path / "one.js").write_text("class One {}")

        structure = analyze_code(tmp_path)

        assert structure.language == "python"


class TestStructureToContext:
    """Tests for structure_to_context function."""

    def test_groups_entities_by_type(self):
        """Should group entities by type."""
        structure = CodeStructure(language="python", analysis_method="regex")
        structure.entities.append(CodeEntity(name="UserService", entity_type="class", file_path="a.py"))
        structure.entities.append(CodeEntity(name="OrderService", entity_type="class", file_path="b.py"))
        structure.entities.append(CodeEntity(name="helper", entity_type="function", file_path="c.py"))

        context = structure_to_context(structure)

        assert context["language"] == "python"
        assert context["analysis_method"] == "regex"
        assert "class" in context["entities"]
        assert "function" in context["entities"]
        assert len(context["entities"]["class"]) == 2
        assert len(context["entities"]["function"]) == 1

    def test_includes_entity_counts(self):
        """Should include entity counts."""
        structure = CodeStructure()
        for i in range(25):
            structure.entities.append(CodeEntity(name=f"Class{i}", entity_type="class", file_path="x.py"))

        context = structure_to_context(structure)

        assert context["entity_counts"]["class"] == 25
        # But entities list should be limited
        assert len(context["entities"]["class"]) <= 20

    def test_summarizes_relations(self):
        """Should summarize relation counts."""
        structure = CodeStructure()
        for i in range(5):
            structure.relations.append(CodeRelation(
                source=f"Class{i}",
                target="Base",
                relation_type="inherits",
                file_path="x.py",
            ))
        for i in range(3):
            structure.relations.append(CodeRelation(
                source="x.py",
                target=f"module{i}",
                relation_type="imports",
                file_path="x.py",
            ))

        context = structure_to_context(structure)

        assert context["relation_counts"]["inherits"] == 5
        assert context["relation_counts"]["imports"] == 3

    def test_includes_sample_relations(self):
        """Should include sample relations."""
        structure = CodeStructure()
        for i in range(15):
            structure.relations.append(CodeRelation(
                source=f"A{i}",
                target=f"B{i}",
                relation_type="calls",
                file_path="x.py",
            ))

        context = structure_to_context(structure)

        assert len(context["sample_relations"]) <= 10
        assert context["sample_relations"][0]["from"] == "A0"
        assert context["sample_relations"][0]["to"] == "B0"
        assert context["sample_relations"][0]["type"] == "calls"
