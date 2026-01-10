"""Static analysis support for Reframe.

Uses available tools to extract structural information from code:
- tree-sitter for multi-language AST parsing
- pyan for Python call graphs
- Simple regex fallbacks when tools unavailable

This provides a foundation for the AI to work with, rather than
having to infer everything from file names and diffs.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodeEntity:
    """A code entity (function, class, module, etc.)."""

    name: str
    entity_type: str  # function, class, module, interface, etc.
    file_path: str
    line_number: int | None = None
    docstring: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeRelation:
    """A relationship between code entities."""

    source: str  # Entity name
    target: str  # Entity name
    relation_type: str  # calls, imports, inherits, implements, uses
    file_path: str | None = None
    line_number: int | None = None


@dataclass
class CodeStructure:
    """Extracted code structure from static analysis."""

    entities: list[CodeEntity] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    language: str = "unknown"
    analysis_method: str = "unknown"


def analyze_python_with_pyan(path: Path) -> CodeStructure | None:
    """Use pyan to analyze Python code and extract call graph.

    Requires: pip install pyan3
    """
    try:
        # Check if pyan is available
        result = subprocess.run(
            ["pyan3", "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
    except FileNotFoundError:
        return None

    # Find Python files
    if path.is_file():
        py_files = [path] if path.suffix == ".py" else []
    else:
        py_files = list(path.glob("**/*.py"))

    if not py_files:
        return None

    # Run pyan to get call graph in dot format
    try:
        result = subprocess.run(
            ["pyan3", "--dot", "--no-defines", "--grouped"] + [str(f) for f in py_files[:50]],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, Exception):
        return None

    if result.returncode != 0:
        return None

    # Parse dot output
    structure = CodeStructure(language="python", analysis_method="pyan")

    # Extract nodes (functions/methods)
    for match in re.finditer(r'"([^"]+)"\s*\[', result.stdout):
        name = match.group(1)
        # pyan uses module.class.method format
        parts = name.split(".")
        entity_type = "function" if len(parts) <= 2 else "method"
        structure.entities.append(CodeEntity(
            name=name,
            entity_type=entity_type,
            file_path="",  # pyan doesn't give us this easily
        ))

    # Extract edges (calls)
    for match in re.finditer(r'"([^"]+)"\s*->\s*"([^"]+)"', result.stdout):
        source, target = match.groups()
        structure.relations.append(CodeRelation(
            source=source,
            target=target,
            relation_type="calls",
        ))

    return structure


def analyze_with_ctags(path: Path) -> CodeStructure | None:
    """Use universal-ctags to extract code structure.

    Requires: universal-ctags installed
    Works with many languages.
    """
    try:
        result = subprocess.run(
            ["ctags", "--version"],
            capture_output=True,
            text=True,
        )
        if "Universal Ctags" not in result.stdout:
            return None
    except FileNotFoundError:
        return None

    # Run ctags
    try:
        result = subprocess.run(
            [
                "ctags",
                "-R",
                "--output-format=json",
                "--fields=+n+S+K",  # line number, signature, kind
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, Exception):
        return None

    structure = CodeStructure(analysis_method="ctags")

    import json
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            tag = json.loads(line)
            kind = tag.get("kind", "unknown")
            entity_type = {
                "function": "function",
                "method": "method",
                "class": "class",
                "interface": "interface",
                "module": "module",
                "variable": "variable",
            }.get(kind, kind)

            structure.entities.append(CodeEntity(
                name=tag.get("name", ""),
                entity_type=entity_type,
                file_path=tag.get("path", ""),
                line_number=tag.get("line"),
            ))
        except json.JSONDecodeError:
            continue

    return structure


def analyze_python_simple(path: Path) -> CodeStructure:
    """Simple regex-based Python analysis (no dependencies).

    Less accurate than AST parsing but always available.
    """
    structure = CodeStructure(language="python", analysis_method="regex")

    if path.is_file():
        py_files = [path] if path.suffix == ".py" else []
    else:
        py_files = list(path.glob("**/*.py"))

    for py_file in py_files[:100]:  # Limit for performance
        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue

        rel_path = str(py_file)

        # Find classes
        for match in re.finditer(r'^class\s+(\w+)(?:\(([^)]*)\))?:', content, re.MULTILINE):
            class_name = match.group(1)
            bases = match.group(2)
            structure.entities.append(CodeEntity(
                name=class_name,
                entity_type="class",
                file_path=rel_path,
                line_number=content[:match.start()].count("\n") + 1,
            ))

            # Track inheritance
            if bases:
                for base in bases.split(","):
                    base = base.strip().split("(")[0].split("[")[0]
                    if base and base not in ("object", "ABC", "Protocol"):
                        structure.relations.append(CodeRelation(
                            source=class_name,
                            target=base,
                            relation_type="inherits",
                            file_path=rel_path,
                        ))

        # Find functions/methods
        # Use [ \t]* instead of \s* to avoid matching newlines as indentation
        for match in re.finditer(r'^([ \t]*)def\s+(\w+)\s*\(', content, re.MULTILINE):
            indent = len(match.group(1))
            func_name = match.group(2)
            entity_type = "method" if indent > 0 else "function"
            structure.entities.append(CodeEntity(
                name=func_name,
                entity_type=entity_type,
                file_path=rel_path,
                line_number=content[:match.start()].count("\n") + 1,
            ))

        # Find imports
        for match in re.finditer(r'^(?:from\s+(\S+)\s+)?import\s+(.+)$', content, re.MULTILINE):
            module = match.group(1) or ""
            imports = match.group(2)
            for imp in imports.split(","):
                imp = imp.strip().split(" as ")[0].strip()
                if imp:
                    target = f"{module}.{imp}" if module else imp
                    structure.relations.append(CodeRelation(
                        source=rel_path,
                        target=target,
                        relation_type="imports",
                        file_path=rel_path,
                    ))

    return structure


def analyze_javascript_simple(path: Path) -> CodeStructure:
    """Simple regex-based JavaScript/TypeScript analysis."""
    structure = CodeStructure(language="javascript", analysis_method="regex")

    if path.is_file():
        js_files = [path] if path.suffix in (".js", ".ts", ".jsx", ".tsx") else []
    else:
        js_files = []
        for ext in ("*.js", "*.ts", "*.jsx", "*.tsx"):
            js_files.extend(path.glob(f"**/{ext}"))

    for js_file in js_files[:100]:
        # Skip node_modules (check for path segment, not just substring)
        path_parts = js_file.parts
        if "node_modules" in path_parts:
            continue

        try:
            content = js_file.read_text(errors="ignore")
        except Exception:
            continue

        rel_path = str(js_file)

        # Find classes
        for match in re.finditer(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', content):
            class_name = match.group(1)
            base = match.group(2)
            structure.entities.append(CodeEntity(
                name=class_name,
                entity_type="class",
                file_path=rel_path,
                line_number=content[:match.start()].count("\n") + 1,
            ))
            if base:
                structure.relations.append(CodeRelation(
                    source=class_name,
                    target=base,
                    relation_type="extends",
                    file_path=rel_path,
                ))

        # Find functions
        for match in re.finditer(r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)', content):
            func_name = match.group(1) or match.group(2)
            if func_name:
                structure.entities.append(CodeEntity(
                    name=func_name,
                    entity_type="function",
                    file_path=rel_path,
                    line_number=content[:match.start()].count("\n") + 1,
                ))

        # Find React components (PascalCase functions returning JSX)
        for match in re.finditer(r'(?:function|const)\s+([A-Z]\w+)', content):
            comp_name = match.group(1)
            if comp_name not in [e.name for e in structure.entities]:
                structure.entities.append(CodeEntity(
                    name=comp_name,
                    entity_type="component",
                    file_path=rel_path,
                    line_number=content[:match.start()].count("\n") + 1,
                ))

        # Find imports
        for match in re.finditer(r'import\s+(?:{[^}]+}|\w+)\s+from\s+[\'"]([^\'"]+)[\'"]', content):
            module = match.group(1)
            structure.relations.append(CodeRelation(
                source=rel_path,
                target=module,
                relation_type="imports",
                file_path=rel_path,
            ))

    return structure


def analyze_code(path: Path) -> CodeStructure:
    """Analyze code using best available method.

    Tries in order:
    1. pyan (for Python, if available)
    2. ctags (multi-language, if available)
    3. Simple regex (always works)
    """
    path = Path(path)

    # Detect primary language
    if path.is_file():
        suffix = path.suffix.lower()
    else:
        # Look at file distribution
        py_count = len(list(path.glob("**/*.py")))
        js_count = len(list(path.glob("**/*.js"))) + len(list(path.glob("**/*.ts")))
        suffix = ".py" if py_count >= js_count else ".js"

    # Try specialized tools first
    if suffix == ".py":
        result = analyze_python_with_pyan(path)
        if result and result.entities:
            return result

    # Try ctags
    result = analyze_with_ctags(path)
    if result and result.entities:
        return result

    # Fall back to simple regex
    if suffix == ".py":
        return analyze_python_simple(path)
    elif suffix in (".js", ".ts", ".jsx", ".tsx"):
        return analyze_javascript_simple(path)
    else:
        # Try both
        py_result = analyze_python_simple(path)
        js_result = analyze_javascript_simple(path)
        if len(py_result.entities) >= len(js_result.entities):
            return py_result
        return js_result


def structure_to_context(structure: CodeStructure) -> dict:
    """Convert code structure to context for the AI prompt."""
    # Group entities by type
    by_type: dict[str, list[str]] = {}
    for entity in structure.entities:
        if entity.entity_type not in by_type:
            by_type[entity.entity_type] = []
        by_type[entity.entity_type].append(entity.name)

    # Summarize relations
    relation_summary: dict[str, int] = {}
    for rel in structure.relations:
        if rel.relation_type not in relation_summary:
            relation_summary[rel.relation_type] = 0
        relation_summary[rel.relation_type] += 1

    return {
        "language": structure.language,
        "analysis_method": structure.analysis_method,
        "entities": {
            etype: names[:20]  # Limit for prompt size
            for etype, names in by_type.items()
        },
        "entity_counts": {etype: len(names) for etype, names in by_type.items()},
        "relation_counts": relation_summary,
        "sample_relations": [
            {"from": r.source, "to": r.target, "type": r.relation_type}
            for r in structure.relations[:10]
        ],
    }


# Example usage
if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    print(f"Analyzing: {path}")
    structure = analyze_code(path)

    print(f"\nLanguage: {structure.language}")
    print(f"Method: {structure.analysis_method}")
    print(f"Entities: {len(structure.entities)}")
    print(f"Relations: {len(structure.relations)}")

    # Group by type
    by_type: dict[str, list] = {}
    for e in structure.entities:
        if e.entity_type not in by_type:
            by_type[e.entity_type] = []
        by_type[e.entity_type].append(e.name)

    print("\nEntities by type:")
    for etype, names in by_type.items():
        print(f"  {etype}: {len(names)}")
        for name in names[:5]:
            print(f"    - {name}")
        if len(names) > 5:
            print(f"    ... and {len(names) - 5} more")

    print("\nSample relations:")
    for rel in structure.relations[:10]:
        print(f"  {rel.source} --{rel.relation_type}--> {rel.target}")
