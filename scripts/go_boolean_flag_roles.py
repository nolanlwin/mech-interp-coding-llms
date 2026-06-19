"""
Boolean flags extraction for Go (tree-sitter).

Mirrors the Python/Java heuristics:
- if / for tests
- short declarations and assignments to true/false, boolean operators, or ``!name``
- return of boolean-shaped expressions
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from tree_sitter import Node

from go_csn_parse import (
    GoFunction,
    assignment_target_names,
    build_parent_map,
    identifier_nodes_in,
    inside_nested_function,
    iter_top_level_functions,
    parse_go,
)


@dataclass
class GoFlagHit:
    variable: str
    line: int
    pattern: str
    node: Node


_BOOL_LITERAL_TYPES = frozenset({"true", "false"})
_BOOL_BIN_OPS = frozenset({"&&", "||"})
_BOOL_COMPARE_OPS = frozenset({"==", "!="})


def _node_text(code: str, node: Node) -> str:
    return code[node.start_byte : node.end_byte]


def _identifier_name(node: Node) -> str | None:
    if node.type != "identifier":
        return None
    return node.text.decode("utf-8")


def _is_bool_literal(node: Node) -> bool:
    if node.type in _BOOL_LITERAL_TYPES:
        return True
    if node.type == "expression_list" and node.child_count == 1:
        return _is_bool_literal(node.child(0))
    return False


def _unwrap_expression(node: Node) -> Node:
    if node.type == "expression_list" and node.child_count == 1:
        return node.child(0)
    return node


def _binary_operator(node: Node) -> str | None:
    for i in range(node.child_count):
        child = node.child(i)
        if child.type in _BOOL_BIN_OPS | _BOOL_COMPARE_OPS:
            return child.type
    return None


def compare_boolean_flag_names(node: Node) -> set[str]:
    if node.type != "binary_expression":
        return set()
    op = _binary_operator(node)
    if op not in _BOOL_COMPARE_OPS:
        return set()
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
    if left is None or right is None:
        return set()
    out: set[str] = set()
    for a, b in ((left, right), (right, left)):
        name = _identifier_name(a)
        if name and _is_bool_literal(b):
            out.add(name)
    return out


def names_in_boolean_test(node: Node) -> set[str]:
    if node.type == "identifier":
        name = _identifier_name(node)
        return {name} if name else set()
    if node.type == "unary_expression":
        op = node.child(0)
        operand = node.child_by_field_name("argument") or (
            node.child(1) if node.child_count > 1 else None
        )
        if op is not None and op.type == "!" and operand is not None:
            return names_in_boolean_test(operand)
        return set()
    if node.type == "binary_expression" and _binary_operator(node) in _BOOL_BIN_OPS:
        out: set[str] = set()
        for sub in (node.child_by_field_name("left"), node.child_by_field_name("right")):
            if sub is not None:
                out |= names_in_boolean_test(sub)
        return out
    if node.type == "parenthesized_expression":
        for i in range(node.child_count):
            child = node.child(i)
            if child.type not in {"(", ")"}:
                return names_in_boolean_test(child)
        return set()
    if node.type == "binary_expression":
        return compare_boolean_flag_names(node)
    return set()


def bool_expression_load_names(node: Node) -> set[str]:
    if node.type == "identifier":
        name = _identifier_name(node)
        return {name} if name else set()
    if node.type == "unary_expression":
        op = node.child(0)
        operand = node.child_by_field_name("argument") or (
            node.child(1) if node.child_count > 1 else None
        )
        if op is not None and op.type == "!" and operand is not None:
            return bool_expression_load_names(operand)
        return set()
    if node.type == "binary_expression" and _binary_operator(node) in _BOOL_BIN_OPS:
        out: set[str] = set()
        for sub in (node.child_by_field_name("left"), node.child_by_field_name("right")):
            if sub is not None:
                out |= bool_expression_load_names(sub)
        return out
    if node.type == "parenthesized_expression":
        for i in range(node.child_count):
            child = node.child(i)
            if child.type not in {"(", ")"}:
                return bool_expression_load_names(child)
        return set()
    if node.type == "binary_expression":
        return compare_boolean_flag_names(node)
    return set()


def _if_condition(node: Node) -> Node | None:
    if node.type != "if_statement":
        return None
    for i in range(node.child_count):
        child = node.child(i)
        if child.type not in {"if", "else", "block"}:
            return child
    return None


def _for_condition(node: Node) -> Node | None:
    if node.type != "for_statement":
        return None
    for i in range(node.child_count):
        child = node.child(i)
        if child.type == "for_clause":
            for j in range(child.child_count):
                part = child.child(j)
                if part.type == "binary_expression":
                    return part
                if part.type in {"identifier", "unary_expression", "parenthesized_expression"}:
                    return part
            return None
        if child.type not in {"for", "block"}:
            return child
    return None


def hits_from_short_var(node: Node, code: str) -> list[GoFlagHit]:
    hits: list[GoFlagHit] = []
    line = node.start_point[0] + 1
    value = node.child_by_field_name("right")
    if value is None:
        return []
    value = _unwrap_expression(value)
    targets = assignment_target_names(node)
    if _is_bool_literal(value):
        for vid in targets:
            hits.append(GoFlagHit(vid, line, "assign_bool_literal", node))
    elif _binary_operator(value) in _BOOL_BIN_OPS:
        names_rhs = bool_expression_load_names(value)
        for vid in targets:
            hits.append(GoFlagHit(vid, line, "assign_boolop_lhs", node))
        for vid in names_rhs:
            hits.append(GoFlagHit(vid, line, "assign_boolop_rhs", value))
    elif value.type == "unary_expression":
        op = value.child(0)
        operand = value.child_by_field_name("argument") or (
            value.child(1) if value.child_count > 1 else None
        )
        if op is not None and op.type == "!" and operand is not None:
            inner_name = _identifier_name(operand)
            if inner_name:
                for vid in targets:
                    hits.append(GoFlagHit(vid, line, "assign_not_name", node))
                hits.append(GoFlagHit(inner_name, line, "assign_not_name_inner", value))
    return hits


def hits_from_assignment(node: Node, code: str) -> list[GoFlagHit]:
    if node.type != "assignment_statement":
        return []
    hits: list[GoFlagHit] = []
    line = node.start_point[0] + 1
    value = node.child_by_field_name("right")
    if value is None:
        return []
    value = _unwrap_expression(value)
    targets = assignment_target_names(node)
    if _is_bool_literal(value):
        for vid in targets:
            hits.append(GoFlagHit(vid, line, "assign_bool_literal", node))
    elif _binary_operator(value) in _BOOL_BIN_OPS:
        names_rhs = bool_expression_load_names(value)
        for vid in targets:
            hits.append(GoFlagHit(vid, line, "assign_boolop_lhs", node))
        for vid in names_rhs:
            hits.append(GoFlagHit(vid, line, "assign_boolop_rhs", value))
    elif value.type == "unary_expression":
        op = value.child(0)
        operand = value.child_by_field_name("argument") or (
            value.child(1) if value.child_count > 1 else None
        )
        if op is not None and op.type == "!" and operand is not None:
            inner_name = _identifier_name(operand)
            if inner_name:
                for vid in targets:
                    hits.append(GoFlagHit(vid, line, "assign_not_name", node))
                hits.append(GoFlagHit(inner_name, line, "assign_not_name_inner", value))
    return hits


def _iter_if_tests(node: Node) -> Iterator[Node]:
    if node.type != "if_statement":
        return
    cond = _if_condition(node)
    if cond is not None:
        yield cond
    for i in range(node.child_count):
        child = node.child(i)
        if child.type == "else":
            for j in range(child.child_count):
                sub = child.child(j)
                if sub.type == "if_statement":
                    yield from _iter_if_tests(sub)


def collect_flag_hits(fn: GoFunction, code: str) -> list[GoFlagHit]:
    parents = build_parent_map(fn.node)
    hits: list[GoFlagHit] = []

    def walk(node: Node) -> None:
        if node is fn.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_function(fn, node, parents):
            return
        if node.type == "if_statement":
            for test in _iter_if_tests(node):
                for vid in names_in_boolean_test(test):
                    hits.append(GoFlagHit(vid, test.start_point[0] + 1, "if_test", test))
        elif node.type == "for_statement":
            test = _for_condition(node)
            if test is not None:
                for vid in names_in_boolean_test(test):
                    hits.append(GoFlagHit(vid, test.start_point[0] + 1, "while_test", test))
        elif node.type == "short_var_declaration":
            hits.extend(hits_from_short_var(node, code))
        elif node.type == "assignment_statement":
            hits.extend(hits_from_assignment(node, code))
        for i in range(node.child_count):
            walk(node.child(i))

    walk(fn.node)
    return hits


def collect_return_hits(fn: GoFunction, code: str) -> list[GoFlagHit]:
    parents = build_parent_map(fn.node)
    hits: list[GoFlagHit] = []

    def walk(node: Node) -> None:
        if node is fn.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_function(fn, node, parents):
            return
        if node.type == "return_statement":
            value = node.child_by_field_name("result")
            if value is None:
                for i in range(node.child_count):
                    child = node.child(i)
                    if child.type not in {"return"}:
                        value = child
                        break
            if value is not None:
                value = _unwrap_expression(value)
                allowed = names_in_boolean_test(value)
                if allowed:
                    for sub in identifier_nodes_in(value):
                        name = _identifier_name(sub)
                        if name and name in allowed:
                            hits.append(
                                GoFlagHit(name, sub.start_point[0] + 1, "return_bool", sub)
                            )
        for i in range(node.child_count):
            walk(node.child(i))

    walk(fn.node)
    return hits


def extract_boolean_flags(fn: GoFunction, code: str) -> list[dict[str, Any]]:
    hits = collect_flag_hits(fn, code)
    by_var: dict[str, list[GoFlagHit]] = defaultdict(list)
    for h in hits:
        by_var[h.variable].append(h)

    out: list[dict[str, Any]] = []
    for var in sorted(by_var):
        hs = by_var[var]
        hs.sort(key=lambda h: (h.line, h.pattern))
        first = hs[0]
        snippet = _node_text(code, first.node).strip().splitlines()
        out.append(
            {
                "variable": var,
                "role": "boolean_flag",
                "line": first.line,
                "code": snippet[0] if snippet else "",
                "function": fn.name,
            }
        )
    return out


def labeled_rows_from_go_code(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    tree = parse_go(code)
    if tree.root_node.has_error:
        return [], "go parse error"

    funcs = list(iter_top_level_functions(tree.root_node))
    if not funcs:
        return [], "no top-level function declaration"

    rows: list[dict[str, Any]] = []
    for fn in funcs:
        for ex in extract_boolean_flags(fn, code):
            if repo is not None:
                ex["repo"] = repo
            if path is not None:
                ex["path"] = path
            if source_row is not None:
                ex["source_row"] = source_row
            rows.append(ex)
    return rows, None
