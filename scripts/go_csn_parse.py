"""
Tree-sitter helpers for Go CodeSearchNet snippets (single function per row).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_go as tsgo
from tree_sitter import Language, Node, Parser, Tree

_FUNC_DECL_TYPES = frozenset({"function_declaration", "method_declaration"})


@dataclass(frozen=True)
class GoFunction:
    node: Node
    name: str
    start_byte: int
    end_byte: int
    start_line: int


def _go_language() -> Language:
    return Language(tsgo.language())


@lru_cache(maxsize=1)
def go_parser() -> Parser:
    return Parser(_go_language())


def parse_go(code: str) -> Tree:
    return go_parser().parse(bytes(code, "utf-8"))


def build_parent_map(root: Node) -> dict[Node, Node | None]:
    parents: dict[Node, Node | None] = {}

    def visit(node: Node, parent: Node | None) -> None:
        parents[node] = parent
        for i in range(node.child_count):
            visit(node.child(i), node)

    visit(root, None)
    return parents


def function_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is None:
        return None
    return name.text.decode("utf-8")


def iter_top_level_functions(root: Node) -> Iterator[GoFunction]:
    for i in range(root.child_count):
        child = root.child(i)
        if child.type not in _FUNC_DECL_TYPES:
            continue
        name = function_name(child)
        if not name:
            continue
        yield GoFunction(
            node=child,
            name=name,
            start_byte=child.start_byte,
            end_byte=child.end_byte,
            start_line=child.start_point[0] + 1,
        )


def inside_nested_function(
    fn: GoFunction, node: Node, parents: dict[Node, Node | None]
) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not fn.node:
        if cur.type in (*_FUNC_DECL_TYPES, "func_literal"):
            return True
        cur = parents.get(cur)
    return False


def identifier_nodes_in(node: Node) -> Iterator[Node]:
    if node.type == "identifier":
        yield node
    for i in range(node.child_count):
        yield from identifier_nodes_in(node.child(i))


def _names_from_expression_list(node: Node) -> list[str]:
    if node.type != "expression_list":
        return []
    out: list[str] = []
    for i in range(node.child_count):
        child = node.child(i)
        if child.type == "identifier":
            out.append(child.text.decode("utf-8"))
    return out


def assignment_target_names(node: Node) -> list[str]:
    left = node.child_by_field_name("left")
    if left is not None:
        return _names_from_expression_list(left)
    if node.type in {"short_var_declaration", "assignment_statement"}:
        for i in range(node.child_count):
            child = node.child(i)
            if child.type == "expression_list":
                return _names_from_expression_list(child)
    return []
