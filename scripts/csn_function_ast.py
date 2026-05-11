"""
Parse Python snippets, extract structure, and expose traversal helpers (parent map, names, assignments, loops).

Input is typically canonical CodeSearchNet JSONL from `codesearchnet_python.py`
(fields: repo, path, code). Output JSON lines:

  {"function_name": "...", "variables": [...], "nodes": [...], ...}
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AST_OUT = PROJECT_ROOT / "outputs" / "ast_parsed"


# --- span / serialization -------------------------------------------------


def _span(node: ast.AST) -> dict[str, int | None]:
    ln = getattr(node, "lineno", None)
    en = getattr(node, "end_lineno", ln)
    co = getattr(node, "col_offset", None)
    ec = getattr(node, "end_col_offset", None)
    return {"lineno": ln, "end_lineno": en, "col_offset": co, "end_col_offset": ec}


def _node_entry(kind: str, node: ast.AST, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "ast_type": type(node).__name__, **_span(node)}
    out.update(extra)
    return out


# --- traversal utilities --------------------------------------------------


def build_parent_map(root: ast.AST) -> dict[ast.AST, ast.AST | None]:
    parents: dict[ast.AST, ast.AST | None] = {}

    def visit(node: ast.AST, parent: ast.AST | None) -> None:
        parents[node] = parent
        for child in ast.iter_child_nodes(node):
            visit(child, node)

    visit(root, None)
    return parents


def find_name_occurrences(root: ast.AST, name: str) -> list[ast.Name]:
    return [n for n in ast.walk(root) if isinstance(n, ast.Name) and n.id == name]


def names_in_assignment_target(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Starred):
        return names_in_assignment_target(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(names_in_assignment_target(elt))
        return out
    return []


def iter_assignment_events(func: ast.AST) -> Iterator[dict[str, Any]]:
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets: list[str] = []
            for t in node.targets:
                targets.extend(names_in_assignment_target(t))
            yield {
                "kind": "assign",
                "targets": targets,
                **_span(node),
            }
        elif isinstance(node, ast.AnnAssign):
            if node.target is not None:
                targets = names_in_assignment_target(node.target)
            else:
                targets = []
            yield {
                "kind": "ann_assign",
                "targets": targets,
                **_span(node),
            }
        elif isinstance(node, ast.AugAssign):
            targets = names_in_assignment_target(node.target)
            yield {
                "kind": "aug_assign",
                "op": type(node.op).__name__,
                "targets": targets,
                **_span(node),
            }


def iter_loop_events(func: ast.AST) -> Iterator[dict[str, Any]]:
    for node in ast.walk(func):
        if isinstance(node, ast.For):
            yield {
                "kind": "for",
                "target_vars": names_in_assignment_target(node.target),
                **_span(node),
            }
        elif isinstance(node, ast.AsyncFor):
            yield {
                "kind": "async_for",
                "target_vars": names_in_assignment_target(node.target),
                **_span(node),
            }
        elif isinstance(node, ast.While):
            yield {"kind": "while", **_span(node)}


def collect_variables(func: ast.AST) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            by_id[node.id].append(
                {
                    "line": node.lineno,
                    "col": node.col_offset,
                    "ctx": type(node.ctx).__name__,
                }
            )
    return [{"name": n, "occurrences": occ} for n, occ in sorted(by_id.items())]


def extract_highlight_nodes(func: ast.AST) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in ast.walk(func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                _node_entry(
                    "function_def",
                    node,
                    name=node.name,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )
            )
        elif isinstance(node, ast.For):
            out.append(
                _node_entry(
                    "for_loop",
                    node,
                    target_vars=names_in_assignment_target(node.target),
                )
            )
        elif isinstance(node, ast.AsyncFor):
            out.append(
                _node_entry(
                    "async_for_loop",
                    node,
                    target_vars=names_in_assignment_target(node.target),
                )
            )
        elif isinstance(node, ast.While):
            out.append(_node_entry("while_loop", node))
        elif isinstance(node, ast.Assign):
            targets: list[str] = []
            for t in node.targets:
                targets.extend(names_in_assignment_target(t))
            out.append(_node_entry("assign", node, targets=targets))
        elif isinstance(node, ast.AnnAssign):
            targets = (
                names_in_assignment_target(node.target) if node.target is not None else []
            )
            out.append(_node_entry("ann_assign", node, targets=targets))
        elif isinstance(node, ast.AugAssign):
            targets = names_in_assignment_target(node.target)
            out.append(
                _node_entry(
                    "aug_assign",
                    node,
                    targets=targets,
                    op=type(node.op).__name__,
                )
            )
        elif isinstance(node, ast.Return):
            out.append(_node_entry("return", node))
        elif isinstance(node, ast.Subscript):
            out.append(_node_entry("subscript", node))
        elif isinstance(node, ast.If):
            out.append(_node_entry("if_stmt", node))
        elif isinstance(node, ast.IfExp):
            out.append(_node_entry("if_exp", node))
    out.sort(
        key=lambda e: (
            e.get("lineno") is None,
            e.get("lineno") or 0,
            e.get("col_offset") or 0,
            e.get("kind", ""),
        )
    )
    return out


def parse_module(code: str) -> ast.Module:
    return ast.parse(code, mode="exec")


def iter_top_level_functions(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield stmt


def function_ast_record(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    repo: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "function_name": func.name,
        "variables": collect_variables(func),
        "nodes": extract_highlight_nodes(func),
        "assignments": list(iter_assignment_events(func)),
        "loops": list(iter_loop_events(func)),
    }
    if repo is not None:
        record["repo"] = repo
    if path is not None:
        record["path"] = path
    return record


def parse_code_to_records(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        tree = parse_module(code)
    except SyntaxError as e:
        return [], f"{e.msg} (line {e.lineno})"

    funcs = list(iter_top_level_functions(tree))
    if not funcs:
        return [], "no top-level function definition in module"

    return [function_ast_record(fn, repo=repo, path=path) for fn in funcs], None


def _cmd_parse(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_rows = args.max_rows
    n_in = 0
    n_out = 0
    n_err = 0

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(desc="parse jsonl", unit="row", total=max_rows)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            pbar.update(1)
            row = json.loads(line)
            code = row.get("code") or ""
            repo = row.get("repo")
            path = row.get("path")
            records, err = parse_code_to_records(code, repo=repo, path=path)
            if err is not None:
                n_err += 1
                err_obj: dict[str, Any] = {
                    "parse_error": err,
                    "function_name": None,
                    "variables": [],
                    "nodes": [],
                }
                if repo is not None:
                    err_obj["repo"] = repo
                if path is not None:
                    err_obj["path"] = path
                fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            else:
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
            if max_rows is not None and n_in >= max_rows:
                break
        pbar.close()

    print(f"read_lines={n_in} written_records={n_out} parse_errors={n_err} -> {out_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Day 2: Python function AST parse + JSONL export.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("parse", help="Parse canonical JSONL (repo,path,code) to AST JSONL.")
    pr.add_argument("--input", type=str, required=True)
    pr.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_AST_OUT / "parsed.jsonl"),
        help=f"Default: {DEFAULT_AST_OUT / 'parsed.jsonl'}",
    )
    pr.add_argument("--max-rows", type=int, default=None)
    pr.set_defaults(func=_cmd_parse)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
