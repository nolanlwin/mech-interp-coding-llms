"""
Repository-level train/validation/test split and frozen Dataset v0 export.

split-repo: assign each GitHub repository to exactly one split (default 80/10/10),
            reproducible with --seed.

freeze:     join cleaned labels to canonical snippets, attach variable occurrences
            and full-function code, partition rows by repo split, write manifest
            with file hashes and git revision.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from clean_boolean_labels import (  # noqa: E402
    _build_canonical_db,
    _find_top_level_function,
    _lookup_code_by_idx,
    _lookup_code_by_repo_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V0_DIR = PROJECT_ROOT / "outputs" / "dataset_v0"
DOCS_DATASET_V0 = PROJECT_ROOT / "docs" / "dataset_v0.md"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _unique_repos_from_labels(labels_path: Path) -> list[str]:
    repos: set[str] = set()
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("parse_error"):
                continue
            r = row.get("repo")
            if r:
                repos.add(r)
    return sorted(repos)


def assign_repo_to_split(
    repos: list[str],
    *,
    seed: int,
    train_p: float = 0.8,
    val_p: float = 0.1,
    test_p: float = 0.1,
) -> dict[str, str]:
    if abs(train_p + val_p + test_p - 1.0) > 1e-6:
        raise ValueError("train, val, test fractions must sum to 1.0")
    n = len(repos)
    if n == 0:
        return {}
    rng = random.Random(seed)
    order = list(repos)
    rng.shuffle(order)
    n_train = int(train_p * n)
    n_val = int(val_p * n)
    n_test = n - n_train - n_val
    out: dict[str, str] = {}
    i = 0
    for repo in order[:n_train]:
        out[repo] = "train"
    i += n_train
    for repo in order[i : i + n_val]:
        out[repo] = "validation"
    i += n_val
    for repo in order[i : i + n_test]:
        out[repo] = "test"
    return out


def cmd_split_repo(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repos = _unique_repos_from_labels(labels_path)
    mapping = assign_repo_to_split(
        repos,
        seed=args.seed,
        train_p=args.train_fraction,
        val_p=args.val_fraction,
        test_p=args.test_fraction,
    )
    split_path = out_dir / "repo_split.jsonl"
    with split_path.open("w", encoding="utf-8") as f:
        for repo in sorted(mapping):
            f.write(
                json.dumps({"repo": repo, "split": mapping[repo]}, ensure_ascii=False)
                + "\n"
            )
    counts = Counter(mapping.values())
    summary = {
        "labels_path": str(labels_path.resolve()),
        "unique_repositories": len(repos),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "repositories_per_split": dict(counts),
        "repo_split_jsonl": str(split_path.resolve()),
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def _occurrences_in_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, var: str
) -> list[dict[str, Any]]:
    occ: list[dict[str, Any]] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == var:
            occ.append(
                {
                    "line": node.lineno,
                    "col": node.col_offset,
                    "ctx": type(node.ctx).__name__,
                }
            )
        elif isinstance(node, ast.arg) and node.arg == var:
            occ.append(
                {
                    "line": node.lineno,
                    "col": node.col_offset,
                    "ctx": "arg",
                }
            )
    occ.sort(key=lambda d: (d["line"], d["col"] if d["col"] is not None else 0))
    return occ


def cmd_freeze(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels)
    canonical_path = Path(args.canonical)
    split_path = Path(args.repo_split)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_path.is_file():
        print(
            f"error: repo split file not found: {split_path}\n\n"
            "Run split-repo first (same --output-dir is typical), for example:\n"
            f"  uv run python scripts/dataset_v0.py split-repo \\\n"
            f"    --labels {labels_path} \\\n"
            f"    --output-dir {out_dir} \\\n"
            f"    --seed 42\n",
            file=sys.stderr,
        )
        return 1

    repo_split: dict[str, str] = {}
    with split_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            repo_split[o["repo"]] = o["split"]

    paths_out = {
        "train": out_dir / "boolean_flags_v0_train.jsonl",
        "validation": out_dir / "boolean_flags_v0_validation.jsonl",
        "test": out_dir / "boolean_flags_v0_test.jsonl",
    }
    writers: dict[str, Any] = {}
    skipped: Counter[str] = Counter()
    written: Counter[str] = Counter()
    conn: sqlite3.Connection | None = None
    db_path: str | None = None

    try:
        for split_name, p in paths_out.items():
            writers[split_name] = p.open("w", encoding="utf-8")

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            db_path = tmp.name
        conn = sqlite3.connect(db_path)
        _build_canonical_db(conn, canonical_path, args.max_canonical_rows)

        parse_cache: dict[tuple[Any, ...], ast.Module] = {}

        with labels_path.open(encoding="utf-8") as fin:
            for line in tqdm(fin, desc="freeze v0"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("parse_error"):
                    skipped["parse_error"] += 1
                    continue
                if rec.get("role") != "boolean_flag":
                    skipped["non_boolean"] += 1
                    continue
                repo = rec.get("repo")
                path = rec.get("path")
                fn_name = rec.get("function")
                var = rec.get("variable")
                if not repo or not path or not fn_name or not var:
                    skipped["missing_fields"] += 1
                    continue

                split = repo_split.get(repo)
                if split is None:
                    skipped["repo_not_in_split"] += 1
                    continue

                sr = rec.get("source_row")
                idx: int | None = None
                if sr is not None:
                    try:
                        idx = int(sr)
                    except (TypeError, ValueError):
                        skipped["bad_source_row"] += 1
                        continue
                    code = _lookup_code_by_idx(conn, idx)
                    cache_key: tuple[Any, ...] = ("idx", idx)
                else:
                    code = _lookup_code_by_repo_path(conn, repo, path)
                    cache_key = ("rp", repo, path)

                if code is None:
                    skipped["missing_code"] += 1
                    continue

                tree = parse_cache.get(cache_key)
                if tree is None:
                    try:
                        tree = ast.parse(code, mode="exec")
                    except SyntaxError:
                        skipped["bad_code"] += 1
                        continue
                    parse_cache[cache_key] = tree

                fn = _find_top_level_function(tree, fn_name)
                if fn is None:
                    skipped["no_function"] += 1
                    continue

                occ = _occurrences_in_function(fn, var)
                frozen: dict[str, Any] = {
                    "repo": repo,
                    "path": path,
                    "function": fn_name,
                    "variable": var,
                    "role": "boolean_flag",
                    "occurrences": occ,
                    "code": code,
                    "split": split,
                }
                if idx is not None:
                    frozen["source_row"] = idx

                writers[split].write(json.dumps(frozen, ensure_ascii=False) + "\n")
                written[split] += 1
    finally:
        if conn is not None:
            conn.close()
        if db_path is not None:
            Path(db_path).unlink(missing_ok=True)
        for w in writers.values():
            w.close()

    manifest: dict[str, Any] = {
        "schema": "dataset_v0_boolean_flags",
        "git_revision": _git_rev(),
        "inputs": {
            "labels": {
                "path": str(labels_path.resolve()),
                "sha256": _sha256_file(labels_path),
            },
            "canonical": {
                "path": str(canonical_path.resolve()),
                "sha256": _sha256_file(canonical_path),
            },
            "repo_split": {
                "path": str(split_path.resolve()),
                "sha256": _sha256_file(split_path),
            },
        },
        "outputs": {
            k: {"path": str(v.resolve()), "sha256": _sha256_file(v)}
            for k, v in paths_out.items()
            if v.exists()
        },
        "scripts_sha256": {
            str(p.relative_to(PROJECT_ROOT)): _sha256_file(p)
            for p in (
                PROJECT_ROOT / "scripts" / "boolean_flag_roles.py",
                PROJECT_ROOT / "scripts" / "clean_boolean_labels.py",
                PROJECT_ROOT / "scripts" / "codesearchnet_python.py",
                PROJECT_ROOT / "scripts" / "csn_function_ast.py",
                PROJECT_ROOT / "scripts" / "dataset_v0.py",
            )
            if p.is_file()
        },
        "row_counts_by_split": dict(written),
        "skipped": dict(skipped),
    }
    (out_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"written": dict(written), "skipped": dict(skipped)}, indent=2))
    print(f"manifest -> {out_dir / 'freeze_manifest.json'}")

    _write_dataset_docs(out_dir, manifest)
    print(f"documentation -> {DOCS_DATASET_V0}")

    return 0


def _write_dataset_docs(out_dir: Path, manifest: dict[str, Any]) -> None:
    DOCS_DATASET_V0.parent.mkdir(parents=True, exist_ok=True)
    rev = manifest.get("git_revision") or "unknown"
    content = f"""# Dataset v0 — boolean control variables (CodeSearchNet Python)

This document describes the **frozen** boolean-flag slice used for mechanistic
interpretation experiments. Regenerate artifacts with `scripts/dataset_v0.py`
after reproducing upstream JSONL with the same script versions recorded in
`outputs/dataset_v0/freeze_manifest.json` (script SHA-256 digests and git revision).

## Role definition

- **Role:** `boolean_flag`
- **Intent:** Names used primarily as **conditional control signals** in Python
  (truthiness tests, `not` / `and` / `or` in conditions, bool literal assignments,
  and comparisons of a name to `True`/`False` with `==`, `!=`, `is`, `is not`).
- **Scope:** Top-level `def` / `async def` in each CodeSearchNet snippet; inner
  nested function bodies are excluded from the outer function’s extraction.

## Cleaning exclusions (before split)

Rows in the cleaned label file omit examples that:

- Fail Python parse or lack a matching top-level function name.
- Fall below **five** non-empty source lines for the snippet.
- Show fewer than **two** AST occurrences of the variable (including parameters).
- Match lightweight **autogenerated** or **notebook / IPython** heuristics.
- Duplicate the same `(source_row, function, variable)` key (when `source_row`
  is present).

## Train / validation / test split

- **Unit of split:** GitHub `repository` string (not per-function randomization).
- **Default ratios:** 80% train, 10% validation, 10% test.
- **Shuffle:** Repositories are shuffled with a fixed `--seed` before contiguous
  assignment so the split is reproducible.

## Frozen record schema

Each line in `boolean_flags_v0_*.jsonl` is one JSON object:

- `repo`, `path`, `function`, `variable`, `role`
- `occurrences`: sorted list of `{{line, col, ctx}}` for `ast.Name` and `arg`
- `code`: full canonical snippet for that label (same as Hub export row)
- `split`: `train` | `validation` | `test`
- `source_row`: optional 1-based canonical line index when present on the label

## Reproducibility

- **Git revision at freeze:** `{rev}`
- **Manifest path:** `{out_dir / "freeze_manifest.json"}`
- **Hashes:** SHA-256 of inputs and output shards are stored in the manifest.

## Known limitations

- Heuristic boolean detection misses typed or call-based conditions and can
  include generic short names (`i`, `ok`) that pass occurrence filters.
- Repository-level split does not remove **near-duplicate** repositories (only
  exact repo string collision is handled by construction).
- CodeSearchNet repeats `(repo, path)` across rows; labels and freeze rely on
  `source_row` for correct alignment with canonical order.
"""
    DOCS_DATASET_V0.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset v0: repo split + freeze export.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split-repo", help="Write repo_split.jsonl from cleaned labels.")
    sp.add_argument("--labels", type=str, required=True)
    sp.add_argument("--output-dir", type=str, default=str(DEFAULT_V0_DIR))
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--train-fraction", type=float, default=0.8)
    sp.add_argument("--val-fraction", type=float, default=0.1)
    sp.add_argument("--test-fraction", type=float, default=0.1)
    sp.set_defaults(func=cmd_split_repo)

    fr = sub.add_parser("freeze", help="Write frozen split JSONL + freeze_manifest.json.")
    fr.add_argument("--labels", type=str, required=True, help="Cleaned boolean JSONL.")
    fr.add_argument("--canonical", type=str, required=True)
    fr.add_argument(
        "--repo-split",
        type=str,
        required=True,
        help="Path to repo_split.jsonl (create with: dataset_v0.py split-repo …).",
    )
    fr.add_argument("--output-dir", type=str, default=str(DEFAULT_V0_DIR))
    fr.add_argument("--max-canonical-rows", type=int, default=None)
    fr.set_defaults(func=cmd_freeze)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
