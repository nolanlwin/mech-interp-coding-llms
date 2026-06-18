"""
CodeSearchNet: download from the Hugging Face Hub and load canonical JSONL.

Uses the `datasets` mirror (`code_search_net`, per-language config) so `uv run` is enough.

Canonical record:
  {"repo": "...", "path": "...", "code": "..."}

Supported languages: go, java, javascript, php, python, ruby
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset, load_dataset_builder
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSN_LANGUAGES = ("go", "java", "javascript", "php", "python", "ruby")


def default_data_dir(language: str) -> Path:
    return PROJECT_ROOT / "data" / f"codesearchnet_{language}"


def row_to_canonical(
    row: dict,
    *,
    code_field: str = "whole",
    include_docstring: bool = False,
) -> dict:
    if code_field == "body":
        code = row.get("func_code_string") or ""
    elif code_field == "whole":
        code = row.get("whole_func_string") or ""
    else:
        raise ValueError("code_field must be 'whole' or 'body'")

    out: dict = {
        "repo": row.get("repository_name") or "",
        "path": row.get("func_path_in_repository") or "",
        "code": code,
    }
    if include_docstring:
        out["docstring"] = row.get("func_documentation_string") or ""
    return out


def _split_num_examples(language: str, split: str) -> int | None:
    builder = load_dataset_builder("code_search_net", language)
    info_split = builder.info.splits.get(split) if builder.info.splits else None
    return info_split.num_examples if info_split is not None else None


def download_split_to_jsonl(
    *,
    language: str,
    split: str,
    output_dir: Path,
    max_rows: int | None = None,
    cache_dir: str | None = None,
    code_field: str = "whole",
    include_docstring: bool = False,
) -> Path:
    if language not in CSN_LANGUAGES:
        raise ValueError(f"language must be one of {CSN_LANGUAGES}, got {language!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{language}_{split}.jsonl"

    total = _split_num_examples(language, split)
    if max_rows is not None and total is not None:
        total = min(total, max_rows)
    elif max_rows is not None:
        total = max_rows

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    stream = load_dataset(
        "code_search_net",
        language,
        split=split,
        streaming=True,
        **kwargs,
    )

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        it = tqdm(stream, total=total, desc=f"code_search_net/{language}/{split}")
        for row in it:
            rec = row_to_canonical(
                row,
                code_field=code_field,
                include_docstring=include_docstring,
            )
            rec["language"] = language
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if max_rows is not None and written >= max_rows:
                break

    return out_path


def iter_canonical_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cmd_download(args: argparse.Namespace) -> int:
    out = download_split_to_jsonl(
        language=args.language,
        split=args.split,
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
        cache_dir=args.cache_dir,
        code_field=args.code_field,
        include_docstring=args.include_docstring,
    )
    print(str(out))
    return 0


def _cmd_peek(args: argparse.Namespace) -> int:
    path = Path(args.file)
    for i, rec in enumerate(iter_canonical_jsonl(path)):
        if i >= args.n:
            break
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CodeSearchNet: download + JSONL loader.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Stream split to JSONL (canonical schema).")
    d.add_argument(
        "--language",
        default="python",
        choices=list(CSN_LANGUAGES),
        help="Hub config name (default: python).",
    )
    d.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Hub split name (validation, not 'valid').",
    )
    d.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for <language>_<split>.jsonl (default: data/codesearchnet_<language>/).",
    )
    d.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N rows (useful for smoke tests).",
    )
    d.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    d.add_argument(
        "--code-field",
        choices=["whole", "body"],
        default="whole",
        help="'whole' = whole_func_string (def + docstring + body); 'body' = func_code_string.",
    )
    d.add_argument(
        "--include-docstring",
        action="store_true",
        help="Add a top-level docstring field to each JSON object (beyond AGENDA v0 schema).",
    )
    d.set_defaults(func=_cmd_download)

    pk = sub.add_parser("peek", help="Print the first N JSONL records as pretty JSON.")
    pk.add_argument("--file", type=str, required=True)
    pk.add_argument("--n", type=int, default=3)
    pk.set_defaults(func=_cmd_peek)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "download" and args.output_dir is None:
        args.output_dir = str(default_data_dir(args.language))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
