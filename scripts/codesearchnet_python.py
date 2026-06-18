"""
CodeSearchNet (Python): backward-compatible wrapper around ``codesearchnet.py``.

For other languages use:
  uv run python scripts/codesearchnet.py download --language java
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from codesearchnet import download_split_to_jsonl, iter_canonical_jsonl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "codesearchnet_python"


def download_python_split_to_jsonl(
    *,
    split: str,
    output_dir: Path,
    max_rows: int | None = None,
    cache_dir: str | None = None,
    code_field: str = "whole",
    include_docstring: bool = False,
) -> Path:
    return download_split_to_jsonl(
        language="python",
        split=split,
        output_dir=output_dir,
        max_rows=max_rows,
        cache_dir=cache_dir,
        code_field=code_field,
        include_docstring=include_docstring,
    )


def _cmd_download(args: argparse.Namespace) -> int:
    out = download_python_split_to_jsonl(
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
    p = argparse.ArgumentParser(description="CodeSearchNet Python: download + JSONL loader.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Stream split to JSONL (canonical schema).")
    d.add_argument("--split", default="train", choices=["train", "validation", "test"])
    d.add_argument("--output-dir", type=str, default=str(DEFAULT_DATA_DIR))
    d.add_argument("--max-rows", type=int, default=None)
    d.add_argument("--cache-dir", type=str, default=None)
    d.add_argument("--code-field", choices=["whole", "body"], default="whole")
    d.add_argument("--include-docstring", action="store_true")
    d.set_defaults(func=_cmd_download)

    pk = sub.add_parser("peek", help="Print the first N JSONL records as pretty JSON.")
    pk.add_argument("--file", type=str, required=True)
    pk.add_argument("--n", type=int, default=3)
    pk.set_defaults(func=_cmd_peek)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
