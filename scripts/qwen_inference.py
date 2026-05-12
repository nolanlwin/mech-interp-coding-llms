"""
Qwen2.5-1.5B inference setup: load model with hidden states and attentions,
run a forward pass on source text, verify layer coverage, and save a small
JSON diagnostic (token ids, decoded pieces, tensor shapes).

Requires PyTorch and Hugging Face Transformers; first run downloads weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "qwen_inference"
DEFAULT_CODE_FILE = PROJECT_ROOT / "fixtures" / "minimal_forward.py"


def _accelerate_installed() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def resolve_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, prefer: str) -> torch.dtype:
    if prefer == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
    if prefer == "fp16":
        if device.type in ("cuda", "mps"):
            return torch.float16
    return torch.float32


def load_causal_lm_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    eager_attn: bool = False,
) -> tuple[Any, Any]:
    """Load ``AutoTokenizer`` and ``AutoModelForCausalLM`` (``trust_remote_code=True``)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    load_kw: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if eager_attn:
        load_kw["attn_implementation"] = "eager"
    use_device_map = device.type == "cuda" and _accelerate_installed()
    if use_device_map:
        load_kw["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    except TypeError:
        load_kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    if not use_device_map:
        model = model.to(device)
    return model, tok


def forward_hidden_cached(
    model: Any,
    tok: Any,
    code: str,
    *,
    max_length: int = 2048,
    output_attentions: bool = False,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """
    Run one forward with ``output_hidden_states=True`` using an already-loaded model.

    Returns ``hidden_states`` (tuple length ``num_hidden_layers + 1``) and metadata.
    """
    enc = tok(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=False,
    )
    in_dev = _model_input_device(model)
    input_ids = enc["input_ids"].to(in_dev)
    att_mask = enc.get("attention_mask")
    if att_mask is not None:
        att_mask = att_mask.to(in_dev)

    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", 0))
    hidden_size = int(getattr(cfg, "hidden_size", 0))

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=att_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            use_cache=False,
        )
    hs = out.hidden_states
    meta: dict[str, Any] = {
        "seq_len": int(input_ids.shape[1]),
        "num_hidden_layers": n_layers,
        "hidden_size": hidden_size,
        "num_hidden_tensors": len(hs),
    }
    return hs, meta


@dataclass
class ForwardSummary:
    model_id: str
    device: str
    dtype: str
    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    seq_len: int
    input_ids: list[int]
    decoded_tokens: list[str]
    offset_mapping: list[list[int]] | None
    hidden_state_shapes: list[list[int | None]]
    attention_shapes: list[list[int | None]] | None
    layer_count_ok: bool
    expected_hidden_tuple_len: int
    actual_hidden_tuple_len: int
    notes: list[str]


def _tensor_shape(t: torch.Tensor) -> list[int]:
    return list(t.shape)


def _model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def forward_code(
    *,
    code: str,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 2048,
) -> tuple[ForwardSummary, dict[str, Any]]:
    notes: list[str] = []
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not tok.is_fast:
        notes.append("Tokenizer is slow; offset_mapping unavailable.")

    enc = tok(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=tok.is_fast,
    )
    offset_mapping: list[list[int]] | None = None
    if tok.is_fast and enc.get("offset_mapping") is not None:
        offset_mapping = enc["offset_mapping"][0].tolist()

    load_kw: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
        # SDPA / flash path does not materialize attention weights; Day 11 expects them.
        "attn_implementation": "eager",
    }
    use_device_map = device.type == "cuda" and _accelerate_installed()
    if use_device_map:
        load_kw["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    except TypeError:
        load_kw.pop("attn_implementation", None)
        notes.append(
            "from_pretrained rejected attn_implementation; loaded without (attentions may be empty with sdpa)."
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    if not use_device_map:
        model = model.to(device)

    in_dev = _model_input_device(model)
    input_ids = enc["input_ids"].to(in_dev)
    att_mask = enc.get("attention_mask")
    if att_mask is not None:
        att_mask = att_mask.to(in_dev)

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=att_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    hs = out.hidden_states
    att = out.attentions
    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", 0))
    expected_hs = n_layers + 1

    hs_shapes = [_tensor_shape(t) for t in hs]
    att_shapes: list[list[int | None]] = []
    if att is not None:
        att_shapes = [_tensor_shape(t) for t in att if t is not None]
    if not att_shapes:
        notes.append(
            "output_attentions produced no tensors (eager attention may be unsupported on this stack)."
        )

    ids_list = input_ids[0].tolist()
    decoded = tok.convert_ids_to_tokens(ids_list)

    summary = ForwardSummary(
        model_id=model_id,
        device=str(device),
        dtype=str(dtype),
        num_hidden_layers=n_layers,
        num_attention_heads=int(getattr(cfg, "num_attention_heads", 0)),
        hidden_size=int(getattr(cfg, "hidden_size", 0)),
        seq_len=int(input_ids.shape[1]),
        input_ids=ids_list,
        decoded_tokens=decoded,
        offset_mapping=offset_mapping,
        hidden_state_shapes=hs_shapes,
        attention_shapes=att_shapes if att_shapes else None,
        layer_count_ok=len(hs) == expected_hs,
        expected_hidden_tuple_len=expected_hs,
        actual_hidden_tuple_len=len(hs),
        notes=notes,
    )

    raw: dict[str, Any] = {
        "logits_shape": list(out.logits.shape) if out.logits is not None else None,
    }
    return summary, raw


def forward_hidden_states(
    *,
    code: str,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 2048,
    output_attentions: bool = False,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """
    Single causal forward with ``output_hidden_states=True``.

    Returns ``hidden_states`` (length ``num_hidden_layers + 1``, each ``[1, seq, H]`` on
    ``device``) and a small metadata dict. Loads a fresh model for this call (use
    ``load_causal_lm_tokenizer`` + ``forward_hidden_cached`` in a loop to reuse weights).
    """
    model, tok = load_causal_lm_tokenizer(
        model_id, device, dtype, eager_attn=output_attentions
    )
    return forward_hidden_cached(
        model, tok, code, max_length=max_length, output_attentions=output_attentions
    )


def cmd_verify(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    cf = args.code_file
    if cf == "-":
        code = sys.stdin.read()
    else:
        code = Path(cf).read_text(encoding="utf-8")

    summary, raw = forward_code(
        code=code,
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        max_length=args.max_length,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forward_verify.json"
    payload = {**asdict(summary), "logits_shape": raw.get("logits_shape")}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in (
        "model_id", "device", "dtype", "seq_len",
        "num_hidden_layers", "expected_hidden_tuple_len",
        "actual_hidden_tuple_len", "layer_count_ok",
        "hidden_state_shapes", "attention_shapes",
    )}, indent=2))
    print(f"wrote {out_path}")
    if not summary.layer_count_ok:
        print(
            "warning: hidden_states tuple length does not match num_hidden_layers+1",
            file=sys.stderr,
        )
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen2.5-1.5B forward pass with hidden states and attentions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser(
        "verify",
        help="Forward UTF-8 source (default: fixtures/minimal_forward.py; '-' = stdin) and save JSON diagnostics.",
    )
    v.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    v.add_argument("--device", type=str, default=None, help="cpu | cuda | mps | cuda:0 … (default: auto)")
    v.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    v.add_argument("--max-length", type=int, default=2048)
    v.add_argument(
        "--code-file",
        type=str,
        default=str(DEFAULT_CODE_FILE),
        help=(
            "UTF-8 source file to forward, or '-' for stdin. "
            f"Default: {DEFAULT_CODE_FILE.relative_to(PROJECT_ROOT)} (tiny neutral snippet)."
        ),
    )
    v.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
