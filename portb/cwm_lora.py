"""Apply the CWM LoRA to the diffusers-format MiniMax-H3 Ref2VA transformer.

The LoRA was trained through musubi-tuner against the official package layout, whose
module names and tensor layouts differ from the diffusers conversion. The differences
were established by comparing the two checkpoints tensor by tensor, not assumed:

  official                   diffusers            relation
  blocks.N.attn.qkv_proj     attn.to_{q,k,v}      head-interleaved split
  blocks.N.attn.out_proj     attn.to_out.0        identical
  blocks.N.mlp.fc1           ff.net.0.proj        the two halves are swapped
  blocks.N.mlp.fc2           ff.net.2             identical

The fused qkv is laid out [n_heads, 3, head_dim, hidden] -- head-interleaved, NOT three
contiguous blocks. Slicing it contiguously looks plausible and is wrong; checked with
torch.equal against the converted weights (contiguous: False, head-interleaved: True).

Both reorderings are row permutations of the delta, and delta = up @ down, so permuting
the rows of `up` is exact and preserves the rank-128 factorisation: nothing is densified
until the caller asks for it.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors.torch import load_file

N_HEADS = 56
HEAD_DIM = 128
FF_HALF = 14336

_RE = re.compile(r"^lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)$")


def _split_qkv_up(up: torch.Tensor, which: int) -> torch.Tensor:
    """Take the q, k or v rows out of a fused-qkv `lora_up` of shape [3*n*d, rank]."""
    rank = up.shape[1]
    return up.view(N_HEADS, 3, HEAD_DIM, rank)[:, which].reshape(-1, rank).contiguous()


def _swap_ff_halves(up: torch.Tensor) -> torch.Tensor:
    """The gated MLP's two halves are stored in the opposite order after conversion."""
    return torch.cat([up[FF_HALF:], up[:FF_HALF]], dim=0).contiguous()


def build_deltas(lora_path: str | Path, *, scale: float = 1.0) -> dict[str, tuple]:
    """Return {diffusers_module_path: (lora_down, lora_up, scale)}.

    Factors stay separate so the caller can fold them into the weight or hook them into
    the forward pass; on a 66 GB model a densified delta per module is not free.
    """
    state = load_file(str(lora_path))
    mods: dict[str, dict] = {}
    for key, tensor in state.items():
        head, _, tail = key.partition(".")
        mods.setdefault(head, {})[tail] = tensor

    out: dict[str, tuple] = {}
    for name, parts in mods.items():
        m = _RE.match(name)
        if not m:
            raise ValueError(f"unexpected LoRA module name: {name}")
        block, kind = int(m.group(1)), m.group(2)
        down, up = parts["lora_down.weight"], parts["lora_up.weight"]
        alpha = parts.get("alpha")
        rank = down.shape[0]
        s = scale * (float(alpha) / rank if alpha is not None else 1.0)
        base = f"transformer_blocks.{block}"
        if kind == "attn_qkv_proj":
            for i, nm in enumerate(("to_q", "to_k", "to_v")):
                out[f"{base}.attn.{nm}"] = (down, _split_qkv_up(up, i), s)
        elif kind == "attn_out_proj":
            out[f"{base}.attn.to_out.0"] = (down, up, s)
        elif kind == "mlp_fc1":
            out[f"{base}.ff.net.0.proj"] = (down, _swap_ff_halves(up), s)
        else:
            out[f"{base}.ff.net.2"] = (down, up, s)
    return out


def merge_into(transformer, lora_path: str | Path, *, scale: float = 1.0) -> dict:
    """Fold the LoRA into the transformer's weights in place. Returns a report."""
    deltas = build_deltas(lora_path, scale=scale)
    by_name = dict(transformer.named_modules())
    applied, missing, shape_bad = 0, [], []
    for path, (down, up, s) in deltas.items():
        mod = by_name.get(path)
        if mod is None or not hasattr(mod, "weight"):
            missing.append(path)
            continue
        w = mod.weight
        delta = (up.to(torch.float32) @ down.to(torch.float32)) * s
        if tuple(delta.shape) != tuple(w.shape):
            shape_bad.append((path, tuple(w.shape), tuple(delta.shape)))
            continue
        with torch.no_grad():
            w.add_(delta.to(w.dtype).to(w.device))
        applied += 1
    return {"targets": len(deltas), "applied": applied,
            "missing": missing[:8], "shape_mismatch": shape_bad[:8]}


def dry_run(lora_path: str | Path, index_json: str | Path) -> dict:
    """Check the mapping against a weight index without loading the model."""
    import json

    wm = json.load(open(index_json))["weight_map"]
    shapes = {}
    import struct

    root = Path(index_json).parent
    for shard in sorted(set(wm.values())):
        with open(root / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                shapes[k] = v["shape"]

    deltas = build_deltas(lora_path)
    ok, bad = 0, []
    for path, (down, up, _) in deltas.items():
        want = shapes.get(path + ".weight")
        got = [up.shape[0], down.shape[1]]
        if want == got:
            ok += 1
        else:
            bad.append((path, want, got))
    return {"targets": len(deltas), "shape_ok": ok, "bad": bad[:8]}
