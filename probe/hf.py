"""Download helpers for the 40 released CWM inference examples."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

REPO = "NTU-yiwen/code-world-model-inference-examples-40"
REV = "f8307bf50cb19bed191682734fdf444738160f91"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}"
DATA = Path(os.environ.get("CWM_DATA", Path.home() / "cwm-data")) / "examples"
EXAMPLES = range(1, 41)


def fetch(rel: str, *, force: bool = False) -> Path:
    """Download `rel` (path inside the dataset repo) into DATA, return local path."""
    dest = DATA / rel
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def metadata(example: int) -> dict:
    return json.loads(fetch(f"{example}/metadata.json").read_text(encoding="utf-8"))


def prompt(example: int, index: int) -> str:
    return fetch(f"{example}/prompts/{index:03d}.txt").read_text(encoding="utf-8")


def chunk_count(meta: dict) -> int:
    """Condition NPZ chunks are non-overlapping 124-frame blocks over the whole take.

    They are NOT the inference windows, which advance by 90 frames and therefore
    straddle chunk boundaries. The final chunk is short when frames % 124 != 0.
    """
    return -(-int(meta["frames"]) // 124)


def fetch_chunk(example: int, index: int) -> Path:
    return fetch(f"{example}/condition/{index:03d}.npz")
