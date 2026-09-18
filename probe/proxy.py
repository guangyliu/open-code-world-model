"""Decode released CWM proxy conditions and re-encode them the way inference does."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Frozen by the released inference path (cwm_h3_inference/constants.py).
W, H = 336, 192
SRC_W, SRC_H = 672, 384
WINDOW_FRAMES = 124
STRIDE_FRAMES = 90
FPS = 24
NEAR_M, FAR_M = 0.3, 256.0
VALID_EPS_M = 1.0e-3
SEMANTIC_U = (32, 96, 160, 224)
SEMANTIC_V = (43, 128, 213)

CLASSES = (
    "void_unknown", "sky", "water", "terrain",
    "road_paved", "vegetation", "building_structure", "infrastructure",
    "human", "animal", "vehicle", "prop",
)
VOID, SKY, WATER, TERRAIN, ROAD, VEGETATION, BUILDING, INFRA, HUMAN, ANIMAL, VEHICLE, PROP = range(12)

# Classes the paper describes as discrete moving primitives rather than scene geometry.
ACTOR_CLASSES = (HUMAN, ANIMAL, VEHICLE, PROP)


@dataclass
class Window:
    """One 124-frame proxy window: metric depth plus categorical semantic IDs."""

    example: int
    index: int
    depth: np.ndarray      # (124, 192, 336) float32, metres; <= VALID_EPS_M means invalid
    semantic: np.ndarray   # (124, 192, 336) uint8 in [0, 11]

    @property
    def start_frame(self) -> int:
        return self.index * STRIDE_FRAMES

    def valid(self) -> np.ndarray:
        return self.depth > np.float32(VALID_EPS_M)


def log_codes_to_depth(codes: np.ndarray) -> np.ndarray:
    """uint16 fixed-log codes -> metric depth. Inverse of depth_to_log_codes.

    Code 0 is ambiguous: it means both "invalid" and "at or beyond the 256 m FAR
    plane". We resolve it to 0 (invalid) so that the sky rule of section 3 holds,
    which loses genuinely distant geometry -- up to 6.4% of pixels in wide scenes.
    Use the float32 examples for anything metric.
    """
    scale = math.log(FAR_M) - math.log(NEAR_M)
    out = np.exp(math.log(FAR_M) - codes.astype(np.float64) / 65535.0 * scale)
    return np.where(codes == 0, 0.0, out).astype(np.float32)


def load_window(example: int, index: int) -> Window:
    from . import hf

    with np.load(hf.fetch_chunk(example, index)) as z:
        raw = z["depth"]
        # The two published depth formats differ by dtype: float32 holds metres,
        # uint16 holds fixed-log codes. Decoding the latter as metres is silent nonsense.
        depth = log_codes_to_depth(raw) if raw.dtype == np.uint16 else np.asarray(raw, dtype=np.float32)
        semantic = np.asarray(z["semantic_id"], dtype=np.uint8)
    # Chunks are 124-frame blocks; the final chunk of a take is short.
    if depth.shape[1:] != (H, W) or semantic.shape != depth.shape or not 0 < depth.shape[0] <= WINDOW_FRAMES:
        raise ValueError(f"unexpected chunk shape {depth.shape} for example {example}/{index}")
    if semantic.max() > 11:
        raise ValueError(f"semantic ID out of range in example {example}/{index}")
    return Window(example=example, index=index, depth=depth, semantic=semantic)


def depth_to_log_codes(metric: np.ndarray) -> np.ndarray:
    """Metric depth -> uint16 fixed-log codes, matching duv.load_duv_frame exactly."""
    valid = metric > np.float32(VALID_EPS_M)
    codes = np.zeros(metric.shape, dtype=np.uint16)
    if valid.any():
        clipped = np.clip(metric[valid].astype(np.float64), NEAR_M, FAR_M)
        normalized = (math.log(FAR_M) - np.log(clipped)) / (math.log(FAR_M) - math.log(NEAR_M))
        codes[valid] = np.floor(np.clip(normalized, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    return codes


def to_duv(metric: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Metric depth + semantic ID -> the 3-channel uint8 DUV image the model sees."""
    codes = depth_to_log_codes(metric)
    d8 = np.floor(codes.astype(np.float64) / 257.0 + 0.5).astype(np.uint8)
    u = np.asarray(SEMANTIC_U, dtype=np.uint8)[semantic % 4]
    v = np.asarray(SEMANTIC_V, dtype=np.uint8)[semantic // 4]
    return np.stack((d8, u, v), axis=-1)
