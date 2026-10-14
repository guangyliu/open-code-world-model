"""Proxy-driven chunked generation with LynnReal-Omni: our proxy as the frame-aligned control,
22 frames per chunk, frame 0 of each chunk pinned to the last delivered RGB frame.

Nothing in the release exposes this combination: the streaming entries take no control, and
the fixed-head reference encoder accepts one appearance image only. This borrows the
recipe of script/anchored_structure_chunk_editor.py -- a reference encoder that encodes
[appearance image, aligned control video] normally and pins a separately supplied boundary
image into the target's first latent slot -- and drives it from one resident process, so the
per-chunk cost is measured without model reloads.

    python chunk_drive.py --weights weight/standard --chunks chunks/boulevard --out out_chunks/boulevard \
        --width 960 --height 544 [--light-vae weight/light-vae] [--attention _flash_3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "LynnReal-Omni"
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weight/standard")
    ap.add_argument("--chunks", type=Path, required=True, help="dir with proxy_chunkNN.mp4, anchor_540.png, prompt_edit.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attention", default="auto")
    ap.add_argument("--light-vae", type=Path, default=None)
    ap.add_argument("--max-chunks", type=int, default=99)
    ap.add_argument("--resident", default="cpu", choices=("cpu", "cuda"),
                    help="where the text encoder waits between chunks; cpu restages it every chunk (~40 s)")
    ap.add_argument("--no-stage", action="store_true",
                    help="keep every module on the GPU (H200: peak was 78 GB with staging); skips the CPU<->GPU "
                         "shuffles of 8 DiT blocks around each VAE phase and of the VAE decoder around the denoise")
    ap.add_argument("--text-only", action="store_true",
                    help="drop the Qwen vision blocks (anchor + proxy video, ~7.7k rows) from the text context; "
                         "the aligned latent reference still carries the geometry")
    ap.add_argument("--history", action="store_true",
                    help="carry the release's bounded FramePack latent history (model/hybrid_history.py) from chunk 1 on, "
                         "beside the pinned boundary and the aligned control")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
    from diffusers.utils.export_utils import encode_video
    from model import offload
    from model.native_head import InitialReferenceEncoder, FixedReferenceHeadDenoise
    from model.offload import StagedReferenceEncoder
    from model.pipeline import Pipeline, encode_conditioning

    class BoundaryEncoder(InitialReferenceEncoder):
        """All references (appearance image + aligned control video) are encoded as usual; the
        boundary image, which is NOT among the references, is encoded into the head rows."""
        __call__ = StagedReferenceEncoder.__call__

        def encode_references(self, components, references, device=None):
            video, audio = StagedReferenceEncoder.encode_references(components, references, device)
            self.current_rows = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(
                components, [self.boundary_image], device)
            self.initial_encodes += 1
            return video, audio

    configure = offload.configure_vae_offload

    def configure_boundary(blocks, *args, **kw):
        configure(blocks, *args, **kw)
        encoder = BoundaryEncoder()
        blocks.sub_blocks["reference_encoder"] = encoder
        blocks.sub_blocks["denoise"] = FixedReferenceHeadDenoise(encoder, blocks.sub_blocks["denoise"].offload_blocks)

    offload.configure_vae_offload = configure_boundary
    if a.no_stage:
        from contextlib import nullcontext
        offload.temporarily_on_cpu = lambda module: nullcontext()   # module-global lookup at call time
    if a.resident == "cuda":
        # conditioning_device() budgets the VLM against *free* memory, which no longer counts a VLM that
        # already sits on the GPU, so it staged every text layer to CPU and back each chunk (66 s). Pin it.
        from model import pipeline as _pl
        _cd = _pl.conditioning_device
        _pl.conditioning_device = lambda pipe, offload_blocks=None: _cd(pipe, 0)

    chunks = sorted(a.chunks.glob("proxy_chunk*.mp4"))[: a.max_chunks]
    anchor = a.chunks / "anchor_540.png"
    prompt = (a.chunks / "prompt_edit.txt").read_text().strip()
    height, width = ((a.height + 31) // 32) * 32, ((a.width + 31) // 32) * 32

    t0 = time.perf_counter()
    pipe = Pipeline(a.weights, reference=True, aligned_indices=[1], vae_offload=True,
                    attention_backend=a.attention, light_vae=a.light_vae)
    encoder = pipe.pipe._blocks.sub_blocks["reference_encoder"]
    assert isinstance(encoder, BoundaryEncoder), type(encoder)
    load_s = time.perf_counter() - t0
    print(f"pipeline loaded in {load_s:.0f}s, attention={pipe.attention_backend}", flush=True)

    history = None
    if a.history:
        import numpy as np
        from diffusers.modular_pipelines.minimax_h3.packing import _ROPE_FRAME_RESCALE
        from model.hybrid_history import HybridHistory

        class ControlHistory(HybridHistory):
            """HybridHistory moves every target row four latent frames later to make room for the history
            clock. The frame-aligned control rows must move with the target, or the trained constant
            target-minus-control time offset (AlignedReferenceLayout) breaks."""

            def inject(self, state):
                condition = int(state.get("num_condition_video_rows"))
                super().inject(state)
                vi, pos = state.get("video_indices"), state.get("position_ids")
                image_rows = state.get("latent_height") * state.get("latent_width") // 4
                pos[vi[image_rows:condition], 0] += 4 * _ROPE_FRAME_RESCALE
                state.set("position_ids", pos)
                self.audit["control_rows_shifted"] = int(condition - image_rows)

        history = ControlHistory()
    denoise = pipe.pipe._blocks.sub_blocks["denoise"]

    resident = {}
    boundary = Image.open(anchor).convert("RGB").resize((width, height))
    delivered = [boundary]                      # frame 0 of the take is the anchor itself
    log = []
    for k, ctl in enumerate(chunks):
        paths = [anchor, ctl]
        tc = time.perf_counter()
        cond, enc = encode_conditioning(a.weights, prompt, paths, height, width, a.frames, a.out / "cache",
                                        aligned_indices=[1], resident=resident, resident_device=a.resident,
                                        text_only=a.text_only)
        cond_s = time.perf_counter() - tc
        encoder.boundary_image = boundary
        tg = time.perf_counter()
        state, timing = pipe.generate(cond, paths, height, width, a.frames, a.seed + k, a.steps)
        gen_s = time.perf_counter() - tg
        frames = state["videos"][0][: a.frames]
        audit = pipe.pipe._blocks.sub_blocks["denoise"].audit
        new = frames[1:]                        # frame 0 is the pinned boundary
        delivered += new
        boundary = new[-1]
        if history is not None and k == 0:
            # Seed the latent history with the first chunk's own output (22 = 17 + 5 frames, the
            # native window); from chunk 1 on the fixed head injects it and advances it.
            pixels = torch.from_numpy(np.stack([np.asarray(f) for f in frames])).permute(3, 0, 1, 2)[None].float() / 255
            history.initialize(pipe.pipe, pixels)
            denoise.history = history
        if history is not None:
            rec_hist = {k2: v for k2, v in denoise.history.audit.items() if k2 in ("history_rows", "stored_history_latents", "control_rows_shifted")} if k else {}
        rec = dict(chunk=k, control=ctl.name, conditioning_s=round(cond_s, 2), cache_hit=enc.get("cache_hit"),
                   text_rows=int(cond["prompt_embeds"].shape[1]), text_only=a.text_only,
                   vlm_forward_s=round(enc.get("seconds", 0), 2), staged_text_layers=enc.get("staged_text_layers"),
                   generate_s=round(gen_s, 2), dit_s=round(timing["dit_ms"] / 1000, 2),
                   dit_step_s=[round(x / 1000, 2) for x in timing["dit_forward_ms"]],
                   decode_s=round(timing["video_decoder_ms"] / 1000, 2), peak_gb=round(timing["peak_allocated_bytes"] / 1e9, 1),
                   fixed_head_forwards=audit.get("fixed_head_forwards"), new_frames=len(new),
                   resident=a.resident, staged=not a.no_stage, **(rec_hist if history is not None else {}))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        new[-1].save(a.out / f"boundary_{k:02d}.png")
    encode_video(delivered, fps=24, output_path=str(a.out / "take.mp4"))
    (a.out / "timing.json").write_text(json.dumps(dict(load_s=round(load_s, 1), width=width, height=height,
                                                      frames_per_chunk=a.frames, steps=a.steps, chunks=log), indent=1))
    per = [r["generate_s"] for r in log[1:]] or [log[0]["generate_s"]]
    print(f"{len(delivered)} frames delivered; per chunk (after the first): median generate {sorted(per)[len(per)//2]:.1f}s "
          f"for {a.frames - 1} new frames = {(a.frames - 1) / 24:.2f}s of video", flush=True)


if __name__ == "__main__":
    main()
