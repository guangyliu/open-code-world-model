"""CWM LoRA on MiniMax-H3 Ref2VA, 4-way context parallel, using the shared services.

    torchrun --nproc_per_node=4 run_cp.py --example 1 --window 0

Division of labour:

  DiT + CWM LoRA        local   (65 GB)  unavoidable, and the LoRA lives here
  conditioner           local   (48 GB)  see below
  VAE encode_references remote           shared service, op="encode_references"
  VAE decode            remote           shared service, no `op` field

The conditioner stays local even though the cond service covers ref2va. The CWM LoRA was
trained with a fixed system prompt wrapped around the whole presentation
(`<|im_start|>system ... <|im_start|>user [labels + vision blocks + caption] <|im_end|>`),
and the service calls `Ref2VATextEncoderStep.encode_prompt` directly, which applies no chat
template. The `prompt` argument lands *after* the vision blocks, so the template cannot be
smuggled in from outside -- it has to wrap them. Dropping it would condition the LoRA
differently from how it was trained, which is exactly the kind of silent mismatch this port
is trying to avoid. 65 + 48 = 113 GB fits one H200.

Service etiquette, per the operator: no `S.sweep()` (it would delete other callers' pending
files), no `S.beat()` under a name the live stack uses, absolute paths on weka for out_path,
and ref2va requests go as `.pt` because `prepared_references` carries PIL images and arrays.

Multi-GPU structure follows h3_ref_worker / h3_pdd_worker, including three bugs their
comments record: `prepared_references` must be broadcast explicitly (the reference encoder
fills latent geometry into those objects in place, so a state diff never sees it); each
broadcast key keeps its original device (`text_token_tags` must stay on CPU); and the
geometry returned by the service must be written back onto the local objects, because the
service mutated its own unpickled copy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The MiniMax-H3 runtime, its weights, and the CWM inference repo all live outside this
# repository; set these to wherever you cloned them. The defaults assume they sit beside
# the checkout, which is the layout the quick start describes.
H3_RUNTIME = Path(os.environ.get("H3_RUNTIME", HERE.parent.parent / "minimax-h3" / "runtime"))
sys.path.insert(0, str(H3_RUNTIME))
sys.path.insert(0, str(H3_RUNTIME / "h3_kit" / "code"))

MODEL = os.environ.get("H3_MODEL", str(HERE.parent.parent / "minimax-h3"))
CWM_PROMPTS = Path(os.environ.get(
    "CWM_PROMPTS_DIR",
    HERE.parent.parent / "code-world-model" / "src" / "cwm_h3_inference" / "prompts"))
NEAR_M, FAR_M = 0.3, 256.0
SEMANTIC_U = (32, 96, 160, 224)
SEMANTIC_V = (43, 128, 213)


def duv_frames(depth_m: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Metric depth + semantic IDs -> the 3-channel uint8 the LoRA was trained on.

    U and V are exact palette values, so these frames travel as arrays; a lossy container
    would quantise them into neighbouring classes.
    """
    valid = depth_m > np.float32(1.0e-3)
    codes = np.zeros(depth_m.shape, dtype=np.uint16)
    if valid.any():
        clipped = np.clip(depth_m[valid].astype(np.float64), NEAR_M, FAR_M)
        norm = (math.log(FAR_M) - np.log(clipped)) / (math.log(FAR_M) - math.log(NEAR_M))
        codes[valid] = np.floor(np.clip(norm, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    d8 = np.floor(codes.astype(np.float64) / 257.0 + 0.5).astype(np.uint8)
    u = np.asarray(SEMANTIC_U, dtype=np.uint8)[semantic % 4]
    v = np.asarray(SEMANTIC_V, dtype=np.uint8)[semantic // 4]
    return np.stack((d8, u, v), axis=-1)


def window_from_example(root: Path, example: int, start: int, n: int) -> np.ndarray:
    meta = json.loads((root / str(example) / "metadata.json").read_text())
    is_codes = meta["depth_format"].startswith("uint16")
    out = []
    for k in range(start, start + n):
        chunk, off = divmod(k, 124)
        with np.load(root / str(example) / "condition" / f"{chunk:03d}.npz") as z:
            raw, sem = z["depth"][off], z["semantic_id"][off]
        if is_codes:
            scale = math.log(FAR_M) - math.log(NEAR_M)
            d = np.exp(math.log(FAR_M) - raw.astype(np.float64) / 65535.0 * scale)
            d = np.where(raw == 0, 0.0, d).astype(np.float32)
        else:
            d = raw.astype(np.float32)
        out.append(duv_frames(d, sem))
    return np.stack(out)


def window_from_dir(cond: Path, start: int, n: int) -> np.ndarray:
    from PIL import Image
    out = []
    for k in range(start, start + n):
        d = np.fromfile(cond / f"{k:06d}.depth.f32", dtype="<f4").reshape(192, 336)
        s = np.asarray(Image.open(cond / f"{k:06d}.semantic_id.png"), dtype=np.uint8)
        out.append(duv_frames(d, s))
    return np.stack(out)


def chat_affixes(processor, system_prompt: str) -> tuple[str, str]:
    """Slice the processor's own chat template around a sentinel.

    Writing the `<|im_start|>` scaffolding by hand would be a guess about the tokenizer's
    template; asking the processor and slicing is exact.
    """
    sentinel = "CWM_BODY"
    formatted = processor.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=False)
    if formatted.count(sentinel) != 1:
        raise ValueError("sentinel did not survive the chat template")
    if formatted.count("<|im_start|>system\n") != 1 or formatted.count("<|im_start|>user\n") != 1:
        raise ValueError("chat template must contain exactly one system and one user block")
    if "<|im_start|>assistant" in formatted:
        raise ValueError("chat template must not add an assistant block")
    return tuple(formatted.split(sentinel))


def patch_presentation(holder: dict) -> None:
    from diffusers.modular_pipelines.minimax_h3 import encoders as E
    original = E.build_ref2va_presentation

    def wrapped(tokenizer, prompt, references, image_counts, video_counts):
        ids, tags = original(tokenizer, prompt, references, image_counts, video_counts)
        pre, suf = holder["affixes"]
        pre_ids = tokenizer(pre, add_special_tokens=False)["input_ids"]
        suf_ids = tokenizer(suf, add_special_tokens=False)["input_ids"]
        return (list(pre_ids) + list(ids) + list(suf_ids),
                [0] * len(pre_ids) + list(tags) + [0] * len(suf_ids))

    E.build_ref2va_presentation = wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=int, default=1)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=9942001)
    # Every window used to draw its noise from the same seed. With a proxy that barely
    # changes between windows and a caption that repeats, that made a 4-window take nearly
    # periodic at the 90-frame stride (frame k vs k-90 differed by 7.9 grey levels where
    # k vs k-45 differed by 18). Window w now seeds at seed + w * step; step 0 restores
    # the old behaviour.
    ap.add_argument("--window-seed-step", type=int, default=1)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--examples-root",
                    default=os.environ.get("CWM_EXAMPLES", "examples40"))
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt-dir", default=None,
                    help="per-window captions NNN.txt; required with --windows > 1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--out", default=os.environ.get("CWM_OUT", "out"))
    ap.add_argument("--save-latents", default=None,
                    help="write the raw video latents here, for the next window's Retake34 prefix")
    ap.add_argument("--serve", default=None,
                    help="stay resident and take whole takes from this queue directory")
    ap.add_argument("--worker", default="cwm", help="worker name, for logging")
    ap.add_argument("--poll", type=float, default=3.0,
                    help="seconds between queue polls in serve mode")
    ap.add_argument("--windows", type=int, default=1,
                    help="generate this many chained windows in one process")
    ap.add_argument("--pdd", default=None,
                    help="PAI PDD acceleration checkpoint; overrides --steps with nfe+1")
    ap.add_argument("--retake-prefix", default=None,
                    help=".pt from retake_prep.py: clean rows to pin at the head of the target video")
    a = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    is0 = rank == 0
    if world > 1:
        from datetime import timedelta
        # Explicit device_id: without it NCCL guesses by global rank, which wedges
        # ALLTOALL_BASE when several torchrun groups share a node.
        dist.init_process_group("nccl", device_id=torch.device(dev), timeout=timedelta(minutes=45))

    def log(*m):
        if is0:
            print(*m, flush=True)

    import h3_svc_common as S

    root = Path(a.examples_root)

    def window_inputs(w: int):
        """The DUV, caption and system prompt for window `w`. Everything else is shared."""
        if a.conditions:
            d = window_from_dir(Path(a.conditions), w * 90, a.frames)
            sr = str(a.conditions)
        else:
            d = window_from_example(root, a.example, w * 90, a.frames)
            sr = f"released example {a.example}"
        pdir = Path(a.prompt_dir) if a.prompt_dir else (root / str(a.example) / "prompts")
        cap = Path(a.prompt_file if (a.prompt_file and a.windows == 1)
                   else pdir / f"{w:03d}.txt").read_text(encoding="utf-8").strip()
        sysp = (CWM_PROMPTS / ("system_w0.txt" if w == 0 else "system_wn.txt")
                ).read_text(encoding="utf-8").strip()
        return d, sr, cap, sysp

    # Shared by every window: the anchor is the take's authoritative first frame, not a
    # per-window input. (It lived inside the block window_inputs replaced, and going missing
    # here is what broke two runs.)
    anchor = a.anchor or str(root / str(a.example) / "first_frame.png")
    duv, src, caption, system_prompt = window_inputs(a.window)
    log(f"DUV {duv.shape} from {src} | caption {len(caption)} | world={world} | svc={S.DIRS['vae_req']}")

    import h3_context_parallel as h3cp
    import h3_harness as H
    from diffusers import ComponentsManager
    from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VAReferenceEncoderStep, MiniMaxH3Ref2VATextEncoderStep)
    # Both live in packing, not before_denoise, in this build.
    from diffusers.modular_pipelines.minimax_h3.packing import (
        MINIMAX_H3_KEYFRAME_NOISE_AUG, keyframe_condition_noise)

    holder: dict = {}
    patch_presentation(holder)

    t0 = time.time()
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline(MODEL, components_manager=ComponentsManager())
    # Load everything except the two VAEs, whose work goes to the shared service. Naming
    # the wanted components explicitly is how `audio_scheduler` got left out and the run
    # died on `NoneType.set_timesteps`: the light schedulers and processors are needed too.
    heavy_remote = {"vae", "audio_vae"}
    if not is0:
        heavy_remote.add("text_encoder")      # only rank0 conditions
    names = [n for n in ("audio_scheduler", "processor", "scheduler", "text_encoder",
                         "tokenizer", "transformer_ref", "video_processor")
             if n not in heavy_remote]
    pipe.load_components(names=names, dtype=torch.bfloat16)
    dit = pipe.transformer_ref              # ref2va's DiT attribute is not `transformer`
    dit.to(dev)
    type(pipe)._execution_device = property(lambda self: torch.device(dev))
    log(f"[{time.time()-t0:.0f}s] components loaded")

    if is0:
        holder["affixes"] = chat_affixes(pipe.processor, system_prompt)
        # The last 13 decoder layers and lm_head never run (H3 reads hidden_states[50]) yet
        # still hold ~13.6 GB. `load_components` leaves everything on the CPU and
        # `dit.to(dev)` moves only the DiT -- a conditioner left behind does not fail, it
        # runs on the CPU slowly enough that the other ranks' NCCL watchdog fires.
        H._truncate_conditioner(pipe.text_encoder)
        pipe.text_encoder.to(dev)

    if not a.no_lora:
        import cwm_lora
        # The CWM LoRA is a separate download; CWM_LORA_DIR points at the directory
        # holding it.
        lora_dir = Path(os.environ.get("CWM_LORA_DIR", "lora"))
        try:
            lora = next(lora_dir.rglob("*.safetensors")).as_posix()
        except StopIteration:
            raise SystemExit(f"no .safetensors under {lora_dir}; set CWM_LORA_DIR")
        rep = cwm_lora.merge_into(dit, lora)
        if rep["applied"] != rep["targets"]:
            raise SystemExit(f"LoRA applied {rep['applied']}/{rep['targets']}; refusing to run")
        log(f"LoRA merged {rep['applied']}/{rep['targets']}")

    if a.pdd:
        # minimax_h3_pdd ships beside the acceleration checkpoint itself.
        sys.path.insert(0, str(Path(a.pdd).resolve().parent))
        from minimax_h3_pdd import apply_pdd_lora
        nfe = apply_pdd_lora(dit, a.pdd, pipe.scheduler.shift, pipe.audio_scheduler.shift)
        # The H3 scheduler counts the terminal sigma, so the pipeline takes nfe + 1.
        a.steps = nfe + 1
        log(f"PDD acceleration: nfe={nfe} -> num_inference_steps={a.steps}")

    if world > 1:
        # The online PDD workers run the default `native` (SDPA) backend; this path had
        # `_native_flash` hardcoded. CWM_BACKEND makes it comparable.
        h3cp.enable(dit, ulysses_degree=world,
                    backend=os.environ.get("CWM_BACKEND", "native"))
    else:
        h3cp.patch_model_class()

    # ---------------- text encoder: local on rank0, then broadcast ----------------
    _orig_txt = MiniMaxH3Ref2VATextEncoderStep.__call__

    def _txt(self, components, state):
        payload = [None]
        if is0:
            components, state = _orig_txt(self, components, state)
            payload[0] = (state.values["prompt_embeds"].cpu(),
                          state.values["text_token_tags"].cpu())
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        pe, tt = payload[0]
        state.set("prompt_embeds", pe.to(device=torch.device(dev), dtype=dit.dtype))
        state.set("text_token_tags", tt.cpu())     # upstream requires this on the CPU
        return components, state

    # ---------------- reference encoder: shared VAE service ----------------
    def _svc_ref(self, components, state):
        payload = [None]
        if is0:
            bs = self.get_block_state(state)
            rid = S.new_id()
            tmp = f"{S.DIRS['vae_req']}/{rid}.pt.tmp"
            torch.save(dict(op="encode_references", prepared_references=bs.prepared_references,
                            patch_size=tuple(dit.config.patch_size),
                            audio_latent_channels=getattr(components, "audio_latent_channels", 32)),
                       tmp)
            os.replace(tmp, f"{S.DIRS['vae_req']}/{rid}.pt")
            if not S.wait_for(f"{S.DIRS['vae_out']}/{rid}.json", timeout=900):
                raise RuntimeError("VAE service (reference encode) timed out")
            r = S.read_json(f"{S.DIRS['vae_out']}/{rid}.json") or {}
            os.remove(f"{S.DIRS['vae_out']}/{rid}.json")
            if not r.get("ok"):
                raise RuntimeError(f"reference encode failed: {r.get('error')}")
            blob = torch.load(f"{S.DIRS['vae_out']}/{rid}.pt", map_location="cpu")
            os.remove(f"{S.DIRS['vae_out']}/{rid}.pt")
            payload[0] = (blob["condition_latents"], blob["audio_condition_latents"],
                          blob["geometry"], bs.prepared_references)
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        cl, al, geo, prepared = payload[0]
        # The service mutated its own unpickled copy, so write the latent geometry back;
        # the packing layer reads it off these objects.
        state.set("prepared_references", prepared)
        refs = state.values.get("prepared_references") or []
        for obj, g in zip(refs, geo):
            for k, v in g.items():
                setattr(obj, k, v)
        d = torch.device(dev)
        if cl is not None:
            bs = self.get_block_state(state)
            # Drawn on every rank, after the broadcast, so the generator advances in step.
            noise = keyframe_condition_noise(
                tuple((o.num_latent_frames, o.latent_height, o.latent_width)
                      for o in refs if o.kind != "audio"),
                tuple(dit.config.patch_size),
                getattr(components, "vae_latent_channels", 24),
                generator=bs.generator, device=d)
            cl = components.scheduler.scale_noise(cl.to(d), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
        state.set("condition_latents", cl)
        state.set("audio_condition_latents", None if al is None else al.to(d))
        return components, state

    # ---------------- decode steps: stand in for the absent VAEs ----------------
    # The decode steps read `components.vae.config` before the `output_type == "latent"`
    # branch, so with the VAEs offloaded to the service every rank hits
    # `'NoneType' object has no attribute 'config'`. h3_ref_worker hits the same thing on
    # its non-rank0 workers and answers it the same way: a config-only stub.
    from types import SimpleNamespace

    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep)

    def _cfg_stub(sub):
        with open(f"{MODEL}/{sub}/config.json") as f:
            return SimpleNamespace(config=SimpleNamespace(**json.load(f)))

    _VAE_STUB, _AVAE_STUB = _cfg_stub("vae"), _cfg_stub("audio_vae")

    class _CompProxy:
        """Pass everything through, but hand out a config-only stub for a missing VAE."""

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, k):
            v = getattr(object.__getattribute__(self, "_real"), k)
            if v is None and k == "vae":
                return _VAE_STUB
            if v is None and k == "audio_vae":
                return _AVAE_STUB
            return v

    for _cls in (MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep):
        def _wrap(orig):
            def _c(self, components, state):
                return orig(self, _CompProxy(components), state)
            return _c
        _cls.__call__ = _wrap(_cls.__call__)

    MiniMaxH3Ref2VATextEncoderStep.__call__ = _txt
    MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = _svc_ref

    # ---- keep the DUV reference at its native resolution ----------------------------
    # `prepare_reference_frames` puts every video reference on the full canvas: a 336x192 DUV
    # resolves to 1344x768 and contributes 37 x 1008 = 37,296 condition rows -- as many as the
    # entire target video, and 15x what CWM itself uses. Their constants are explicit:
    #   REFERENCE_LATENT_SHAPE  = (24, 37, 12, 21)    <- native 336x192 / 16
    #   REFERENCE_RUNTIME_SHAPE = (1, 24, 37, 12, 22) <- width padded to an even latent
    # So the released LoRA was trained on a 12x21 DUV latent and we have been feeding it 48x84:
    # out of distribution, and 4.5x the attention for it.
    #
    # 336 gives an odd latent width, which cannot be patched, so the frames are padded (not
    # resized -- DUV U/V are exact palette values and interpolation would move them into
    # neighbouring classes) to 352 = 22 * 16.
    # Default off. Keeping the DUV native is 3x faster (46k rows against 86k) and matches
    # CWM's own REFERENCE_LATENT_SHAPE, but it loses proxy adherence here: 6x11 = 66
    # tokens per frame cannot resolve a distant vehicle that occupies 18x6 DUV pixels,
    # and diffusers gives a reference its own rotary time rather than aligning it frame
    # for frame with the target the way CWM's packing does. Verified not to be the pad:
    # edge and zero padding fail identically.
    if os.environ.get("CWM_NATIVE_DUV", "0") == "1":
        import diffusers.modular_pipelines.minimax_h3.packing_ref2va as _pr

        _orig_prep = _pr.prepare_reference_frames

        def _prep_native(frames, num_frames):
            f = frames[:num_frames]
            if f.shape[1:3] == (192, 336):
                # Black is not a legal DUV value: U must be one of (32,96,160,224) and V one
                # of (43,128,213), so a zero strip is a semantic class the model has never
                # seen. CWM pads the *latent* with zeros, which is that space's mean; the
                # nearest pixel-space equivalent that stays in-palette is to repeat the edge.
                mode = os.environ.get("CWM_DUV_PAD", "edge")
                w = 352 - 336
                if mode == "edge":
                    pad = np.repeat(f[:, :, -1:, :], w, axis=2)
                else:
                    pad = np.zeros((f.shape[0], 192, w, 3), dtype=f.dtype)
                return np.concatenate((f, pad), axis=2)
            return _orig_prep(frames, num_frames)

        _pr.prepare_reference_frames = _prep_native
        import diffusers.modular_pipelines.minimax_h3.before_encoder as _be
        _be.prepare_reference_frames = _prep_native
        log("DUV reference kept native (192x336 -> padded 192x352)")

    ref_edge = int(os.environ.get("CWM_REF_EDGE", "1024"))
    import diffusers.modular_pipelines.minimax_h3.packing_ref2va as _pk
    _pk.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = ref_edge
    log(f"reference image short edge: {ref_edge}")

    # A resident VAE on rank 0 turns each window's latents into the next window's pinned
    # prefix without leaving the process. The standalone retake_prep.py costs ~82 s a window,
    # almost all of it Python start-up, CUDA init and reloading this same VAE.
    # Built on first use, NOT from a.windows here: this line runs during setup, while the
    # serve loop rewrites a.windows per job much later, so a worker started with the
    # default --windows 1 used to have no VAE and every multi-window job it claimed died
    # on the first retake prefix.
    _prep_vae = {}

    def get_prep_vae():
        if "v" not in _prep_vae:
            from diffusers import AutoencoderKLMiniMaxH3
            tv = time.time()
            _prep_vae["v"] = AutoencoderKLMiniMaxH3.from_pretrained(
                MODEL, subfolder="vae", torch_dtype=torch.bfloat16).to(dev).eval()
            log(f"[{time.time()-tv:.0f}s] resident VAE loaded for retake prep")
        return _prep_vae["v"]

    def make_prefix(lat):
        """Previous window's latents -> the rows the next window pins. Rank 0 only."""
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.modular_pipelines.minimax_h3.packing import (
            MINIMAX_H3_KEYFRAME_ENCODE_SEED, MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
            patchify_video_latents)
        prep_vae = get_prep_vae()
        pm = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=dev).view(1, -1, 1, 1, 1)
        ps = torch.tensor(MINIMAX_H3_PIXEL_STD, device=dev).view(1, -1, 1, 1, 1)
        lm = torch.tensor(prep_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        ls = torch.tensor(prep_vae.config.latents_std).view(1, -1, 1, 1, 1)
        with torch.no_grad():
            # Only the last 34 pixel frames are needed, so only the latents that produce them
            # are decoded. A valid latent sequence has 5n+2 frames -- feeding 10 or 14 gives a
            # malformed chunk structure and a *different* reconstruction -- but the last
            # 5*2+2 = 12 latents decode to 39 frames whose final 34 are bit-identical to the
            # full 37-latent decode (measured: mean and max |diff| both exactly 0). 13.7 s
            # against 48.5 s.
            keep = min(12, lat.shape[2])
            v = prep_vae.decode(lat[:, :, -keep:].to(dev, prep_vae.dtype), return_dict=False)[0]
            v = (v.float() * ps + pm).clamp(0, 1)
            tail = v[:, :, -34:]
            # Pad the 34 real frames back to a full window with the last frame: the VAE's
            # temporal receptive field has to be filled or the leading latents differ from
            # those of a genuine 124-frame clip.
            padded = torch.cat((tail, tail[:, :, -1:].repeat(1, 1, a.frames - 34, 1, 1)), dim=2)
            moments = prep_vae._encode(((padded - pm) / ps).to(prep_vae.dtype))
            clean = DiagonalGaussianDistribution(moments).sample(
                generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            clean = clean.to(torch.float16).float().cpu()
        rows = patchify_video_latents((clean - lm) / ls, (1, 2, 2))
        per = rows.shape[0] // clean.shape[2]
        return rows[: per * 10].contiguous()

    import cwm_retake
    if a.retake_prefix:
        blob = torch.load(a.retake_prefix, map_location="cpu")
        pinned = cwm_retake.install(blob["rows"])
        log(f"Retake34: pinned {pinned} rows "
            f"({blob['prefix_latents']} latent frames x {blob['per_latent_frame']})")

    from diffusers.modular_pipelines.minimax_h3 import before_denoise as _bd
    _orig_plan = _bd.MiniMaxH3SetTimestepsStep.__call__

    def _plan_logged(self, components, state):
        out = _orig_plan(self, components, state)
        lay = state.get("layout")
        if lay is not None and is0:
            ncv = int(lay.num_condition_video_rows)
            nv = int(lay.video_indices.numel())
            log(f"sequence: total={int(lay.sequence_length)} rows | video={nv} "
                f"(condition {ncv} + target {nv - ncv}) | audio={int(lay.audio_indices.numel())} "
                f"| text={int(lay.text_indices.numel())}")
        return out

    _bd.MiniMaxH3SetTimestepsStep.__call__ = _plan_logged

    def window_tag(w: int) -> str:
        """Output name for window `w`.

        With `--windows 1` an explicit `--tag` is used verbatim, because the single-window
        callers (the benchmarks, the A/B scripts) name their own outputs and appending `_w0`
        would rename every one of them.
        """
        if a.tag:
            t = a.tag if a.windows == 1 else f"{a.tag}_w{w}"
        else:
            stem = (f"ex{a.example:02d}" if not a.conditions
                    else Path(a.conditions).parent.name)
            t = f"{stem}_w{w}"
        return t + "_nolora" if a.no_lora else t

    prev_lat = None

    def run_take():
        """Generate one take with the model already loaded.

        Reads the per-take fields off `a`, which the serve loop rewrites per job, so the
        DiT, the LoRA, the PDD heads and the resident VAE survive from one take to the
        next. Everything above this point is setup and runs once.
        """
        nonlocal prev_lat
        duv, src, caption, system_prompt = window_inputs(a.window)
        anchor = a.anchor or str(root / str(a.example) / "first_frame.png")
        holder["affixes"] = chat_affixes(pipe.processor, system_prompt)
        prev_lat = None
        for wi in range(a.windows):
            w = a.window + wi
            if wi:
                duv, src, caption, system_prompt = window_inputs(w)
                holder["affixes"] = chat_affixes(pipe.processor, system_prompt)
                # Rank 0 builds the prefix and hands it to the others: every rank must pin the
                # same rows or the collectives diverge.
                payload = [None]
                if is0:
                    tp = time.time()
                    payload[0] = make_prefix(prev_lat)
                    log(f"[{time.time()-tp:.0f}s] retake prefix for window {w}: "
                        f"{payload[0].shape[0]} rows")
                if world > 1:
                    dist.broadcast_object_list(payload, src=0)
                cwm_retake.install(payload[0])

            # Built every pass, not only for continuation windows: the pre-loop assignment used
            # to cover window 0 and removing it left `refs` undefined on the first iteration.
            refs = [MiniMaxH3Reference(image=anchor), MiniMaxH3Reference(video=duv, fps=24.0)]

            if world > 1:
                torch.cuda.synchronize()
                dist.barrier()
            t0 = time.time()
            state = pipe(prompt=caption, references=refs, num_frames=a.frames,
                         height=a.height, width=a.width, num_inference_steps=a.steps,
                         output_type="latent",
                         generator=torch.Generator("cpu").manual_seed(a.seed + w * a.window_seed_step))
            torch.cuda.synchronize()
            dt = time.time() - t0
            log(f"[{dt:.0f}s] window {w} denoise done ({dt/max(a.steps,1):.1f}s/step)")

            lat = state.get("videos")
            prev_lat = lat.cpu() if is0 else None
            if not is0:
                continue

            out_dir = Path(a.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = window_tag(w)
            out_path = str((out_dir / f"{tag}.mp4").resolve())
            alat = state.get("audio")
            if a.save_latents and a.windows == 1:
                Path(a.save_latents).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"video_latents": lat.cpu()}, a.save_latents)
                log(f"saved latents {tuple(lat.shape)} -> {a.save_latents}")
            rid = S.new_id()
            tmp = f"{S.DIRS['vae_req']}/{rid}.pt.tmp"
            torch.save(dict(video_latents=lat.cpu(),
                            audio_latents=alat.cpu() if torch.is_tensor(alat) else None,
                            sampling_rate=state.get("sampling_rate"), out_path=out_path), tmp)
            os.replace(tmp, f"{S.DIRS['vae_req']}/{rid}.pt")
            if not S.wait_for(f"{S.DIRS['vae_out']}/{rid}.json", timeout=1200):
                raise RuntimeError("VAE service (decode) timed out")
            r = S.read_json(f"{S.DIRS['vae_out']}/{rid}.json") or {}
            os.remove(f"{S.DIRS['vae_out']}/{rid}.json")
            if not r.get("ok"):
                raise RuntimeError(f"decode failed: {r.get('error')}")
            (out_dir / f"{tag}.json").write_text(json.dumps(
                {"source": src, "example": a.example, "window": w, "steps": a.steps,
                 "seed": a.seed + w * a.window_seed_step, "frames": a.frames, "size": [a.width, a.height],
                 "lora": not a.no_lora, "world": world, "denoise_s": round(dt, 1)}, indent=1))
            print(f"WROTE {out_path}", flush=True)


    if not a.serve:
        run_take()
    else:
        import glob
        qdir = Path(a.serve)
        for sub in ("todo", "work", "done", "fail"):
            (qdir / sub).mkdir(parents=True, exist_ok=True)
        log(f"worker {a.worker} ready on {qdir}, world={world}")
        n_done = 0
        while True:
            spec = [None]
            if is0:
                # Claiming is an atomic rename, the same trick the h3 services use: the
                # worker that wins the rename owns the job and no lock daemon is needed.
                for f in sorted(glob.glob(str(qdir / "todo" / "*.json"))):
                    dst = qdir / "work" / Path(f).name
                    try:
                        os.rename(f, dst)
                    except OSError:
                        continue
                    spec[0] = dict(json.loads(dst.read_text()), _path=str(dst))
                    break
            if world > 1:
                # Every rank has to agree on whether there is work: a rank that skips the
                # take while the others run it deadlocks the next collective.
                dist.broadcast_object_list(spec, src=0)
            if spec[0] is None:
                time.sleep(a.poll)
                continue
            job = spec[0]
            # Consumed once during setup, above the loop: --pdd merges the acceleration
            # LoRA and rewrites a.steps, --no-lora decides whether the LoRA is merged,
            # --examples-root is already baked into `root`. Assigning them per job changes
            # the attribute and nothing else, so a mismatch is refused rather than run as
            # a take the job did not ask for.
            SETUP_ONLY = ("pdd", "no_lora", "examples_root")
            bad = [k for k in SETUP_ONLY
                   if k in job and job[k] != getattr(a, k)]
            if bad:
                detail = ", ".join(f"{k}={job[k]!r} but worker started with "
                                   f"{getattr(a, k)!r}" for k in bad)
                log(f"job {Path(job['_path']).stem}: REFUSED -- {detail}. These take "
                    f"effect only at worker startup; relaunch the worker with them.")
                if is0:
                    os.replace(job["_path"],
                               qdir / "fail" / Path(job["_path"]).name)
                if world > 1:
                    dist.barrier()
                continue
            unknown = [k for k in job if k != "_path" and not hasattr(a, k)]
            if unknown:
                log(f"job {Path(job['_path']).stem}: ignoring unknown keys {unknown}")
            for k, v in job.items():
                if k != "_path" and hasattr(a, k):
                    setattr(a, k, v)
            log(f"--- job {Path(job['_path']).stem}: tag={a.tag} windows={a.windows} ---")
            t_job = time.time()
            try:
                run_take()
                ok = True
            except Exception as exc:
                ok = False
                log(f"job failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            if world > 1:
                dist.barrier()
            if is0:
                dest = qdir / ("done" if ok else "fail") / Path(job["_path"]).name
                os.replace(job["_path"], dest)
                n_done += 1
                log(f"[{time.time()-t_job:.0f}s] job {'ok' if ok else 'FAILED'} "
                    f"({n_done} served)")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
