"""Retake34 for the diffusers MiniMax-H3 stack: pin the leading target-video rows.

CWM makes a 60s take by sliding a 124-frame window with a 34-frame overlap. Window k+1
regenerates everything except its first 34 frames, which are *given* -- they are the last
34 frames window k already produced. In latent terms 34 pixel frames is the first
VIDEO_PREFIX_LATENTS = 10 latent frames, and with patch (1,2,2) on a 48x84 latent that is
10 * (48//2) * (84//2) = 10080 packed rows.

The upstream musubi patch spends most of its 422 lines teaching the transformer to accept a
*per-row* timestep. diffusers already does that natively -- `AdaLayerNormContinuous` takes
`timestep_indices` and index_selects -- so the port reduces to two much smaller things:

  1. give the pinned rows timestep 1.0, which in this convention (`timestep == 1 - sigma`)
     means sigma = 0, i.e. "this row is already clean, do not denoise it";
  2. hold those rows at the clean latent for the whole loop, so the model always *reads*
     the real previous frames rather than its own partial reconstruction of them.

On (2) the reference does something that looks different but is not. It leaves the pinned
rows noisy in the latent buffer and overrides the velocity with `(clean - x_t)/sigma`;
since this scheduler forms `denoised = x_t + (1-t)*v` and then blends
`x_next = r*x_t + (1-r)*denoised`, that override drives `denoised` to exactly `clean` and
the trajectory lands on `clean` at the last step, where `r = 0`. Pinning the buffer instead
feeds the transformer the identical input at every step and reaches the identical endpoint,
with no division by a sigma that goes to zero. The two agree on everything the model sees.
"""

import torch

_S = {"rows": None, "n": 0, "installed": False}


def _pin(latents: torch.Tensor, ncv: int) -> None:
    n = _S["n"]
    if n:
        latents[ncv : ncv + n] = _S["rows"].to(device=latents.device, dtype=latents.dtype)


def install(prefix_rows: torch.Tensor) -> int:
    """Pin `prefix_rows` (num_rows, channels*prod(patch)) to the head of the target video.

    Returns the number of rows pinned. Call before the pipeline runs.
    """
    from diffusers.modular_pipelines.minimax_h3 import before_denoise as _bd
    from diffusers.modular_pipelines.minimax_h3 import denoise as _dn

    if prefix_rows.ndim != 2:
        raise ValueError(f"prefix rows must be 2-D (rows, features), got {tuple(prefix_rows.shape)}")
    _S["rows"] = prefix_rows.float().contiguous()
    _S["n"] = int(prefix_rows.shape[0])

    if _S["installed"]:
        return _S["n"]

    def build_row_timesteps(layout, video_timestep, audio_timestep,
                            condition_video_timestep, condition_audio_timestep):
        # Mirrors packing.build_row_timesteps, plus the pinned block. Kept as a copy rather
        # than a wrapper because the upstream return is already reduced through
        # torch.unique, and a new distinct timestep cannot be injected after that.
        row_timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32)
        row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
        row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
        row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
        n = _S["n"]
        if n:
            ncv = layout.num_condition_video_rows
            target = layout.video_indices.numel() - ncv
            if n > target:
                raise ValueError(f"pinned {n} rows but the target video only has {target}")
            row_timesteps[layout.video_indices[ncv : ncv + n]] = 1.0
        return torch.unique(row_timesteps, sorted=True, return_inverse=True)

    _bd.build_row_timesteps = build_row_timesteps

    for name in ("MiniMaxH3LoopSchedulerStep", "MiniMaxH3DenoiseLoopWrapper"):
        if not hasattr(_dn, name):
            raise RuntimeError(f"denoise module has no {name}; upstream renamed it, "
                               "so this patch would silently no-op")

    _orig_step = _dn.MiniMaxH3LoopSchedulerStep.__call__

    def scheduler_step(self, components, block_state, i, t):
        components, block_state = _orig_step(self, components, block_state, i, t)
        _pin(block_state.latents, block_state.num_condition_video_rows)
        return components, block_state

    _dn.MiniMaxH3LoopSchedulerStep.__call__ = scheduler_step

    _orig_loop = _dn.MiniMaxH3DenoiseLoopWrapper.__call__

    def loop(self, components, state):
        # Step 0 reads the latents before any after-denoiser has run, so pin here too.
        latents = state.get("latents")
        layout = state.get("layout")
        if latents is not None:
            if layout is None:
                raise RuntimeError("no layout in state; cannot locate the target video rows")
            # Off the layout, not off the `num_condition_video_rows` input, whose default of 0
            # would silently pin the *conditioning* rows instead of the target ones.
            _pin(latents, int(layout.num_condition_video_rows))
        return _orig_loop(self, components, state)

    _dn.MiniMaxH3DenoiseLoopWrapper.__call__ = loop

    _S["installed"] = True
    return _S["n"]
