#!/bin/bash
# Render one examples/ltx_inputs.py scene with LTX-2.5 + the Layout-To-Render IC-LoRA,
# using the recipe on the LoRA card: two-stage distilled with the IC-LoRA kept on stage 2,
# at 480p-class 896x512 by default (LTX_W / LTX_H; the card's own size is 1920x1088),
# seed 42, 24 fps. The card puts the first frame at temporal position -1;
# upstream main (9ec55f9) rejects any index outside [0, num_frames), so it goes in at 0.
#
#   sbatch --partition=main --gres=gpu:1 --cpus-per-task=16 --time=01:00:00 \
#          examples/ltx_run.sh out/ltx/boulevard
#
# 124 simulated frames are cut to 121 (8k+1), the length the pipeline accepts.
set -euo pipefail

SCENE=$(realpath "${1:?scene dir from examples/ltx_inputs.py}")
LTX=${LTX_ROOT:-/mnt/weka/home/guangyi.liu/code/ltx2.5-iclora/LTX-2}
M=${LTX_MODELS:-/mnt/weka/home/guangyi.liu/models/ltx-2.5}
LORA=${LTX_LORA:-/mnt/weka/home/guangyi.liu/models/ltx-2.5-layout-lora/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors}
OUT=$(realpath -m "${2:-$SCENE/ltx.mp4}")

cd "$LTX"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L || true
time "$LTX/.venv/bin/python" -m ltx_pipelines.ic_lora \
  --transformer-path       "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --text-encoder-path      "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path         "$M/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path         "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial-upsampler-path "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --lora "$LORA" 1.0 \
  --video-conditioning "$SCENE/clay.mp4" 1.0 \
  --stage-2-ic-lora \
  --image "$SCENE/first_frame.png" 0 1.0 \
  --prompt "$(cat "$SCENE/prompt.txt")" \
  --height "${LTX_H:-512}" --width "${LTX_W:-896}" --num-frames 121 --frame-rate 24 \
  --seed 42 \
  --output-path "$OUT"
echo "OUTPUT: $OUT"
