#!/usr/bin/env bash
# One CWM Ref2VA run under 4-way context parallel, on the GPUs this node has spare.
# Uses the shared cond/VAE services but never the shared job queue: no H3_JOBS_DIR,
# no S.sweep(), no heartbeat under a live worker's name, own output directory.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

export CUDA_VISIBLE_DEVICES=${CWM_GPUS:-4,5,6,7}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# h3_svc_common reads H3_SVC_DIR at import time; its built-in default points at a
# /lustre path that does not exist on this cluster, so leaving it unset fails silently
# by writing requests nobody serves.
export H3_SVC_DIR=${H3_SVC_DIR:?set H3_SVC_DIR to the shared VAE service directory}
H3_RUNTIME=${H3_RUNTIME:?set H3_RUNTIME to the MiniMax-H3 runtime checkout}
export PYTHONPATH="$H3_RUNTIME:$H3_RUNTIME/h3_kit/code:${PYTHONPATH:-}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${CWM_PORT:-29751}
# The first step waits on the conditioner and the reference-encode service while the other
# ranks sit in a collective; the default 480 s NCCL watchdog fires before a cold run gets there.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

BIN=${CWM_BIN:-$(dirname "$(command -v python3)")}
LOG=${CWM_LOG:-cp_run.log}
NP=${CWM_NP:-4}

"$BIN/torchrun" --nproc_per_node="$NP" --master_port="$MASTER_PORT" \
    "${CWM_ENTRY:-run_cp.py}" "$@" > "$LOG" 2>&1
echo "exit=$? log=$LOG"
