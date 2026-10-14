#!/bin/bash
#SBATCH --job-name=znm_demo
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/home/guangyi.liu/lynnreal/chunk-%j.out
set -o pipefail
export HOST=$(hostname)
cd /mnt/weka/home/guangyi.liu/lynnreal
echo "node=$(hostname) start=$(date -Is)"; nvidia-smi -L
source /mnt/weka/home/guangyi.liu/anaconda3/etc/profile.d/conda.sh
conda activate lynnreal
export HF_HUB_OFFLINE=1
ATTN=${LYNN_ATTN:-auto}
echo "=== 540p, 22-frame chunks, official VAE, attention=$ATTN resident=${LYNN_RES:-cpu} nostage=${LYNN_STAGE:-0} tag=${LYNN_TAG:-}"
python chunk_drive.py --chunks chunks/boulevard --out out_chunks/boulevard_540_${ATTN}${LYNN_TAG} --width 960 --height 544 --attention $ATTN --resident ${LYNN_RES:-cpu} ${LYNN_STAGE:+--no-stage} 2>&1 | grep -v "Loading\|Warning\|warn" | tail -20
echo "=== same with the lightweight VAE"
python chunk_drive.py --chunks chunks/boulevard --out out_chunks/boulevard_540_${ATTN}${LYNN_TAG}_lightvae --width 960 --height 544 --attention $ATTN --resident ${LYNN_RES:-cpu} ${LYNN_STAGE:+--no-stage} --light-vae LynnReal-Omni/weight/light-vae 2>&1 | grep -v "Loading\|Warning\|warn" | tail -20
echo "finished=$(date -Is)"
