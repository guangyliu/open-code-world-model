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
ATTN=_flash_3
run(){ tag=$1; shift; echo "=== $tag: $*"; python chunk_drive.py --out out_chunks/boulevard_540_$tag --width 960 --height 544 --attention $ATTN --resident cuda --no-stage "$@" 2>&1 | grep -v "Loading\|Warning\|warn" | tail -14; }
run hist22 --chunks chunks/boulevard --history
run c39 --chunks chunks/boulevard39 --frames 39
run c56 --chunks chunks/boulevard56 --frames 56
run single124 --chunks chunks/boulevard124 --frames 124
echo "finished=$(date -Is)"
