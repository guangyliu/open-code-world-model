#!/bin/bash
#SBATCH --job-name=znm_demo
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/weka/home/guangyi.liu/lynnreal/batch-%j.out
# All six gallery scenes through LynnReal-Omni video-edit with our (solid-vehicle) proxy as
# the aligned control; boulevard additionally with the depth-only control.
set -o pipefail
# conda-installed toolchain activation scripts read $HOST; under `set -u` that aborts the job.
export HOST=$(hostname)
cd /mnt/weka/home/guangyi.liu/lynnreal/LynnReal-Omni
echo "node=$(hostname) start=$(date -Is)"; nvidia-smi -L
source /mnt/weka/home/guangyi.liu/anaconda3/etc/profile.d/conda.sh
conda activate lynnreal
export HF_HUB_OFFLINE=1
O=/mnt/weka/home/guangyi.liu/lynnreal/output_solid; mkdir -p $O
run() {  # run <scene> <ctl>
  local I=/mnt/weka/home/guangyi.liu/lynnreal/inputs_solid/$1
  [ -f $O/$1_$2.mp4 ] && { echo "skip $1 $2"; return; }
  echo "[$(date +%H:%M:%S)] === $1 control=$2 ==="
  python script/sample.py --weights weight/standard --mode video-edit \
      --prompt-file $I/prompt_edit.txt --reference $I/anchor.png --reference $I/proxy_$2.mp4 \
      --aligned-reference 1 --output $O/$1_$2.mp4 --cache $O/cache_$1_$2 \
      --frames 124 --steps 4 --seed 7 --attention-backend ${LYNN_ATTN:-auto} 2>&1 | grep -E "Measured|Error|error|Traceback" | tail -4
  [ -f $O/$1_$2.mp4 ] && echo "[$(date +%H:%M:%S)] ok $1 $2" || echo "[$(date +%H:%M:%S)] FAILED $1 $2"
}
for sc in boulevard causeway stream witchroad frostcourt monastery; do run $sc game; done
run boulevard depth
echo "finished=$(date -Is)"; ls -la $O/*.mp4
