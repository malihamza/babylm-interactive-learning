#!/bin/bash
set -euo pipefail
trap 'echo "Error occurred in $0 at line $LINENO"; exit 1' ERR

# ----- Defaults -----
#MODEL_PATH="BabyLM-community/babylm-baseline-100m-gpt2"
#MODEL_PATH="BabyLM-community/babylm-interaction-baseline-simpo"
MODEL_PATH="llm-slice/babylm-gpt2-small-90M-seed42"
LR="3e-5"
BSZ="32"
BIG_BSZ="16"
MAX_EPOCHS="10"
REVISION_NAME="chck_1000M" # or 900 M
SEED="42"

# You can optionally parse command-line overrides here

logdir="logs/eval"
mkdir -p "$logdir"

echo "MODEL_PATH=$MODEL_PATH LR=$LR BSZ=$BSZ BIG_BSZ=$BIG_BSZ MAX_EPOCHS=$MAX_EPOCHS REVISION_NAME=$REVISION_NAME SEED=$SEED"
EXPORT_VARS="MODEL_PATH=${MODEL_PATH},LR=${LR},BSZ=${BSZ},BIG_BSZ=${BIG_BSZ},MAX_EPOCHS=${MAX_EPOCHS},REVISION_NAME=${REVISION_NAME},SEED=${SEED}"

echo "Submitting BabyLM evaluation to SLURM..."
sbatch --export="$EXPORT_VARS" \
       --output="$logdir/babylm_eval_%j.log" \
       run_babylm_finetune.hpc

echo "Submitted. Check $logdir for logs."
