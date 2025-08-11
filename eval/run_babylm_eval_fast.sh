#!/bin/bash
set -euo pipefail
trap 'echo "Error occurred in $0 at line $LINENO"; exit 1' ERR

# ----- Defaults -----
#MODEL_PATH="BabyLM-community/babylm-baseline-100m-gpt2"
#MODEL_PATH="BabyLM-community/babylm-interaction-baseline-simpo"
#MODEL_PATH="llm-slice/babylm-gpt2-small-90M-seed42"
MODEL_PATH="llm-slice/babylm-gpt2-small-90M-seed41_chck_200M_ppo-1600K-seed41"
BACKEND="causal"
EVAL_SCRIPT="eval_zero_shot_fast.sh"
LOGDIR="logs/eval_fast"
mkdir -p "$LOGDIR"

# List of all desired checkpoints (as strings, matching their revision_name)
#CHECKPOINTS=(chck_1M chck_2M chck_3M chck_4M chck_5M chck_6M chck_7M chck_8M chck_9M \
#chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M \
#chck_100M chck_200M chck_300M chck_400M chck_500M chck_600M chck_700M chck_800M chck_900M chck_1000M)

# Checkpoints for 90M PPO models
CHECKPOINTS=(chck_100K chck_200K chck_300K chck_400K chck_500K chck_600K chck_700K chck_800K chck_900K chck_1000K \
chck_1200K chck_1400K main)

CHECKPOINT_FILE="checkpoints.txt"
printf "%s\n" "${CHECKPOINTS[@]}" > "$CHECKPOINT_FILE"
N_CHECKPOINTS=${#CHECKPOINTS[@]}

# Submit to slurm
sbatch --export=MODEL_PATH=${MODEL_PATH},BACKEND=${BACKEND},EVAL_SCRIPT=${EVAL_SCRIPT} \
    --output="${LOGDIR}/babylm_eval_fast_%A_%a.log" \
    --array=0-$((${N_CHECKPOINTS}-1))%4 \
    run_babylm_eval_fast.hpc

echo "Submitted evaluation job for all checkpoints. Check $LOGDIR for logs."