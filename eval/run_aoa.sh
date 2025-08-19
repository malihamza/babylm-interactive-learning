#!/bin/bash
set -euo pipefail
trap 'echo "Error occurred in $0 at line $LINENO"; exit 1' ERR

#MODEL_PATH="BabyLM-community/babylm-baseline-100m-gpt2"
#MODEL_PATH="BabyLM-community/babylm-interaction-baseline-simpo"
#MODEL_PATH="llm-slice/blm-gpt2s-90M-s42"
#MODEL_PATH="llm-slice/blm-gpt2s-90M-s42_chck_200M_ppo-1000K-seed42"
MODEL_PATH="llm-slice/blm-gpt2s-90M-s42_901M-s42_submission"
#REVISION_NAME="main" # main, chck_1000M, or chck_900M
BACKEND="causal"
EVAL_SCRIPT="eval_aoa.sh"
LOGDIR="logs/eval"
mkdir -p "$LOGDIR"

sbatch --export=MODEL_PATH="${MODEL_PATH}",BACKEND="${BACKEND}",EVAL_SCRIPT="${EVAL_SCRIPT}" \
    --output="${LOGDIR}/babylm_aoa_%j.log" \
    run_babylm_eval.hpc

echo "Submitted aoa eval job. Check $LOGDIR for logs."