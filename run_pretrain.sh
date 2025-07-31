#!/bin/bash
set -euo pipefail
trap 'echo "Error occurred in $0 at line $LINENO"; exit 1' ERR

LOGDIR="logs/pretrain"
mkdir -p "$LOGDIR"

echo "Submitting GPT-2 pretraining job to SLURM..."
sbatch --output="${LOGDIR}/pretrain_%j.log" run_pretrain.hpc
echo "Submitted. Check $LOGDIR/pretrain_JobID.log for logs."