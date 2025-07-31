#!/bin/bash
set -euo pipefail
trap 'echo "Error occurred in $0 at line $LINENO"; exit 1' ERR

LOGDIR="logs/ppo"
mkdir -p "$LOGDIR"

echo "Submitting PPO job to SLURM..."
sbatch --output="${LOGDIR}/ppo_train_%j.log" run_ppo.hpc
echo "Submitted. Check $LOGDIR/ppo_train_JobID.log for logs."