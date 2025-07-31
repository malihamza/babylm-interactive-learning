#!/bin/bash

MODEL_PATH=$1
LR=${2:-3e-5}           # default: 3e-5
BSZ=${3:-32}            # default: 32
BIG_BSZ=${4:-16}        # default: 16
MAX_EPOCHS=${5:-10}     # default: 10
REVISION_NAME=${6:-"main"}  # default: "main"
SEED=${7:-42}           # default: 42

# WANDB parameters
WANDB_PROJECT="BLM_eval_finetuning"

model_basename=$(basename $MODEL_PATH)

for task in {boolq,multirc}; do
    EXP_NAME="$task"
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --revision_name "$REVISION_NAME" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BIG_BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --wandb \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --exp_name "$EXP_NAME" \
        --take_final \
        --verbose
done

EXP_NAME="rte"

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --revision_name "$REVISION_NAME" \
    --train_data "evaluation_data/full_eval/glue_filtered/rte.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --task rte \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --exp_name "$EXP_NAME" \
    --take_final \
    --verbose

EXP_NAME="wsc"

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --revision_name "$REVISION_NAME" \
    --train_data "evaluation_data/full_eval/glue_filtered/wsc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --task wsc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --exp_name "$EXP_NAME" \
    --take_final \
    --verbose

for task in {mrpc,qqp}; do
    EXP_NAME="$task"
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --revision_name "$REVISION_NAME" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid f1 \
        --seed $SEED \
        --wandb \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --exp_name "$EXP_NAME" \
        --take_final \
        --verbose
done

EXP_NAME="mnli"

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --revision_name "$REVISION_NAME" \
    --train_data "evaluation_data/full_eval/glue_filtered/mnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --task mnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --exp_name "$EXP_NAME" \
    --take_final \
    --verbose
