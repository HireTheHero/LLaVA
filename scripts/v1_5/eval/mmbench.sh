#!/bin/bash
if [ $# -eq 3 ]; then
    PREFIX="$1"    # Store the argument in variable
    TRAIN_PATH="$2"
    CKPT="$3"
elif [ $# -eq 1 ]; then
    PREFIX=""     # Store an empty string in variable
    TRAIN_PATH="dummy-not-used"
    CKPT="$1"
else
    PREFIX=""     # Store an empty string in variable
    TRAIN_PATH="dummy-not-used"
    CKPT="llava-v1.5-13b"
fi

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/$CKPT \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv,./playground/data/eval/mmbench/$PREFIX$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl,./playground/data/eval/mmbench/answers/$PREFIX$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --train-path $TRAIN_PATH \
    --extract-path ./playground/data/eval/mmbench/intermediate/$PREFIX$CKPT \
    --conv-mode vicuna_v1 --output-hidden-states --export-ids \
    --do-repr-sample --repr-sample-num 4000

## single input
mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT

## multiple inputs
mkdir -p playground/data/eval/mmbench/answers_upload/$PREFIX$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench/$PREFIX$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench/answers/$PREFIX$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench/answers_upload/$PREFIX$SPLIT \
#     --experiment $CKPT
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$PREFIX$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$PREFIX$SPLIT \
    --experiment $CKPT
