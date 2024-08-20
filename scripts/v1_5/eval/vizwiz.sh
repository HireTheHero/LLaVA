#!/bin/bash
if [ $# -eq 3 ]; then
    PREFIX="$1"    # Store the argument in variable
    TRAIN_PATH="$2"
    CKPT="$3"
elif [ $# -eq 1 ]; then
    PREFIX=""     # Store an empty string in variable
    TRAIN_PATH="dummy"
    CKPT="$1"
else
    PREFIX=""     # Store an empty string in variable
    TRAIN_PATH="dummy"
    CKPT="llava-v1.5-13b"
fi

SPLIT="llava_test"
MODEL_DIR=${PREFIX%?}

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$CKPT \
    --question-file ./playground/data/eval/vizwiz/$SPLIT.jsonl,./playground/data/eval/vizwiz/$PREFIX$SPLIT.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl,./playground/data/eval/vizwiz/answers/$PREFIX$CKPT.jsonl \
    --temperature 0 \
    --train-path $TRAIN_PATH \
    --extract-path ./playground/data/eval/vizwiz/intermediate/$PREFIX$CKPT \
    --conv-mode vicuna_v1 --output-hidden-states --export-ids \
    --do-repr-sample --repr-sample-num 4000

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/$SPLIT.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$CKPT.json

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/$PREFIX$SPLIT.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$PREFIX$CKPT.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$PREFIX$CKPT.json
