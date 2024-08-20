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

QUESTION=llava-mm-vet.jsonl
ANSWERS=$CKPT.jsonl

python -m llava.eval.model_vqa \
    --model-path liuhaotian/$CKPT \
    --question-file ./playground/data/eval/mm-vet/$QUESTION,./playground/data/eval/mm-vet/$PREFIX$QUESTION \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$ANSWERS,./playground/data/eval/mm-vet/answers/$PREFIX$ANSWERS \
    --temperature 0 \
    --train-path $TRAIN_PATH \
    --extract-path ./playground/data/eval/mm-vet/intermediate/$PREFIX$CKPT \
    --conv-mode vicuna_v1 --output-hidden-states --export-ids

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$ANSWERS \
    --dst ./playground/data/eval/mm-vet/results/$ANSWERS

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$PREFIX$ANSWERS \
    --dst ./playground/data/eval/mm-vet/results/$PREFIX$ANSWERS

