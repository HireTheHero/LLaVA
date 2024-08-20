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

SPLIT="llava_textvqa_val_v051_ocr"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$CKPT \
    --question-file ./playground/data/eval/textvqa/$SPLIT.jsonl,./playground/data/eval/textvqa/$PREFIX$SPLIT.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl,./playground/data/eval/textvqa/answers/$PREFIX$CKPT.jsonl \
    --temperature 0 \
    --train-path $TRAIN_PATH \
    --extract-path ./playground/data/eval/textvqa/intermediate/$PREFIX$CKPT \
    --conv-mode vicuna_v1 --output-hidden-states --export-ids

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$PREFIX$CKPT.jsonl
