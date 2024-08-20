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

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/$CKPT \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl,./playground/data/eval/vqav2/$PREFIX$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl,./playground/data/eval/vqav2/answers/$PREFIX$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --train-path $TRAIN_PATH \
        --extract-path ./playground/data/eval/vqav2/intermediate/$PREFIX$SPLIT/$CKPT \
        --conv-mode vicuna_v1 --output-hidden-states --export-ids \
        --do-repr-sample --repr-sample-num 4000 &
done

wait

## single input
output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

## multiple input
output_file=./playground/data/eval/vqav2/answers/$PREFIX$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$PREFIX$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $PREFIX$SPLIT --ckpt $CKPT

