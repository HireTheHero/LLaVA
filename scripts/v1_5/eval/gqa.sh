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

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/$CKPT \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl,./playground/data/eval/gqa/$PREFIX$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl,./playground/data/eval/gqa/answers/$PREFIX$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --train-path $TRAIN_PATH \
        --extract-path ./playground/data/eval/gqa/intermediate/$PREFIX$CKPT \
        --conv-mode vicuna_v1 --output-hidden-states --export-ids \
        --do-repr-sample --repr-sample-num 4000&
done

wait

## single input
output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
cd ../../../../../

## multiple inputs
output_file=./playground/data/eval/gqa/answers/$PREFIX$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$PREFIX$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
