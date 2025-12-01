#!/bin/bash
PREFIX="$1"    # Store the argument in variable
TRAIN_PATH="$2"
CKPT="$3"
OUTPUT_DIR="$4"
PROBE_TYPE="$5"
LABEL_TYPE="$6"
LOSS_TYPE="$7"
# if $8 is set, use it as TS
if [ -n "$8" ]; then
    TS="$8"
else
    TS=""
fi

SPLIT="llava_textvqa_val_v051_ocr"
GT="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
ANS_DIR="./playground/data/eval/textvqa/answers"

# mkdir $OUTPUT_DIR/textvqa

# python -m llava.eval.eval_textvqa \
#     --annotation-file $GT \
#     --result-file $ANS_DIR/$CKPT.jsonl
# python -m llava.eval.extract_textvqa_data \
#     --annotation-file $GT \
#     --result-file $ANS_DIR/$CKPT.jsonl \
#     --output-file $OUTPUT_DIR/textvqa/$CKPT.csv

# python -m llava.eval.eval_textvqa \
#     --annotation-file $GT \
#     --result-file $ANS_DIR/$PREFIX$CKPT.jsonl
# python -m llava.eval.extract_textvqa_data \
#     --annotation-file $GT \
#     --result-file $ANS_DIR/$PREFIX$CKPT.jsonl \
#     --output-file $OUTPUT_DIR/textvqa/$PREFIX$CKPT.csv

# interventional evaluation
# if [ "$LABEL_TYPE" = "both" ]; then
#     SAMPLE_OPTIONS="--do-sample --num-samples 100"
# else
#     SAMPLE_OPTIONS=""
# fi
SAMPLE_OPTIONS=""

# python -m llava.eval.select_vqa_strategy \
#     --result-dir $OUTPUT_DIR \
#     --intermediate-dir $TRAIN_PATH \
#     --task textvqa \
#     --prefix $PREFIX \
#     --model $CKPT $SAMPLE_OPTIONS \
#     --probe-type $PROBE_TYPE --label-type $LABEL_TYPE --loss-type $LOSS_TYPE

# test-only
python -m llava.eval.select_vqa_strategy \
    --result-dir $OUTPUT_DIR \
    --intermediate-dir $TRAIN_PATH \
    --task textvqa \
    --prefix $PREFIX \
    --model $CKPT $SAMPLE_OPTIONS \
    --probe-type $PROBE_TYPE --label-type $LABEL_TYPE --loss-type $LOSS_TYPE \
    --test-only --train-timestamp $TS

# test run
# python -m llava.eval.select_vqa_strategy \
#     --result-dir $OUTPUT_DIR \
#     --intermediate-dir $TRAIN_PATH \
#     --task textvqa \
#     --prefix $PREFIX \
#     --model $CKPT $SAMPLE_OPTIONS \
#     --probe-type $PROBE_TYPE --label-type $LABEL_TYPE --loss-type $LOSS_TYPE \
#     --epochs 1 --deactivate-wandb
