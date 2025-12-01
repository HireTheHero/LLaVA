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

TIER="testdev_balanced"
SPLIT="llava_gqa_$TIER"
GQADIR="./playground/data/eval/gqa/data"

# ## single input
# output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# python scripts/convert_gqa_for_eval.py --src $output_file --dst "$GQADIR/"$TIER"_"$CKPT"_predictions.json"

# cd $GQADIR
# python eval/eval.py --tier $TIER --ckpt $CKPT
# python eval/extract_gqa_data.py --tier $TIER --ckpt $CKPT --output $OUTPUT_DIR
# cd ../../../../../

# ## multiple inputs
# output_file=./playground/data/eval/gqa/answers/$PREFIX$SPLIT/$CKPT/merge.jsonl

# python scripts/convert_gqa_for_eval.py --src $output_file --dst "$GQADIR/"$PREFIX$TIER"_"$CKPT"_predictions.json"

cd $GQADIR
# python eval/eval.py --tier $TIER --ckpt $CKPT --prefix $PREFIX
# python eval/extract_gqa_data.py --tier $TIER --ckpt $CKPT --prefix $PREFIX --output $OUTPUT_DIR

# interventional evaluation
# if [ "$LABEL_TYPE" = "both" ]; then
#     SAMPLE_OPTIONS="--do-sample --num-samples 100"
# else
#     SAMPLE_OPTIONS=""
# fi
SAMPLE_OPTIONS=""

# python eval/select_vqa_strategy.py \
#     --result-dir $OUTPUT_DIR \
#     --intermediate-dir $TRAIN_PATH \
#     --task gqa \
#     --prefix $PREFIX \
#     --model $CKPT $SAMPLE_OPTIONS \
#     --probe-type $PROBE_TYPE --label-type $LABEL_TYPE --loss-type $LOSS_TYPE

# test-only
python eval/select_vqa_strategy.py \
    --result-dir $OUTPUT_DIR \
    --intermediate-dir $TRAIN_PATH \
    --task gqa \
    --prefix $PREFIX \
    --model $CKPT $SAMPLE_OPTIONS \
    --probe-type $PROBE_TYPE --label-type $LABEL_TYPE --loss-type $LOSS_TYPE \
    --test-only --train-timestamp $TS

# test run
# python eval/select_vqa_strategy.py \
#     --result-dir $OUTPUT_DIR \
#     --intermediate-dir $TRAIN_PATH \
#     --task gqa \
#     --prefix $PREFIX \
#     --model $CKPT $SAMPLE_OPTIONS \
#     --probe-type $PROBE_TYPE --label-type $LABEL_TYPE --loss-type $LOSS_TYPE \
#     --epochs 1 --deactivate-wandb
