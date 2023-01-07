#!/usr/bin/env bash

if [[ "$#" -ne 2 ]] ; then
    echo "Illegal number of arguments passed"
    exit 0
fi

set -e 

TASK_NAME="$1"
PROJECT_DIR="$2"
DATA_DIR="$PROJECT_DIR/data"

echo "Using $PROJECT_DIR as a project dir"

function binarize_src_tgt {
    local SPLIT_NAME=$1
    local TMP_DIR="$DATA_DIR/tmp-$TASK_NAME"
    rm -f $TMP_DIR/dict.{src,tgt}.txt
    fairseq-preprocess \
        --source-lang src \
        --target-lang tgt \
        --workers 4 \
        --srcdict "$DATA_DIR/dict.txt" \
        --destdir "$TMP_DIR" \
        --trainpref "$DATA_DIR/$SPLIT_NAME.$TASK_NAME" \
        --joined-dictionary

    for SUFFIX in bin idx ; do
        for SIDE in src tgt ; do
            mv "$TMP_DIR/train.src-tgt.$SIDE.$SUFFIX" "$DATA_DIR/$SPLIT_NAME.$TASK_NAME.$SIDE.$SUFFIX"
        done
    done
}

echo "Generating $TASK_NAME"

# # this can take up to an hour per task when train_steps=1_000_000
# python $PROJECT_DIR/prepare_chomsky.py \
#     --base-dir=$PROJECT_DIR \
#     --data-dirname=data \
#     --task-name=$TASK_NAME \
#     --vocab-size=5 \
#     --modulus=5 \
#     --train-length-range=40 \
#     --test-length-range=540 \
#     --valid-ranges="42-84,101-200,201-300,301-400,401-500" \
#     --seed=1 \
#     --batch-size=128 \
#     --train-steps=1000000 \
#     --valid-size=1000 \
#     --test-size=5000

binarize_src_tgt "valid.42-84"
binarize_src_tgt "valid.101-200"
binarize_src_tgt "valid.201-300"
binarize_src_tgt "valid.301-400"
binarize_src_tgt "valid.401-500"
binarize_src_tgt "test"
binarize_src_tgt "train"
