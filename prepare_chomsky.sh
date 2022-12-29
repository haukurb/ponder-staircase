#!/usr/bin/env bash

set -e 

PROJECT_DIR="$1"
DATA_DIR="$PROJECT_DIR/data"

echo "Using $1 as a project dir"

for TASK_NAME in parity_check even_pairs cycle_navigation \
    duplicate_string missing_duplicate_string modular_arithmetic_brackets \
    reverse_string binary_addition binary_multiplication \
    bucket_sort compute_sqrt odds_first solve_equation stack_manipulation modular_arithmetic ; \
    do

    echo "Generating $TASK_NAME"

    # this can take up to an hour per task when train_steps=1_000_000
    python $1/prepare_chomsky.py \
        --base-dir=$PROJECT_DIR \
        --data-dirname=data \
        --task-name=$TASK_NAME \
        --vocab-size=5 \
        --modulus=5 \
        --train-length-range=40 \
        --test-length-range=540 \
        --valid-length-range=140 \
        --seed=1 \
        --batch-size=128 \
        --train-steps=1000000 \
        --valid-size=5000 \
        --test-size=5000

    rm -f $DATA_DIR/tmp/dict.{src,tgt}.txt
    fairseq-preprocess \
        --source-lang src \
        --target-lang tgt \
        --workers 16 \
        --srcdict "$DATA_DIR/dict.txt" \
        --destdir "$DATA_DIR/tmp" \
        --trainpref "$DATA_DIR/train.$TASK_NAME" \
        --validpref "$DATA_DIR/valid.$TASK_NAME" \
        --testpref "$DATA_DIR/test.$TASK_NAME" \
        --joined-dictionary


    for SPLIT_NAME in train test valid ; do
        for SUFFIX in bin idx ; do
            for SIDE in src tgt ; do
                mv "$DATA_DIR/tmp/$SPLIT_NAME.src-tgt.$SIDE.$SUFFIX" "$DATA_DIR/$SPLIT_NAME.$TASK_NAME.$SIDE.$SUFFIX"
            done
        done
    done

done