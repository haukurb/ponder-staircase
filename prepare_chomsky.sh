#!/usr/bin/env bash

if [[ "$#" -ne 1 ]] ; then
    echo "Illegal number of arguments passed"
    exit 0
fi

set -e 

PROJECT_DIR="$1"
# DATA_DIR="$PROJECT_DIR/data"

echo "Using $1 as a project dir"

# function binarize_src_tgt {
#     local TASK_NAME_=$1
#     local SPLIT_NAME_=$2
#     rm -f $DATA_DIR/tmp/dict.{src,tgt}.txt
#     fairseq-preprocess \
#         --source-lang src \
#         --target-lang tgt \
#         --workers 16 \
#         --srcdict "$DATA_DIR/dict.txt" \
#         --destdir "$DATA_DIR/tmp" \
#         --trainpref "$DATA_DIR/$SPLIT_NAME_.$TASK_NAME_" \
#         --joined-dictionary

#     for SUFFIX in bin idx ; do
#         for SIDE in src tgt ; do
#             mv "$DATA_DIR/tmp/train.src-tgt.$SIDE.$SUFFIX" "$DATA_DIR/$SPLIT_NAME_.$TASK_NAME_.$SIDE.$SUFFIX"
#         done
#     done
# }

# for TASK_NAME in \
#     parity_check even_pairs cycle_navigation \
#     duplicate_string missing_duplicate_string modular_arithmetic_brackets \
#     reverse_string binary_addition binary_multiplication \
#     bucket_sort compute_sqrt odds_first solve_equation stack_manipulation modular_arithmetic \
#     ; do

#     echo "Generating $TASK_NAME"

#     # this can take up to an hour per task when train_steps=1_000_000
#     python $PROJECT_DIR/prepare_chomsky.py \
#         --base-dir=$PROJECT_DIR \
#         --data-dirname=data \
#         --task-name=$TASK_NAME \
#         --vocab-size=5 \
#         --modulus=5 \
#         --train-length-range=40 \
#         --test-length-range=540 \
#         --valid-ranges="42-84,101-200,201-300,301-400,401-500" \
#         --seed=1 \
#         --batch-size=128 \
#         --train-steps=1000000 \
#         --valid-size=1000 \
#         --test-size=5000

#     binarize_src_tgt "$TASK_NAME.042-084" "valid"
#     binarize_src_tgt "$TASK_NAME.101-200" "valid"
#     binarize_src_tgt "$TASK_NAME.201-300" "valid"
#     binarize_src_tgt "$TASK_NAME.301-400" "valid"
#     binarize_src_tgt "$TASK_NAME.401-500" "valid"
#     binarize_src_tgt "$TASK_NAME" "train"
#     binarize_src_tgt "$TASK_NAME" "test"

# done

for TASK_NAME in \
    parity_check even_pairs cycle_navigation \
    duplicate_string missing_duplicate_string modular_arithmetic_brackets \
    reverse_string binary_addition binary_multiplication \
    bucket_sort compute_sqrt odds_first solve_equation stack_manipulation modular_arithmetic \
    ; do
    sbatch -t 1-0 --job-name "hbs.$TASK_NAME" ./prepare_chomsky_task.sh $TASK_NAME .
done
