# #!/usr/bin/env bash

export OC_CAUSE=1
export HYDRA_FULL_ERROR=1

NUM_BOTTOM=2
NUM_STAIR=2
NUM_TOP=1

set -e
for TASK_NAME in parity_check even_pairs cycle_navigation \
        duplicate_string missing_duplicate_string modular_arithmetic_brackets \
        reverse_string binary_addition binary_multiplication \
        bucket_sort compute_sqrt odds_first solve_equation stack_manipulation modular_arithmetic ; \
    do

    export WANDB_NAME="$TASK_NAME.layout-$NUM_BOTTOM-$NUM_STAIR-$NUM_TOP"
    fairseq-hydra-train \
        --config-dir `pwd`/example/configs \
        --config-name staircase.chomsky \
        task.task_name="$TASK_NAME" \
        +task.data=`pwd`/data \
        model.num_bottom_layers=2 \
        model.num_staircase_layers=2 \
        model.num_top_layers=1\
        +common.user_dir=`pwd`/fairseq_user_dir #2>&1 >> debug_sweep.txt

        # +common.wandb_project="debug.staircase" \
    
    exit 0
done