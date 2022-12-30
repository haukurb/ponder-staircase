# #!/usr/bin/env bash

export OC_CAUSE=1
export HYDRA_FULL_ERROR=1

NUM_BOTTOM=5
NUM_STAIR=0
NUM_TOP=0
POS_ENC=xpos

# set -e
# for TASK_NAME in parity_check even_pairs cycle_navigation \
#         duplicate_string missing_duplicate_string modular_arithmetic_brackets \
#         reverse_string binary_addition binary_multiplication \
#         bucket_sort compute_sqrt odds_first solve_equation stack_manipulation modular_arithmetic ; \
#     do
#     EXPERIMENT="$TASK_NAME.layout-$NUM_BOTTOM-$NUM_STAIR-$NUM_TOP.100k.xpos.mask"
#     export WANDB_NAME=$EXPERIMENT
#     fairseq-hydra-train \
#         --config-dir `pwd`/example/configs \
#         --config-name staircase.chomsky \
#         task.task_name="$TASK_NAME" \
#         +task.data=`pwd`/data \
#         model.num_bottom_layers=2 \
#         model.num_staircase_layers=2 \
#         model.num_top_layers=1\
#         +common.wandb_project="dbg.staircase" \
#         +common.user_dir=`pwd`/fairseq_user_dir #2>&1 >> debug_sweep.txt
#         # checkpoint.restore_file=`pwd`/checkpoint_last.pt \
#         # checkpoint.save_dir=`pwd`/checkpoints/$EXPERIMENT \
    
#     exit 0
# done

TASK_NAME="even_pairs"
EXPERIMENT="layers-$NUM_BOTTOM-$NUM_STAIR-$NUM_TOP.1000k.pos-$POS_ENC"
export WANDB_NAME=$EXPERIMENT
fairseq-hydra-train \
    --config-dir `pwd`/example/configs \
    --config-name staircase.chomsky \
    task.task_name="$TASK_NAME" \
    +task.data=`pwd`/data \
    model.num_bottom_layers=$NUM_BOTTOM \
    model.num_staircase_layers=$NUM_STAIR \
    model.num_top_layers=$NUM_TOP \
    model.position_encoding=$POS_ENC \
    +common.wandb_project="$TASK_NAME" \
    optimization.max_update=1_000_000 \
    +common.user_dir=`pwd`/fairseq_user_dir