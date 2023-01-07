# #!/usr/bin/env bash

export OC_CAUSE=1
export HYDRA_FULL_ERROR=1

NUM_BOTTOM=1
NUM_STAIR=3
NUM_TOP=1
POS_ENC=xpos

VALID_SUBSETS="test,valid.42-84"
# for RANGE in 101-200 201-300 301-400 401-500 ; do
#     VALID_SUBSETS="$VALID_SUBSETS,valid.$RANGE"
# done

TASK_NAME="parity_check"
EXPERIMENT="layers-$NUM_BOTTOM-$NUM_STAIR-$NUM_TOP.1000k.pos-$POS_ENC"
#export WANDB_NAME=$EXPERIMENT
export WANDB_NAME="layers-2.wd-002.seed-2"
export WANDB_RUN_GROUP="$TASK_NAME"
fairseq-hydra-train \
    --config-dir `pwd`/example/configs \
    --config-name staircase.chomsky \
    task.task_name="$TASK_NAME" \
    dataset.train_subset="valid.42-84" \
    dataset.valid_subset=\'$VALID_SUBSETS\' \
    +task.data=`pwd`/data \
    model.num_bottom_layers=$NUM_BOTTOM \
    model.num_staircase_layers=$NUM_STAIR \
    model.num_top_layers=$NUM_TOP \
    model.position_encoding=$POS_ENC \
    optimization.max_update=1000 \
    checkpoint.no_save=true \
    +model.use_fixed_chunking=true \
    checkpoint.save_dir=`pwd`/checkpoints/$EXPERIMENT \
    +common.user_dir=`pwd`/fairseq_user_dir

    # +common.wandb_project="name.testing" \
