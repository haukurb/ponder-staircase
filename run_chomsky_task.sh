#!/usr/bin/env bash
#SBATCH --mail-user=haukur@mideind.is
#SBATCH --output=/data/scratch/haukur/ponder/ponder-staircase/logs.slurm/slurm-output-%J-%j.out
#SBATCH --job-name=hbs
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:00:00

if [[ "$#" -ne 7 ]] ; then
    echo "Illegal number of arguments passed"
fi

TASK_NAME=$1
POS_ENC=$2
NUM_BOTTOM=$3
NUM_STAIR=$4
NUM_TOP=$5
NUM_STEPS=$6
SEED=$7


EXPERIMENT="layers-$NUM_BOTTOM-$NUM_STAIR-$NUM_TOP.pos-$POS_ENC.$TASK_NAME.steps$NUM_STEPS.wd0.01.seed$SEED"
echo "starting $EXPERIMENT"

. /home/haukur/miniconda3/etc/profile.d/conda.sh
conda activate ponder

WDIR=`pwd`

VALID_SUBSETS="test,valid.42-84"
for RANGE in 101-200 201-300 301-400 401-500 ; do
    VALID_SUBSETS="$VALID_SUBSETS,valid.$RANGE"
done

export WANDB_NAME=$EXPERIMENT
fairseq-hydra-train \
    --config-dir $WDIR/example/configs \
    --config-name staircase.chomsky \
    task.task_name="$TASK_NAME" \
    dataset.valid_subset=\'$VALID_SUBSETS\' \
    +task.data=$WDIR/data \
    model.num_bottom_layers=$NUM_BOTTOM \
    model.num_staircase_layers=$NUM_STAIR \
    model.num_top_layers=$NUM_TOP \
    model.position_encoding=$POS_ENC \
    +common.wandb_project="no-hp-search.$TASK_NAME" \
    optimization.max_update=$NUM_STEPS \
    +common.user_dir=$WDIR/fairseq_user_dir