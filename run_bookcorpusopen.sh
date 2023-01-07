#!/usr/bin/env bash
#SBATCH --mail-user=haukur@mideind.is
#SBATCH --output=/data/scratch/haukur/ponder/ponder-staircase/logs.slurm/bookcorpusopen-%J.out
#SBATCH --job-name=hbs-staircase-bookcorpus
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:00:00

# export OC_CAUSE=1
# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1

set -e

if [[ "$#" -ne 7 ]] ; then
    echo "Illegal number of arguments passed"
    exit 0
fi

PARTITION=$1
POS_ENC=$2
NUM_BOTTOM=$3
NUM_STAIR=$4
NUM_TOP=$5
FIXED_STRAT=$6
LEN_PARAM=$7

WDIR=`pwd`
SEED=1

. /home/haukur/miniconda3/etc/profile.d/conda.sh
conda activate ponder

# there are ~1.6b (gpt2) tokens in the dataset, we only use 3 out of 4 partitions to train
# so each model has 1.2b tokens
# with a eff. bsz=20_000 (200 toks * 100 seqs)
# this is ~57k steps
# 
# with a standard decoder with 10 layers, 640 embed_dim this achieves about 5 ups per second
# so a full 'epoch' is bit over 3 hours

# with layers-3-6-1 and fixed-chunk-length=1 we get 1.1 ups
# so a pessimistic staircase 'epoch' is 15 hours

# with layers-3-6-1 and stochastic-chunk-length=4 we get 3.2 ups

# this command:
#     ./run_bookcorpusopen.sh 1 xpos 3 6 1 false 4
# caused a crash after 57 steps and again at step 127 (seems to have heisenbug behavior)

WARMUP_UPDATES=570
if [[ $NUM_STAIR > 0 ]] ; then
    WARMUP_UPDATES=5700
fi


TASK_NAME="bookcorpusopen"
EXPERIMENT="part$PARTITION.layers-$NUM_BOTTOM-$NUM_STAIR-$NUM_TOP.pos-$POS_ENC.fixed_chunk-$FIXED_STRAT.lenparam$LEN_PARAM.warmup$WARMUP_UPDATES.seed$SEED"
export WANDB_NAME=$EXPERIMENT
export WANDB_RUN_GROUP="$TASK_NAME"
fairseq-hydra-train \
    --config-dir $WDIR/example/configs \
    --config-name staircase.bookcorpusopen \
    +task.data="$WDIR/data-bookcorpusopen/with-sym$PARTITION" \
    model.num_bottom_layers=$NUM_BOTTOM \
    model.num_staircase_layers=$NUM_STAIR \
    model.num_top_layers=$NUM_TOP \
    model.position_encoding=$POS_ENC \
    +model.use_fixed_chunking=$FIXED_STRAT \
    +model.chunk_length_parameter=$LEN_PARAM \
    lr_scheduler.warmup_updates=$WARMUP_UPDATES \
    checkpoint.save_dir=$WDIR/checkpoints/$TASK_NAME-$EXPERIMENT \
    common._name=bookcorpusopen.part$PARTITION \
    common.seed=$SEED \
    +common.wandb_project="bookcorpusopen.baseline" \
    +common.user_dir=$WDIR/fairseq_user_dir