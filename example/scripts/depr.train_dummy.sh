#!/usr/bin/env bash
#SBATCH --mail-user=haukzi@gmail.com
#SBATCH --output=/data/scratch/haukur/causal_lm/logs.slurm/slurm_out_%j.txt
#SBATCH --job-name=causal-lm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
##SBATCH --exclusive

set -e

SRC=is_IS

ENV_NAME=hier
echo "Using conda environment: $ENV_NAME"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" 

WDIR="/data/scratch/haukur/hierarchical"

SRC=is_IS

#TOKENS_PER_SAMPLE=512
TOKENS_PER_SAMPLE=512
BSZ_TOKENS=512
WARMUP_UPDATES=500
SAVE_EVERY=5000
MAX_UPDATES=20000  # used to end training
TOTAL_NUM_UPDATES=20000  # used for lr decay schedule
LR_RATE=1e-4
ACCUM=1
SAMPLE_BREAK_MODE=complete
EXPERIMENT="debug.seqlen$TOKENS_PER_SAMPLE.nseqs$BSZ_TOKENS.accum$ACCUM.lr$LR_RATE.warmup$WARMUP_UPDATES.totalnupdates$TOTAL_NUM_UPDATES.break$SAMPLE_BREAK_MODE"

CKPT_DIR="$WDIR/checkpoints/$EXPERIMENT"
TBOARD_LOG_DIR="$WDIR/logs.tensorboard"
LOG_DIR="$TBOARD_LOG_DIR/$EXPERIMENT"

export CUDA_LAUNCH_BLOCKING=1
fairseq-train example/data \
    --no-progress-bar \
    --log-format simple --log-interval 1 \
    --cpu \
    --store-ema \
    --task language_modeling --criterion cross_entropy \
        --train-subset "train" \
        --valid-subset "valid" \
        --skip-invalid-size-inputs-valid-test \
    --arch transformer_lm_gpt3_small \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-8 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR_RATE \
        --max-update $MAX_UPDATES \
        --total-num-update $TOTAL_NUM_UPDATES \
        --warmup-updates $WARMUP_UPDATES \
        --max-update=$MAX_UPDATES \
        --total-num-update=$MAX_UPDATES \
        --tokens-per-sample $TOKENS_PER_SAMPLE \
        --batch-size $BSZ_TOKENS \
        --update-freq $ACCUM \
    --save-dir $CKPT_DIR \
        --save-interval-updates $SAVE_EVERY \
        --validate-interval-updates 500 \
        --keep-interval-updates 8 \
        --no-save \
    --find-unused-parameters \
    | tee -a log.debug.txt

#    --decoder-layerdrop 0.33 \
