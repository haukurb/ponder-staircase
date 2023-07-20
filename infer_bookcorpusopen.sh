#!/usr/bin/env bash
#SBATCH --mail-user=haukur@mideind.is
#SBATCH --output=/data/scratch/haukur/ponder/ponder-staircase/logs.slurm/bookcorpusopen-%J.out
#SBATCH --job-name=hbs-staircase-bookcorpus
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:00:00

# XXX: change output dest
# export OC_CAUSE=1
# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1

set -e

WDIR=`pwd`
SEED=1
PARTITION=1

. /home/haukur/miniconda3/etc/profile.d/conda.sh
conda activate ponder

# valid/ppl=34.78
REF_CKPT=/data/scratch/haukur/ponder/ponder-staircase/checkpoints/bookcorpusopen-part1.layers-12-0-0.pos-xpos.fixed_chunk-true.lenparam2000.warmup570.polynomial_decay.seed1.lr3e-4/checkpoint1.pt
# # valid/ppl=34.08
# REF_CKPT=/data/scratch/haukur/ponder/ponder-staircase/checkpoints/bookcorpusopen-part1.layers-12-0-0.pos-xpos.fixed_chunk-true.lenparam2000.warmup570.polynomial_decay.seed1.lr3e-4.val_chunk-true.val_lenparam2000.ctx-1.rollstairs0.nepochs1.rstride0.prenorm/checkpoint1.pt


# valid/ppl=33.7
# POT_CKPT=/data/scratch/haukur/ponder/ponder-staircase/checkpoints/bookcorpusopen-part1.layers-1-10-1.pos-xpos.fixed_chunk-true.lenparam1.warmup570.polynomial_decay.seed1.lr3e-4.val_chunk-true.val_lenparam1.ctx-1.rollstairs0.nepochs1/checkpoint1.pt
# valid/ppl=33.37
# POT_CKPT=/data/scratch/haukur/ponder/ponder-staircase/checkpoints/bookcorpusopen-part1.layers-1-10-1.pos-xpos.fixed_chunk-true.lenparam1.warmup570.polynomial_decay.seed1.lr3e-4.val_chunk-true.val_lenparam1.ctx-1.rollstairs0.nepochs1.rstride2/checkpoint1.pt
# valid/ppl=32.35
POT_CKPT=/data/scratch/haukur/ponder/ponder-staircase/checkpoints/bookcorpusopen-part1.layers-1-10-1.pos-xpos.fixed_chunk-true.lenparam2.warmup570.polynomial_decay.seed1.lr3e-4.val_chunk-true.val_lenparam2.ctx-1.rollstairs0.nepochs1.rstride10/checkpoint1.pt
# valid/ppl=31.53 (at ckpt 52250)
POT_CKPT=/data/scratch/haukur/ponder/ponder-staircase/checkpoints/bookcorpusopen-part1.layers-1-10-1.pos-xpos.fixed_chunk-true.lenparam1.warmup570.polynomial_decay.seed1.lr3e-4.val_chunk-true.val_lenparam1.ctx-1.rollstairs0.nepochs1.rstride10/checkpoint1.pt


python fairseq_user_dir/compare_lm.py explore \
    --fairseq-user-dir $WDIR/fairseq_user_dir \
    --reference-checkpoint-path $REF_CKPT \
    --potential-checkpoint-path $POT_CKPT \
    --data $WDIR/data-bookcorpusopen/with-sym1