#!/usr/bin/env bash

if [ -z "$WDIR" ] ; then
    echo "environment variable WDIR is unset, please set before executing this script"
    exit 0
fi

pushd $WDIR

# git clone https://github.com/haukurb/transformer-sequential
# git clone https://github.com/haukurb/metaseq

. /home/haukur/miniconda3/etc/profile.d/conda.sh

# conda create -n ponder python=3.8 pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch # -c nvidia
conda create -n ponder python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.3 -c pytorch -c nvidia
conda activate ponder
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia  # this is too new for Ada's current cuda setup
# conda install pytorch torchvision torchaudio cudatoolkit=11.3.1 -c pytorch -c nvidia
pip install matplotlib tqdm
pip install tensorboard
pip install rotary-embedding-torch

# pushd transformer-sequential
# pip install -r requirements.txt
# popd

# mkdir -p staircase

git clone https://github.com/haukurb/ponder-staircase
pushd ponder-staircase

git clone https://github.com/deepmind/neural_networks_chomsky_hierarchy
pushd neural_networks_chomsky_hierarchy
pip install -r requirements.txt
popd

git clone https://github.com/sunyt32/torchscale
pushd torchscale
pip install -e .
popd
