#!/usr/bin/env bash

# Run this in the example directory to generate dummy data
mkdir -p data

> data/dict.txt
for IDX in {0..255} ; do
    echo "$IDX 1" >> data/dict.txt
done


> data/dummy.txt
> data/dummy_tgt.txt
for _ in {1..100} ; do
    # NOTE: these are of odd length, which with bos token becomes even length
    #       which is divisible by 2 (the shorten_factor, assertion in resampling_transformer_layer)
    echo "0 1 2 3 4" >> data/dummy.txt
    echo "5 4 3" >> data/dummy.txt
    echo "9 8 7 5 4" >> data/dummy.txt
    echo "5 4 3 2 1 1 8" >> data/dummy.txt
    echo "5 5 5 5 5 2 8 2 8" >> data/dummy.txt
    echo "5 5 5 5 5 2 8 2 8" >> data/dummy.txt

    echo "9 3 4" >> data/dummy_tgt.txt
    echo "9" >> data/dummy_tgt.txt
    echo "9" >> data/dummy_tgt.txt
    echo "9" >> data/dummy_tgt.txt
    echo "" >> data/dummy_tgt.txt
    echo "9" >> data/dummy_tgt.txt
done

set -ex

# binarize translation targets
fairseq-preprocess \
    --srcdict data/dict.txt \
    --destdir data \
    --trainpref data/dummy_tgt.txt \
    --validpref data/dummy_tgt.txt \
    --only-source

for SUFFIX in idx bin ; do
    cp data/train.$SUFFIX data/train.l1-l2.l2.$SUFFIX
    cp data/train.$SUFFIX data/valid.l1-l2.l2.$SUFFIX
done

cp data/dict.txt data/dict.l1.txt
cp data/dict.txt data/dict.l2.txt

# binarize source for lm
fairseq-preprocess \
    --srcdict data/dict.txt \
    --destdir data \
    --trainpref data/dummy.txt \
    --validpref data/dummy.txt \
    --only-source

# duplicate lm source as translation source
for SUFFIX in idx bin ; do
    cp data/train.$SUFFIX data/train.l1-l2.l1.$SUFFIX
    cp data/train.$SUFFIX data/valid.l1-l2.l1.$SUFFIX
done
