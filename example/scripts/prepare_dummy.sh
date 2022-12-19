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
    echo "0 1 2 3 4" >> data/dummy.txt
    echo "5 4 3" >> data/dummy.txt
    echo "9 8 7 5 4" >> data/dummy.txt
    echo "5 4 3 2 1 1 8" >> data/dummy.txt
    echo "5 5 5 5 5 2 8 2 8" >> data/dummy.txt
    echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99" >> data/dummy.txt
    echo "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49" >> data/dummy.txt
    echo "10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58" >> data/dummy.txt

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
