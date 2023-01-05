#!/usr/bin/env bash

if [[ "$#" -ne 1 ]] ; then
    echo "Illegal number of arguments passed"
fi

DATA_DIR="$1"
set -e

echo "Binarizing bookcorpusopen to data-dir: $1"

python prepare_hf_dataset.py export-bookcorpusopen --data-dir $DATA_DIR/

mkdir -p $DATA_DIR/tmp
for SPLIT_NAME in partition1 partition2 partition3 partition4 ; do
    rm -f $DATA_DIR/tmp/dict.txt
    fairseq-preprocess \
        --workers 8 \
        --destdir "$DATA_DIR/tmp" \
        --srcdict $DATA_DIR/dict.txt \
        --trainpref "$DATA_DIR/$SPLIT_NAME.bookcorpusopen.ids.txt"

    for SUFFIX in bin idx ; do
        mv "$DATA_DIR/tmp/train.None-None.$SUFFIX" "$DATA_DIR/$SPLIT_NAME.$SUFFIX"
    done
done

