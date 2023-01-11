#!/usr/bin/env bash

if [[ "$#" -ne 1 ]] ; then
    echo "Illegal number of arguments passed"
fi

DATA_DIR="$1"
set -e

echo "Binarizing bookcorpusopen to data-dir: $1"

python prepare_hf_dataset.py export-bookcorpusopen --data-dir $DATA_DIR/

for SPLIT_NAME in partition1 partition2 partition3 partition4 ; do
    TMP_DIR=$DATA_DIR/tmp-$SPLIT_NAME
    mkdir -p $TMP_DIR
    rm -f $TMP_DIR/dict.txt
    # source-lang and only-source is necessary, otherwise srcdict is ignored
    fairseq-preprocess \
        --workers 16 \
        --source-lang txt \
        --destdir $TMP_DIR \
        --srcdict $DATA_DIR/dict.txt \
        --only-source \
        --trainpref "$DATA_DIR/$SPLIT_NAME.bookcorpusopen.ids" &
done

wait

for SPLIT_NAME in partition1 partition2 partition3 partition4 ; do
    TMP_DIR=$DATA_DIR/tmp-$SPLIT_NAME
    for SUFFIX in bin idx ; do
        mv "$TMP_DIR/train.txt-None.txt.$SUFFIX" "$DATA_DIR/$SPLIT_NAME.$SUFFIX"
    done
done