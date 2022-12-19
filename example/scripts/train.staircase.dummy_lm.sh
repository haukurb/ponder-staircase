# #!/usr/bin/env bash

export OC_CAUSE=1
export HYDRA_FULL_ERROR=1

fairseq-hydra-train \
    --config-dir `pwd`/example/configs \
    --config-name staircase.dummy_lm \
    common.cpu="true" \
    task.data=`pwd`/example/data \
    +common.user_dir=`pwd`/fairseq_user_dir
