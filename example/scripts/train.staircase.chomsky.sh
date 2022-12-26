# #!/usr/bin/env bash

export OC_CAUSE=1
export HYDRA_FULL_ERROR=1

fairseq-hydra-train \
    --config-dir `pwd`/example/configs \
    --config-name staircase.chomsky \
    task.task_name=modular_arithmetic \
    +common.user_dir=`pwd`/fairseq_user_dir
