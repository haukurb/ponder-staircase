# #!/usr/bin/env bash

export OC_CAUSE=1
export HYDRA_FULL_ERROR=1

# tasks that fail in commit a5822c6 of chomsky repo:
#   modular_arithmetic
set -e
# for TASK_NAME in parity_check even_pairs cycle_navigation \
#         duplicate_string missing_duplicate_string modular_arithmetic_brackets \
#         reverse_string binary_addition binary_multiplication \
#         bucket_sort compute_sqrt odds_first solve_equation stack_manipulation modular_arithmetic ; \
#     do
for TASK_NAME in modular_arithmetic ; do
    fairseq-hydra-train \
        --config-dir `pwd`/example/configs \
        --config-name staircase.chomsky \
        task.task_name="$TASK_NAME" \
        +task.data=`pwd`/data \
        +common.user_dir=`pwd`/fairseq_user_dir #2>&1 >> debug_sweep.txt

done