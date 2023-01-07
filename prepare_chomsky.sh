#!/usr/bin/env bash

if [[ "$#" -ne 1 ]] ; then
    echo "Illegal number of arguments passed"
    exit 0
fi

set -e 

PROJECT_DIR="$1"

echo "Using $1 as a project dir"

for TASK_NAME in \
    parity_check even_pairs cycle_navigation \
    duplicate_string missing_duplicate_string modular_arithmetic_brackets \
    reverse_string binary_addition binary_multiplication \
    bucket_sort compute_sqrt odds_first solve_equation modular_arithmetic stack_manipulation \
    ; do
    sbatch -t 1-0 --job-name "hbs.$TASK_NAME" ./prepare_chomsky_task.sh "$TASK_NAME" "$PROJECT_DIR"
done
