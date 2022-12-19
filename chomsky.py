from icecream import ic
from neural_networks_chomsky_hierarchy.training import constants
import numpy as np
import haiku as hk


seed = 1
rng_seq = hk.PRNGSequence(hash((seed, 1337)))

all_task_names = ['modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation', 'modular_arithmetic_brackets', 'reverse_string', 'missing_duplicate_string', 'duplicate_string', 'binary_addition', 'binary_multiplication', 'compute_sqrt', 'odds_first', 'solve_equation', 'stack_manipulation', 'bucket_sort']
task_name = "even_pairs"

for task_name in all_task_names[1:]:
    kwargs = {}
    if  "solve_equation" == task_name:
        kwargs["modulus"] = 3
    if "modular" in task_name:
        kwargs["modulus"] = 3
    if task_name in ("reverse_string", "duplicate_string", "odds_first"):
        kwargs["vocab_size"] = 3

    ic(task_name)
    task = constants.TASK_BUILDERS[task_name](**kwargs)

    ic(task.sample_batch(next(rng_seq), length=5, batch_size=1))

ic(next(rng_seq), type(next(rng_seq)))
ic(constants.TASK_BUILDERS.keys())

