import pathlib
import time

import click
import numpy as np
import tqdm

from icecream import ic
from neural_networks_chomsky_hierarchy.training import constants
import haiku as hk


CHOMSKY_SUBTASK_CHOICES = [
    "modular_arithmetic",
    "parity_check",
    "even_pairs",
    "cycle_navigation",
    "duplicate_string",
    "missing_duplicate_string",
    "modular_arithmetic_brackets",
    "reverse_string",
    "binary_addition",
    "binary_multiplication",
    "bucket_sort",
    "compute_sqrt",
    "odds_first",
    "solve_equation",
    "stack_manipulation",
]


def write_dict(dict_path):
    # symbols = "0123456789abcdefghijklmnopqrstuvwxyz"
    numbers = [str(i) for i in range(10)]
    letters = list("abcdefghijklmnopqrstuvwxyz")
    symbols =  numbers + letters
    with open(dict_path, "w") as fh:
        for idx, symbol in enumerate(symbols):
            if idx > 0:
                fh.write("\n")
            fh.write(f"{symbol} 1")


def build_task(*, task_name, modulus,vocab_size):
    task_kwargs = {}
    if "solve_equation" == task_name or "modular" in task_name:
        task_kwargs["modulus"] = modulus
    if task_name in ("reverse_string", "duplicate_string", "odds_first"):
        task_kwargs["vocab_size"] = vocab_size
    task = constants.TASK_BUILDERS[task_name](**task_kwargs)
    return task


def generate_batch(*, jax_task, jax_rng_seq, length, batch_size, task_name):
    if task_name == "modular_arithmetic":
        item = jax_task.sample_batch(next(jax_rng_seq), sequence_length=length, batch_size=batch_size)
    else:
        item = jax_task.sample_batch(next(jax_rng_seq), length=length, batch_size=batch_size)
    assert sorted(list(item.keys())) == ["input", "output"]
    # first make jax/haiku tensor compatible with numpy
    one_hot_source = np.array(item["input"], dtype=np.int64)
    one_hot_targets = np.array(item["output"], dtype=np.int64)
    if len(one_hot_targets.shape) == 2:
        # sometimes the output is a scalar, sometimes a sequence
        # so we transform it into a sequence when it is only a scalar
        one_hot_targets = one_hot_targets[:, None, :]

    assert len(one_hot_targets.shape) == 3
    for seq_idx in range(batch_size):
        src_idxs = one_hot_source[seq_idx].nonzero()[-1]
        # view the index as a string (so [0, 1] becomes ["0 1"])
        src_stringified = " ".join(str(i) for i in src_idxs)

        tgt_idxs = one_hot_targets[seq_idx].nonzero()[-1]
        tgt_stringified = " ".join(str(i) for i in tgt_idxs)
        if task_name in ("binary_addition", "binary_multiplication", "stack_manipulation"):
            # we don't need padding at the end since we are using decoder only models
            # (the encoder models has to use use padding so as not to provide 
            #  output length hints as encoder inputs)
            yield src_stringified, tgt_stringified[:tgt_stringified.index("2")]
            continue
        # with regards to solve_equation task: the paper describes operation set {+,-,*}
        # but their code does not use *
        yield src_stringified, tgt_stringified


def generate_subset(*, src_path, tgt_path, subset_name, jax_task, min_len, max_len, num_batches, batch_size, seed, task_name):
    effective_seed = hash((seed, subset_name)) % (2**16 - 1)
    jax_rng_seq = hk.PRNGSequence(effective_seed)
    np.random.seed(effective_seed)
    batch_lengths = np.random.randint(min_len, max_len + 1, size=[num_batches]).tolist()

    with open(src_path, "w") as src_fh,\
            open(tgt_path, "w") as tgt_fh:
        for sampled_length in tqdm.tqdm(batch_lengths):
            for (src, tgt) in generate_batch(jax_task=jax_task, jax_rng_seq=jax_rng_seq, length=sampled_length, batch_size=batch_size, task_name=task_name):
                src_fh.write(src + "\n")
                tgt_fh.write(tgt + "\n")


@click.command()
@click.option("--base-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path))
@click.option("--data-dirname", type=str)
@click.option("--task-name", type=click.Choice(CHOMSKY_SUBTASK_CHOICES))
@click.option("--vocab-size", type=click.INT)
@click.option("--modulus", type=click.INT)
@click.option("--train-length-range", type=click.INT)
@click.option("--test-length-range", type=click.INT)
@click.option("--valid-ranges", type=str, help="comma separated list of length ranges: '101-200,201-300'")
@click.option("--seed", type=click.INT)
@click.option("--batch-size", type=click.INT)
@click.option("--train-steps", type=click.INT)
@click.option("--test-size", type=click.INT)
@click.option("--valid-size", type=click.INT)
def main(
    base_dir, data_dirname, task_name, vocab_size, modulus, train_length_range, test_length_range, 
    valid_ranges,
    seed, batch_size, train_steps, test_size, valid_size,
):
    ic(base_dir, data_dirname, task_name, vocab_size, modulus, train_length_range, test_length_range, seed, batch_size, train_steps, test_size)
    data_dir = base_dir / data_dirname
    data_dir.mkdir(exist_ok=True)

    write_dict(data_dir / "dict.txt")
    jax_task = build_task(task_name=task_name, modulus=modulus, vocab_size=vocab_size)

    generate_subset(
        src_path=data_dir / f"test.{task_name}.src", 
        tgt_path=data_dir / f"test.{task_name}.tgt", 
        subset_name="test",
        jax_task=jax_task, 
        min_len=train_length_range + 2,
        max_len=test_length_range,
        seed=seed,
        num_batches=test_size // 2, 
        batch_size=2,
        task_name=task_name,
    )

    for valid_range in valid_ranges.split(","):
        start, end = valid_range.split("-")
        start, end = int(start), int(end)
        assert start < end
        generate_subset(
            src_path=data_dir / f"valid.{start}-{end}.{task_name}.src", 
            tgt_path=data_dir / f"valid.{start}-{end}.{task_name}.tgt", 
            subset_name="valid",
            jax_task=jax_task, 
            min_len=start,
            max_len=end,
            seed=seed,
            num_batches=valid_size // 2, 
            batch_size=2,
            task_name=task_name,
        )

    generate_subset(
        src_path=data_dir / f"train.{task_name}.src", 
        tgt_path=data_dir / f"train.{task_name}.tgt", 
        subset_name="train",
        jax_task=jax_task, 
        min_len=1,
        max_len=train_length_range,
        seed=seed,
        num_batches=train_steps // 5, 
        batch_size=batch_size * 5,
        task_name=task_name,
    )


if __name__ == "__main__":
    main()