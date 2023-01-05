#!/usr/bin/env python
import pathlib

import click
import datasets as hf_datasets
import tqdm

from icecream import ic


@click.group()
def main():
    pass

@main.command()
@click.option("--data-dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path))
def export_enwik8(data_dir):
    """
    see https://raw.githubusercontent.com/kimiyoung/transformer-xl/master/prep_text8.py
     and https://github.com/facebookresearch/adaptive-span/issues/17
     for more information on this (zip file can be found at http://mattmahoney.net/dc/enwik8.zip)
    """
    ic(data_dir.resolve())
    data_dir.mkdir(exist_ok=True)
    train = hf_datasets.load_dataset("enwik8", split="train[:80%]")
    valid = hf_datasets.load_dataset("enwik8", split="train[80%:90%]")
    test = hf_datasets.load_dataset("enwik8", split="train[90%:]")
    from fairseq_cli.preprocess import _make_dataset
    dset_splits_prefixes = [
        (train, "train"),
        (valid, "valid"),
        (test, "test"),
    ]
    for dset_split, split_name in dset_splits_prefixes:
        with (data_dir / f"{split_name}.enwik8.txt").open("w") as out_fh:
            for idx, item in enumerate(dset_split):
                text = item["text"].strip().replace(" ", "_")
                if not text:
                    continue
                out_fh.write(" ".join(text))
                out_fh.write("\n")


@main.command()
@click.option("--data-dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path))
def export_wikitext_103_v1(data_dir):
    ic(data_dir.resolve())
    data_dir.mkdir(exist_ok=True)
    dset_dict = hf_datasets.load_dataset("wikitext", name="wikitext-103-v1")
    ic(dset_dict)
    dset_splits_prefixes = [
        (dset_dict["train"], "train"),
        (dset_dict["validation"], "valid"),
        (dset_dict["test"], "test"),
    ]
    for dset_split, split_name in dset_splits_prefixes:
        with (data_dir / f"{split_name}.wikitext-103-v1.txt").open("w") as out_fh:
            for idx, item in enumerate(dset_split):
                text = item["text"].strip()
                if not text:
                    continue
                out_fh.write(text)
                out_fh.write("\n")



@main.command()
@click.option("--data-dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path))
def export_bookcorpusopen(data_dir):
    ic(data_dir.resolve())
    data_dir.mkdir(exist_ok=True)
    dset_splits_prefixes = [
        ("train[:25%]", "partition1"),
        ("train[25%:50%]", "partition2"),
        ("train[50%:75%]", "partition3"),
        ("train[75%:]", "partition4"),
    ]
    ic(dset_splits_prefixes)
    from transformers import AutoTokenizer
    subword_enc = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)

    def encode_to_stringified_ids(line):
        return " ".join(str(i) for i in subword_enc.encode(line))
    
    # since we have a defined dict
    with (data_dir / "dict.txt").open("w") as fh_out:
        for idx in range(subword_enc.vocab_size):
            fh_out.write(f"{idx} 1\n")

    for split_spec, split_name in dset_splits_prefixes:
        dset_split = hf_datasets.load_dataset("bookcorpusopen", split=split_spec)
        ic(dset_split)

        filename = f"{split_name}.bookcorpusopen.ids.txt"

        with (data_dir / filename).open("w") as out_fh:
            for idx, item in enumerate(tqdm.tqdm(dset_split)):
                text = item["text"].strip()
                if not text:
                    continue
                for line in text.split("\n"):
                    if not line:
                        continue
                    out_fh.write(encode_to_stringified_ids(line))
                    out_fh.write("\n")

if __name__ == "__main__":
    main()