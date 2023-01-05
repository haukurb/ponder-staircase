from dataclasses import dataclass, field, MISSING
from typing import Optional
from pathlib import Path
import time

from fairseq.data import data_utils
from fairseq.data.data_utils import collate_tokens
import numpy as np
import torch
import haiku as hk
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.data import Dictionary
from fairseq.data.data_utils import numpy_seed
from fairseq.tasks import FairseqTask, LegacyFairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
from icecream import ic
from neural_networks_chomsky_hierarchy.training import constants
from omegaconf import DictConfig, II
import logging

logger = logging.getLogger(__name__)

CHOMSKY_SUBTASK_CHOICES = ChoiceEnum(
    [
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
)


@dataclass
class ChomskyHierarchyTaskConfig(FairseqDataclass):
    data: str = field(default=".")
    task_name: CHOMSKY_SUBTASK_CHOICES = field(  # type: ignore
        default="parity_check",
        metadata={
            "help": "Sequence-to-sequence tasks belonging to differing levels in the Chomsky Hierarchy."
        },
    )
    modulus: int = field(
        default=5,
        metadata={
            "help": "Modulus for 'modular_arithmetic', 'modular_arithmetic_brackets', and 'solve_equation' tasks."
        },
    )
    vocab_size: int = field(
        default=2,
        metadata={
            "help": "Vocab size for 'reverse_string', 'duplicate_string', and 'odds_first' tasks."
        },
    )
    length_training_range: int = field(
        default=40,
        metadata={
            "help": "Maximum length of samples that are sampled during training."
        },
    )
    length_eval_range: int = field(
        default=540,
        metadata={
            "help": (
                "Maximum length of samples during evaluation, note that all eval"
                "samples are of length at least 'length_training_range'."
            )
        },
    )
    seed: int = II("common.seed")
    batch_size: int = II("dataset.batch_size")
    epoch_size: int = field(
        default=10_000,
        metadata={"help": ("Total number of sequences in training epoch")},
    )
    test_size: int = field(
        default=2,
        metadata={"help": ("Total number of sequences in test set")},
    )
    use_encoder_decoder_format: bool = field(
        default=False,
        metadata={
            "help": ("Put input sequence into source instead of prev_output_tokens")
        },
    )
    allow_loss_on_problem_statement: bool = field(
        default=False,
        metadata={
            "help": (
                "Only include problem solution in cross-entropy loss (discard loss on input part)"
            )
        },
    )

    def __post_init__(self):
        if self.seed == II("common.seed"):
            self.seed = 1
        if self.batch_size == II("data.batch_size"):
            self.batch_size = 128


class ChomskyIndexedDataset(FairseqDataset):
    def __init__(
        self, cfg: ChomskyHierarchyTaskConfig, dictionary, *, src_path, tgt_path
    ):
        super().__init__()
        self.cfg = cfg
        self.dictionary = dictionary
        self.task_name = str(self.cfg.task_name)
        self.seed = self.cfg.seed
        self.src_dataset = data_utils.load_indexed_dataset(
            str(src_path),
            self.dictionary,
            "mmap",
            combine=False,
        )
        self.tgt_dataset = data_utils.load_indexed_dataset(
            str(tgt_path),
            self.dictionary,
            "mmap",
            combine=False,
        )
        # we make this a sequence so we can put it directly into torch.cat
        self.separator_seq = torch.tensor([self.dictionary.bos()]).long()
        self.eos_seq = torch.tensor([self.dictionary.bos()]).long()
        self.eos_idx = self.dictionary.eos()
        assert (
            self.src_dataset is not None
        ), f"Expected to find dataset at {src_path}.bin"
        assert (
            self.tgt_dataset is not None
        ), f"Expected to find dataset at {tgt_path}.bin"

    def __getitem__(self, index):
        src_tokens = self.src_dataset[index]
        # tgt may or may not have eos already
        target = self.tgt_dataset[index]
        # strip eos if it is there (to ensure uniformity)
        if target[-1] == self.eos_idx:
            target = target[:-1]
        separator = torch.tensor([self.dictionary.bos()]).long()
        eos = torch.tensor([self.dictionary.eos()]).long()

        output = {
            "target": torch.cat([src_tokens, self.separator_seq, target, self.eos_seq]),
            "id": index,
        }
        # move eos so that it is in front, for teacher forcing
        output["src_tokens"] = output["target"].roll(1)

        if self.cfg.allow_loss_on_problem_statement:
            output["loss_keep_mask"] = torch.ones_like(output["target"])
        else:
            output["loss_keep_mask"] = torch.cat(
                [
                    torch.zeros_like(src_tokens),
                    torch.zeros_like(self.separator_seq),
                    torch.ones_like(target),
                    torch.ones_like(self.eos_seq),
                ]
            )
        return output

    def __len__(self):
        return len(self.src_dataset)

    def size(self, index):
        return self.src_dataset.sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.src_dataset.sizes[indices]

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        all_keys = list(samples[0].keys())
        collated = {}
        if "src_tokens" in all_keys:
            collated["src_tokens"] = collate_tokens(
                [s["src_tokens"] for s in samples], pad_idx=self.dictionary.pad()
            )
        collated["id"] = torch.LongTensor([s["id"] for s in samples])
        collated["target"] = collate_tokens(
            [s["target"] for s in samples], pad_idx=self.dictionary.pad()
        )
        if "prev_output_tokens" in all_keys:
            collated["prev_output_tokens"] = collate_tokens(
                [s["prev_output_tokens"] for s in samples],
                pad_idx=self.dictionary.pad(),
            )
        collated["loss_keep_mask"] = collate_tokens(
            [s["loss_keep_mask"] for s in samples], pad_idx=0
        )
        net_input_keys = ("src_tokens", "prev_output_tokens")
        res = {key: collated[key] for key in all_keys if key not in net_input_keys}
        res["net_input"] = {
            key: collated[key] for key in net_input_keys if key in all_keys
        }
        ntokens_key = "src_tokens" if "src_tokens" in all_keys else "prev_output_tokens"
        res["ntokens"] = sum(len(s[ntokens_key]) for s in samples)
        res["nsentences"] = len(samples)
        return res


class ChomskyDataset(FairseqDataset):
    def __init__(self, cfg: ChomskyHierarchyTaskConfig, dictionary, *, is_training):
        super().__init__()
        self.cfg = cfg
        self.task_name = str(self.cfg.task_name)
        self.seed = self.cfg.seed
        self.batch_size = self.cfg.batch_size
        self.dictionary = dictionary
        self._sizes = []
        self.is_training = is_training
        self._length = self.cfg.epoch_size if self.is_training else self.cfg.test_size
        task_kwargs = {}
        if "solve_equation" == self.task_name:
            task_kwargs["modulus"] = self.cfg.modulus
        if "modular" in self.task_name:
            task_kwargs["modulus"] = self.cfg.modulus
        if self.task_name in ("reverse_string", "duplicate_string", "odds_first"):
            task_kwargs["vocab_size"] = self.cfg.vocab_size

        if task_name == "modular_arithmetic":
            self._jax_chomsky_task = constants.TASK_BUILDERS[self.task_name](
                **task_kwargs
            )
        else:
            self._jax_chomsky_task = constants.TASK_BUILDERS[self.task_name](
                modulus=self.cfg.modulus, operators=("+", "*", "-")
            )
        self.length_range = self.cfg.length_training_range
        if not self.is_training:
            self.length_range = self.cfg.length_eval_range
        self.set_epoch(1)
        self.items = []

    def __getitem__(self, index):
        if self.items:
            return self.items[index]
        """
        XXX: we could either sample one just in time

        or we could build all of them ahead of time (like 100k or something)
        """
        item_length = self._sizes[index]
        rng_seq = hk.PRNGSequence(hash((self.seed, self.epoch, index)))
        if self.task_name == "modular_arithmetic":
            item = self._jax_chomsky_task.sample_batch(
                next(rng_seq), sequence_length=item_length, batch_size=1
            )
        else:
            item = self._jax_chomsky_task.sample_batch(
                next(rng_seq), length=item_length, batch_size=1
            )
        del rng_seq
        assert sorted(list(item.keys())) == ["input", "output"]
        # make jax/haiku tensor compatible with numpy then transform one-hot into index
        #    we index with 0 to get sequence since we have batch_size=1
        src_idxs = np.array(item["input"][0], dtype=np.int64).nonzero()[1]
        # view the index as a string (so [0, 1] becomes ["0 1"])
        src_stringified = " ".join(str(i) for i in src_idxs)

        # do the same for the targets
        one_hot_targets = np.array(item["output"][0], dtype=np.int64)
        if len(one_hot_targets.shape) == 1:
            # sometimes the output is a scalar, sometimes a sequence
            # so we transform it into a sequence when it is only a scalar
            one_hot_targets = one_hot_targets[None, :]
        assert len(one_hot_targets.shape) == 2

        tgt_idxs = one_hot_targets.nonzero()[1]
        tgt_stringified = " ".join(str(i) for i in tgt_idxs)

        if self.cfg.use_encoder_decoder_format:
            fairseq_item = {
                "id": index,
                "src_tokens": self.dictionary.encode_line(
                    src_stringified, add_if_not_exist=False, append_eos=True
                ).long(),
                "target": self.dictionary.encode_line(
                    tgt_stringified, add_if_not_exist=False, append_eos=True
                ).long(),
            }
            fairseq_item["prev_output_tokens"] = fairseq_item["target"].roll(1)
            fairseq_item["loss_keep_mask"] = torch.ones_like(fairseq_item["target"])
            assert self.dictionary.unk() not in fairseq_item["src_tokens"]
            assert self.dictionary.unk() not in fairseq_item["target"]
            return fairseq_item

        # XXX: consider adding equivalence classes to input/output tokens (ie '0' transforms into one of "abcd" and '1' into one of "efgh")
        #      to test hypothesis of restricted vocabulary being bad for transformers (ie makes inputs more distinuishable from each other)
        source = self.dictionary.encode_line(
            src_stringified, add_if_not_exist=False, append_eos=False
        ).long()
        separator = torch.tensor([self.dictionary.bos()]).long()
        target = self.dictionary.encode_line(
            tgt_stringified, add_if_not_exist=False, append_eos=False
        ).long()
        eos = torch.tensor([self.dictionary.eos()]).long()

        fairseq_item = {
            "target": torch.cat([source, separator, target, eos]),
            "id": index,
        }
        fairseq_item["src_tokens"] = fairseq_item["target"].roll(1)
        if self.cfg.allow_loss_on_problem_statement:
            fairseq_item["loss_keep_mask"] = torch.cat(
                [
                    torch.zeros_like(source),
                    torch.zeros_like(separator),
                    torch.ones_like(target),
                    torch.ones_like(eos),
                ]
            )
        else:
            fairseq_item["loss_keep_mask"] = torch.ones_like(fairseq_item["target"])
        return fairseq_item

    def __len__(self):
        return self._length

    def size(self, index):
        return self._sizes[index]

    def _prepare_epoch(self, epoch: int):
        num_batches_in_epoch = self.cfg.epoch_size // self.batch_size
        with numpy_seed(hash((self.seed, epoch)) % (2**32 - 1)):
            sizes = np.repeat(
                np.random.randint(
                    1, self.length_range, size=[num_batches_in_epoch + 1]
                ),
                repeats=self.batch_size,
            )
        self._sizes = sizes[: self.cfg.epoch_size]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self._sizes[indices]

    def set_epoch(self, epoch: int):
        """Will receive the updated epoch number at the beginning of the epoch."""
        self.epoch = epoch
        self._prepare_epoch(epoch)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        all_keys = list(samples[0].keys())
        collated = {}
        if "src_tokens" in all_keys:
            collated["src_tokens"] = collate_tokens(
                [s["src_tokens"] for s in samples], pad_idx=self.dictionary.pad()
            )
        collated["id"] = torch.LongTensor([s["id"] for s in samples])
        collated["target"] = collate_tokens(
            [s["target"] for s in samples], pad_idx=self.dictionary.pad()
        )
        if "prev_output_tokens" in all_keys:
            collated["prev_output_tokens"] = collate_tokens(
                [s["prev_output_tokens"] for s in samples],
                pad_idx=self.dictionary.pad(),
            )
        collated["loss_keep_mask"] = collate_tokens(
            [s["loss_keep_mask"] for s in samples], pad_idx=0
        )
        net_input_keys = ("src_tokens", "prev_output_tokens")
        res = {key: collated[key] for key in all_keys if key not in net_input_keys}
        res["net_input"] = {
            key: collated[key] for key in net_input_keys if key in all_keys
        }
        ntokens_key = "src_tokens" if "src_tokens" in all_keys else "prev_output_tokens"
        res["ntokens"] = sum(len(s[ntokens_key]) for s in samples)
        res["nsentences"] = len(samples)
        return res


@register_task("chomsky_hierarchy", dataclass=ChomskyHierarchyTaskConfig)
class ChomskyHierarchyTask(FairseqTask):
    def __init__(self, cfg, dictionary):
        super().__init__(None)  # type: ignore
        self._source_dictionary = dictionary
        self._target_dictionary = dictionary
        self.cfg = cfg

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        return self._source_dictionary

    @property
    def dictionary(self):
        return self._source_dictionary

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        # make source dictionary
        numbers = [str(i) for i in range(10)]
        letters = list("abcdefghijklmnopqrstuvwxyz")
        symbols = numbers + letters
        assert 2 <= cfg.vocab_size < len(symbols)
        dictionary = Dictionary()
        # for symbol in symbols[:cfg.vocab_size]:
        for symbol in symbols:
            dictionary.add_symbol(symbol)
        return cls(cfg, dictionary, **kwargs)

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> ChomskyIndexedDataset:
        data_dir = Path(self.cfg.data)
        assert data_dir.is_dir()
        src_fname = f"{split}.{self.cfg.task_name}.src"
        tgt_fname = f"{split}.{self.cfg.task_name}.tgt"
        self.datasets[split] = ChomskyIndexedDataset(
            self.cfg,
            self.dictionary,
            src_path=data_dir / src_fname,
            tgt_path=data_dir / tgt_fname,
        )
        return self.datasets[split]

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
