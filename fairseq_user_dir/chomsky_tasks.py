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
    seed: int = II("common.seed")
    batch_size: int = II("dataset.batch_size")
    epoch_size: int = field(
        default=10_000,
        metadata={"help": ("Total number of sequences in training epoch")},
    )
    allow_loss_on_problem_statement: bool = field(
        default=False,
        metadata={
            "help": (
                "Only include problem solution in cross-entropy loss (discard loss on input part)"
            )
        },
    )
    ponder_prob: float = field(
        default=0,
        metadata={"help": ("Probability to insert a ponder token")},
    )
    ponder_count: int = field(
        default=2,
        metadata={"help": ("Maximum amount of sequential ponder tokensl (bernoulli trial)")},
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
        self.ponder_idx = self.dictionary.unk()
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

        if self.cfg.ponder_prob <= 0:
            return output

        seq_len = len(output["target"])
        with data_utils.numpy_seed(self.cfg.seed, index):
            ponder_counts = np.random.binomial(self.cfg.ponder_count, self.cfg.ponder_prob, size=seq_len)
            # no use ponder after last token
            ponder_counts[-1] = 0

            # we interleave the original sequence with a ponder token after each original token
            target_with_ponder_tokens = np.stack([output["target"].numpy(), np.repeat(self.ponder_idx, seq_len)]).transpose().reshape(-1)
            # we interleave 1's (since we do not want to repeat or omit tokens from original sequence) 
            # and number of repeats of the ponder token that follows it (0 if no ponder token follows)
            all_token_counts = np.stack([np.repeat(1, seq_len), ponder_counts]).transpose().reshape(-1)
            new_output_seq = np.repeat(target_with_ponder_tokens, all_token_counts)

            # same as above, but for loss mask
            # XXX NOTE: we are hardcoding ponder to not participate in CE loss
            new_loss_mask = np.repeat(
                np.stack([output["loss_keep_mask"].numpy(), np.repeat(0, seq_len)]).transpose().reshape(-1),
                all_token_counts,
            )

        output["target"] = torch.from_numpy(new_output_seq)
        output["src_tokens"] = output["target"].roll(1)
        output["new_loss_mask"] = torch.from_numpy(new_loss_mask)
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
        dictionary = Dictionary()
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
