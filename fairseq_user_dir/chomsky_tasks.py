from dataclasses import dataclass, field

import haiku as hk
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, LegacyFairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
from icecream import ic
from neural_networks_chomsky_hierarchy.training import constants
from omegaconf import DictConfig

# seed = 1
# rng_seq = hk.PRNGSequence(seed)

# all_task_names = ['modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation', 'modular_arithmetic_brackets', 'reverse_string', 'missing_duplicate_string', 'duplicate_string', 'binary_addition', 'binary_multiplication', 'compute_sqrt', 'odds_first', 'solve_equation', 'stack_manipulation', 'bucket_sort']
# task_name = "even_pairs"

# for task_name in all_task_names[1:]:
#     kwargs = {}
#     if  "solve_equation" == task_name:
#         kwargs["modulus"] = 3
#     if "modular" in task_name:
#         kwargs["modulus"] = 3
#     if task_name in ("reverse_string", "duplicate_string", "odds_first"):
#         kwargs["vocab_size"] = 3

#     ic(task_name)
#     task = constants.TASK_BUILDERS[task_name](**kwargs)

#     ic(task.sample_batch(next(rng_seq), length=5, batch_size=1))

# ic(constants.TASK_BUILDERS.keys())

CHOMSKY_SUBTASK_CHOICES = ChoiceEnum(
    [
        "modular_arithmetic",
        "parity_check",
        "even_pairs",
        "cycle_navigation",
        "modular_arithmetic_brackets",
        "reverse_string",
        "missing_duplicate_string",
        "duplicate_string",
        "binary_addition",
        "binary_multiplication",
        "compute_sqrt",
        "odds_first",
        "solve_equation",
        "stack_manipulation",
        "bucket_sort",
    ]
)


class ChomskyDataset(FairseqDataset):
    def __init__(
        self, task_name: str, seed: int = 1, modulus: int = 5, vocab_size: int = 2
    ):
        super().__init__()
        self.task_name = task_name
        self.seed = seed
        self.epoch = 0
        task_kwargs = {}
        if "solve_equation" == task_name:
            task_kwargs["modulus"] = modulus
        if "modular" in task_name:
            task_kwargs["modulus"] = modulus
        if task_name in ("reverse_string", "duplicate_string", "odds_first"):
            task_kwargs["vocab_size"] = vocab_size
        self._jax_chomsky_task = constants.TASK_BUILDERS[task_name](**task_kwargs)

    def __getitem__(self, index):
        rng_seq = hk.PRNGSequence(hash((self.seed, self.epoch, index)))
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        self.epoch = epoch


@dataclass
class ChomskyHierarchyTaskConfig(FairseqDataclass):
    subtask: CHOMSKY_SUBTASK_CHOICES = field(  # type: ignore
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


@register_task("chomsky_hierarchy", dataclass=ChomskyHierarchyTaskConfig)
class ChomskyHierarchyTask(FairseqTask):
    def __init__(self, cfg, dictionary):
        super().__init__(None)  # type: ignore
        self._source_dictionary = dictionary
        self._target_dictionary = dictionary

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        return self._target_dictionary

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        # make source dictionary
        return cls(cfg, **kwargs)

    # def get_batch_iterator(
    #     self,
    #     dataset,
    #     max_tokens=None,
    #     max_sentences=None,
    #     max_positions=None,
    #     ignore_invalid_inputs=False,
    #     required_batch_size_multiple=1,
    #     seed=1,
    #     num_shards=1,
    #     shard_id=0,
    #     num_workers=0,
    #     epoch=1,
    #     data_buffer_size=0,
    #     disable_iterator_cache=False,
    #     skip_remainder_batch=False,
    #     grouped_shuffling=False,
    #     update_epoch_batch_itr=False,
    # ):
    #     """
    #     Get an iterator that yields batches of data from the given dataset.

    #     Args:
    #         dataset (~fairseq.data.FairseqDataset): dataset to batch
    #         max_tokens (int, optional): max number of tokens in each batch
    #             (default: None).
    #         max_sentences (int, optional): max number of sentences in each
    #             batch (default: None).
    #         max_positions (optional): max sentence length supported by the
    #             model (default: None).
    #         ignore_invalid_inputs (bool, optional): don't raise Exception for
    #             sentences that are too long (default: False).
    #         required_batch_size_multiple (int, optional): require batch size to
    #             be a multiple of N (default: 1).
    #         seed (int, optional): seed for random number generator for
    #             reproducibility (default: 1).
    #         num_shards (int, optional): shard the data iterator into N
    #             shards (default: 1).
    #         shard_id (int, optional): which shard of the data iterator to
    #             return (default: 0).
    #         num_workers (int, optional): how many subprocesses to use for data
    #             loading. 0 means the data will be loaded in the main process
    #             (default: 0).
    #         epoch (int, optional): the epoch to start the iterator from
    #             (default: 1).
    #         data_buffer_size (int, optional): number of batches to
    #             preload (default: 0).
    #         disable_iterator_cache (bool, optional): don't cache the
    #             EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
    #             (default: False).
    #         skip_remainder_batch (bool, optional): if set, discard the last
    #             batch in each training epoch, as the last batch is often smaller than
    #                 local_batch_size * distributed_word_size (default: ``True``).
    #         grouped_shuffling (bool, optional): group batches with each groups
    #             containing num_shards batches and shuffle groups. Reduces difference
    #             between sequence lengths among workers for batches sorted by length.
    #         update_epoch_batch_itr (bool optional): if true then donot use the cached
    #             batch iterator for the epoch

    #     Returns:
    #         ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
    #             given dataset split
    #     """

    # @classmethod
    # def setup_dictionary(cls, args, **kwargs):

    # @classmethod
    # def setup_task(cls, args, **kwargs):

    # def load_dataset(
    #     self, split: str, epoch=1, combine=False, **kwargs
    # ) -> MonolingualDataset:


#   @abc.abstractmethod
#   def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
#                    length: int) -> Batch:
#     """Returns a batch of inputs/outputs."""

#   def pointwise_loss_fn(self, output: chex.Array,
#                         target: chex.Array) -> chex.Array:
#     """Returns the pointwise loss between an output and a target."""
#     return -target * jnn.log_softmax(output)

#   def accuracy_fn(self, output: chex.Array, target: chex.Array) -> chex.Array:
#     """Returns the accuracy between an output and a target."""
#     return (jnp.argmax(output,
#                        axis=-1) == jnp.argmax(target,
#                                               axis=-1)).astype(jnp.float32)

#   def accuracy_mask(self, target: chex.Array) -> chex.Array:
#     """Returns a mask to compute the accuracies, to remove the superfluous ones."""
#     # Target is a shape of shape (B, T, C) where C is the number of classes.
#     # We want a mask per input (B, T), so we take this shape.
#     return jnp.ones(target.shape[:-1])

#   @property
#   @abc.abstractmethod
#   def input_size(self) -> int:
#     """Returns the size of the input of the models trained on this task."""

#   @property
#   @abc.abstractmethod
#   def output_size(self) -> int:
#     """Returns the size of the output of the models trained on this task."""

#   def output_length(self, input_length: int) -> int:
#     """Returns the length of the output, given an input length."""
#     del input_length
#     return 1
