"""
This file is heavily based on a file from fairseq, fairseq/criterions/cross_entropy.py
which has the following license:

    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
"""

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig


@dataclass
class CrossEntropyWithPonderCriterionConfig(CrossEntropyCriterionConfig):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy_with_ponder", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyWithPonderCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        """
        XXX TODO: compute exact sequence accuracy
        XXX TODO: apply loss_keep_mask
        """
        target = sample["target"]
        loss_keep_mask = sample["loss_keep_mask"].bool()
        ntargets_per_seq = loss_keep_mask.sum(dim=-1)
        assert len(ntargets_per_seq.shape) == 1
        targets_per_seq = target.masked_select(loss_keep_mask).split(ntargets_per_seq.tolist())
        pred = net_output[0].argmax(dim=-1).masked_select(loss_keep_mask).split(ntargets_per_seq.tolist())
        pred_per_seq = net_output[0].argmax(dim=-1).masked_select(loss_keep_mask).split(ntargets_per_seq.tolist())
        ncorrect_force_dec = sum([pred.eq(tgt).all().item() for (pred, tgt) in zip(pred_per_seq, targets_per_seq)])

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "ncorrect": ncorrect_force_dec,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "acc", ncorrect / nsentences, nsentences, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
