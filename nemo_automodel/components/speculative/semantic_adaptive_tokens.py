# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic adaptive token (SAT) sequence mixing for self-speculative fine-tuning.

Adapted from SDSAT (arXiv:2403.18647, "Accelerating LLM Inference through
Speculative Decoding with Semantic Adaptive Tokens"). The paper fine-tunes the
target model on sequences where ``k`` semantic adaptive tokens are inserted
after every standard token; the SAT slots are trained to predict tokens
``2 .. k+1`` steps ahead while standard positions keep ordinary next-token
supervision, so one forward pass learns to self-draft ``k`` tokens without
degrading standard-token accuracy.

This module implements the training-side core of that mechanism as a pure
batch transformation: it rewrites ``input_ids``, ``labels``,
``attention_mask``, ``padding_mask``, and ``position_ids`` into the mixed
layout, after which the recipe's existing model forward and masked
cross-entropy loss implement the paper's joint masked loss unchanged. Two
paper components are intentionally substituted with target-native equivalents:

- The paper's newly-added special token embeddings are replaced by a
  configurable existing vocabulary id (``adaptive_token_id``), e.g. a reserved
  special token the tokenizer never emits in data.
- The paper's two-stage schedule (SAT-only, then joint) is replaced by the
  ``mask_standard_positions`` flag: set it for a SAT-only first run, then
  unset it for a joint run.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SemanticAdaptiveTokenConfig:
    """Typed config for semantic adaptive token (SAT) sequence mixing.

    Attributes:
        num_adaptive_tokens: Number of SAT slots inserted after every standard
            token. The ``m``-th slot in each block is trained to predict the
            token ``m + 1`` positions ahead of the block's standard token.
        adaptive_token_id: Vocabulary id recycled as the SAT input embedding.
            Must be an id the tokenizer never emits in training data (e.g. a
            reserved special token). Required; there is no safe default.
        mask_standard_positions: When True, standard-token positions contribute
            no loss (labels set to ``ignore_index``), reproducing the paper's
            stage-1 SAT-only training. When False, training is joint (stage 2).
        ignore_index: Label value excluded from the cross-entropy loss.
    """

    num_adaptive_tokens: int = 1
    adaptive_token_id: int | None = None
    mask_standard_positions: bool = False
    ignore_index: int = -100

    def build(self) -> "SemanticAdaptiveTokenMixer":
        """Construct the batch mixer from these declarative settings.

        Returns:
            A callable :class:`SemanticAdaptiveTokenMixer` that rewrites an
            SFT batch into the SAT-mixed layout.

        Raises:
            ValueError: If ``adaptive_token_id`` was not configured.
        """
        if self.adaptive_token_id is None:
            raise ValueError(
                "sdsat.adaptive_token_id must be set to a vocabulary id the tokenizer never emits in "
                "training data (e.g. a reserved special token); it is recycled as the SAT input embedding."
            )
        return SemanticAdaptiveTokenMixer(
            num_adaptive_tokens=self.num_adaptive_tokens,
            adaptive_token_id=self.adaptive_token_id,
            mask_standard_positions=self.mask_standard_positions,
            ignore_index=self.ignore_index,
        )


class SemanticAdaptiveTokenMixer:
    """Rewrites an SFT batch into the SAT-mixed layout used by SDSAT training.

    Given a batch whose ``labels`` follow the repo's pre-shifted convention
    (``labels[:, i]`` is the token predicted at position ``i``), the mixed
    batch has ``k + 1`` positions per original token: the standard token
    followed by ``k`` SAT slots. The label window for block ``i`` is
    ``labels[:, i : i + k + 1]``, so position ``m`` of each block predicts
    ``labels[:, i + m]`` -- the standard next-token target for ``m == 0`` and
    the ``(m + 1)``-steps-ahead target for the ``m``-th SAT slot. Prompt
    masking is preserved automatically because it is already encoded in the
    incoming labels.
    """

    def __init__(
        self,
        *,
        num_adaptive_tokens: int,
        adaptive_token_id: int,
        mask_standard_positions: bool = False,
        ignore_index: int = -100,
    ):
        """Initialize the mixer.

        Args:
            num_adaptive_tokens: SAT slots per standard token; must be >= 1.
            adaptive_token_id: Vocabulary id used as the SAT input embedding.
            mask_standard_positions: Mask standard-token positions out of the
                loss (stage-1 SAT-only training).
            ignore_index: Label value excluded from the cross-entropy loss.

        Raises:
            ValueError: If ``num_adaptive_tokens`` < 1 or ``adaptive_token_id``
                is negative.
        """
        if num_adaptive_tokens < 1:
            raise ValueError(f"num_adaptive_tokens must be >= 1, got {num_adaptive_tokens}")
        if adaptive_token_id < 0:
            raise ValueError(f"adaptive_token_id must be a non-negative vocabulary id, got {adaptive_token_id}")
        self.num_adaptive_tokens = num_adaptive_tokens
        self.adaptive_token_id = adaptive_token_id
        self.mask_standard_positions = mask_standard_positions
        self.ignore_index = ignore_index

    def mix_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Expand pre-shifted labels into the SAT-mixed label layout.

        Args:
            labels: Tensor of shape [batch, sequence] holding pre-shifted LM
                targets (``labels[:, i]`` is predicted at position ``i``), with
                prompt/padding positions already set to ``ignore_index``.

        Returns:
            Tensor of shape [batch, sequence * (num_adaptive_tokens + 1)] where
            block ``i`` (positions ``i * (k + 1) .. i * (k + 1) + k``) holds the
            sliding window ``labels[:, i : i + k + 1]``, padded with
            ``ignore_index`` past the sequence end. With
            ``mask_standard_positions`` the first slot of every block (the
            standard next-token target) is set to ``ignore_index``.
        """
        batch, seq = labels.shape
        k = self.num_adaptive_tokens
        pad = labels.new_full((batch, k), self.ignore_index)
        padded = torch.cat([labels, pad], dim=1)  # [batch, sequence + k]
        windows = padded.unfold(1, k + 1, 1)  # [batch, sequence, k + 1]
        if self.mask_standard_positions:
            windows = windows.clone()
            windows[:, :, 0] = self.ignore_index
        return windows.reshape(batch, seq * (k + 1))

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Rewrite an SFT batch dict into the SAT-mixed layout.

        ``input_ids`` gains ``k`` SAT slots (filled with ``adaptive_token_id``)
        after every standard token; ``labels`` is expanded by
        :meth:`mix_labels`; ``attention_mask`` and ``padding_mask`` are
        repeated per block; ``position_ids`` (when present) is recomputed from
        the mixed attention mask, matching the HuggingFace default for
        non-packed batches.

        Args:
            batch: Mapping of batch fields. ``input_ids`` and ``labels`` are
                required tensors of shape [batch, sequence]; ``attention_mask``,
                ``padding_mask``, and ``position_ids`` are optional tensors of
                the same shape. Other fields pass through unchanged.

        Returns:
            A new mapping with the sequence-aligned fields expanded to
            [batch, sequence * (num_adaptive_tokens + 1)] and all other fields
            unchanged. The input mapping is not mutated.

        Raises:
            KeyError: If ``input_ids`` or ``labels`` is missing.
            NotImplementedError: If an unrecognized tensor field is aligned
                with the sequence dimension (e.g. ``loss_mask``, ``cu_seqlens``,
                or THD packing fields), since mixing it correctly is
                format-specific.
        """
        if "input_ids" not in batch or "labels" not in batch:
            raise KeyError("sdsat mixing requires 'input_ids' and 'labels' in the batch")
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        batch_size, seq = input_ids.shape
        k = self.num_adaptive_tokens
        handled = {"input_ids", "labels", "attention_mask", "padding_mask", "position_ids"}
        for key, value in batch.items():
            if key in handled:
                continue
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] == seq:
                raise NotImplementedError(
                    f"sdsat mixing does not support the sequence-aligned batch field {key!r} "
                    f"(shape {tuple(value.shape)}); run without packing/THD dataloaders or extend the mixer"
                )

        sat = input_ids.new_full((batch_size, seq, k), self.adaptive_token_id)
        mixed = dict(batch)
        mixed["input_ids"] = torch.cat([input_ids.unsqueeze(-1), sat], dim=-1).view(batch_size, seq * (k + 1))
        mixed["labels"] = self.mix_labels(labels)
        if "attention_mask" in batch:
            mixed["attention_mask"] = batch["attention_mask"].repeat_interleave(k + 1, dim=1)
        if "padding_mask" in batch:
            mixed["padding_mask"] = batch["padding_mask"].repeat_interleave(k + 1, dim=1)
        if "position_ids" in batch:
            if "attention_mask" in batch:
                mask = mixed["attention_mask"]
                mixed["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0)
            else:
                mixed["position_ids"] = (
                    torch.arange(seq * (k + 1), device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                )
        return mixed
