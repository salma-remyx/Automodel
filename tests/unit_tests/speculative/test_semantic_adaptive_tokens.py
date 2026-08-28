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

"""Tests for SDSAT semantic-adaptive-token mixing and its train_ft wiring."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.utils import default_collater
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.speculative.semantic_adaptive_tokens import (
    SemanticAdaptiveTokenConfig,
    SemanticAdaptiveTokenMixer,
)

SAT_ID = 90
IGNORE = -100


def make_mixer(k=2, mask_standard_positions=False):
    return SemanticAdaptiveTokenMixer(
        num_adaptive_tokens=k,
        adaptive_token_id=SAT_ID,
        mask_standard_positions=mask_standard_positions,
        ignore_index=IGNORE,
    )


def reference_mixed_labels(labels, k, ignore_index=IGNORE):
    """Per-position reference for the SDSAT label rule (independent of the mixer implementation).

    Block ``i`` of the mixed sequence holds the standard token ``t_i`` followed by ``k`` SAT
    slots; slot ``m`` of block ``i`` predicts the token ``m + 1`` steps after ``t_i``, i.e. the
    pre-shifted target ``labels[i + m]``.
    """
    batch, seq = labels.shape
    out = labels.new_full((batch, seq * (k + 1)), ignore_index)
    for b in range(batch):
        for i in range(seq):
            for m in range(k + 1):
                if i + m < seq:
                    out[b, i * (k + 1) + m] = labels[b, i + m]
    return out


def test_config_build_requires_adaptive_token_id():
    with pytest.raises(ValueError, match="adaptive_token_id"):
        SemanticAdaptiveTokenConfig(num_adaptive_tokens=2).build()


def test_config_build_roundtrip():
    mixer = SemanticAdaptiveTokenConfig(num_adaptive_tokens=3, adaptive_token_id=SAT_ID).build()
    assert mixer.num_adaptive_tokens == 3
    assert mixer.adaptive_token_id == SAT_ID
    assert mixer.mask_standard_positions is False


def test_mix_labels_matches_reference_window_rule():
    torch.manual_seed(0)
    labels = torch.randint(0, 50, (3, 7))
    labels[:, :2] = IGNORE  # prompt masking must be preserved
    for k in (1, 2, 4):
        mixed = make_mixer(k=k).mix_labels(labels)
        assert mixed.shape == (3, 7 * (k + 1))
        torch.testing.assert_close(mixed, reference_mixed_labels(labels, k))


def test_mix_labels_mask_standard_positions():
    labels = torch.arange(6).unsqueeze(0)
    mixed = make_mixer(k=2, mask_standard_positions=True).mix_labels(labels)
    mixed = mixed.view(1, 6, 3)
    # Standard next-token targets (slot 0 of each block) are masked out (stage-1 training),
    # while the SAT slots keep their (m + 1)-steps-ahead targets.
    assert (mixed[0, :, 0] == IGNORE).all()
    torch.testing.assert_close(mixed[0, :, 1], torch.tensor([1, 2, 3, 4, 5, IGNORE]))
    torch.testing.assert_close(mixed[0, :, 2], torch.tensor([2, 3, 4, 5, IGNORE, IGNORE]))


def test_call_mixes_default_collater_batch():
    """Batch built by the repo's default_collater is expanded block-wise."""
    examples = [
        {
            "input_ids": [10, 11, 12, 13],
            "labels": [IGNORE, 12, 13, 14],
            "attention_mask": [1, 1, 1, 1],
        },
        {
            "input_ids": [20, 21],
            "labels": [21, 22],
            "attention_mask": [1, 1],
        },
    ]
    batch = default_collater(examples)
    k = 2
    mixed = make_mixer(k=k)(batch)

    assert mixed["input_ids"].shape == (2, 4 * (k + 1))
    assert mixed["labels"].shape == (2, 4 * (k + 1))
    assert mixed["attention_mask"].shape == (2, 4 * (k + 1))
    assert mixed["padding_mask"].shape == (2, 4 * (k + 1))

    # First row: blocks interleave [t_i, SAT, SAT]; labels follow the SDSAT window rule.
    row_ids = mixed["input_ids"][0].view(4, k + 1)
    torch.testing.assert_close(row_ids[:, 0], torch.tensor([10, 11, 12, 13]))
    assert (row_ids[:, 1:] == SAT_ID).all()
    torch.testing.assert_close(mixed["labels"][0], reference_mixed_labels(torch.tensor([[IGNORE, 12, 13, 14]]), k)[0])

    # Second row was padded by the collater to length 4; its attention mask repeats per block.
    torch.testing.assert_close(mixed["attention_mask"][1], torch.tensor([1, 1, 1, 1, 1, 1] + [0] * 6))
    # The input batch must not be mutated.
    assert batch["input_ids"].shape == (2, 4)


def test_call_recomputes_position_ids():
    batch = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "labels": torch.tensor([[6, 7, 8]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
    }
    mixed = make_mixer(k=1)(batch)
    torch.testing.assert_close(mixed["position_ids"], torch.arange(6).unsqueeze(0))


def test_call_rejects_unknown_sequence_aligned_field():
    batch = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "labels": torch.tensor([[6, 7, 8]]),
        "loss_mask": torch.tensor([[0, 1, 1]]),
    }
    with pytest.raises(NotImplementedError, match="loss_mask"):
        make_mixer(k=1)(batch)


def test_mixed_labels_align_with_masked_cross_entropy():
    """The mixed batch scored by the recipe's MaskedCrossEntropy matches the SDSAT per-position targets.

    This is the contract the ``train_ft`` wiring relies on: the recipe's existing loss computes
    CE(logits[p], labels[p]) with no shift, so mixing labels is sufficient to implement the
    paper's joint masked loss.
    """
    k = 2
    vocab = 30
    seq = 5
    torch.manual_seed(0)
    labels = torch.randint(0, vocab, (1, seq))
    labels[0, 0] = IGNORE
    mixer = make_mixer(k=k)
    mixed_labels = mixer.mix_labels(labels)

    logits = torch.randn(1, seq * (k + 1), vocab)
    loss_fn = MaskedCrossEntropy(fp32_upcast=True, reduction="sum")
    observed = loss_fn(logits=logits, labels=mixed_labels)

    # Reference: sum of per-position CE over exactly the non-ignored mixed positions.
    flat_ce = F.cross_entropy(logits.view(-1, vocab).float(), mixed_labels.view(-1), reduction="none")
    expected = flat_ce[mixed_labels.view(-1) != IGNORE].sum()
    torch.testing.assert_close(observed, expected)

    # Sanity: the observed loss differs from the unmixed next-token loss (targets really moved ahead).
    unmixed = loss_fn(logits=logits[:, :seq], labels=labels)
    assert not torch.isclose(observed, unmixed)


def test_train_ft_recipe_wires_sdsat_mixer():
    """The SFT recipe exposes the sdsat wiring surface used by the ``sdsat:`` config block."""
    from nemo_automodel.recipes.llm import train_ft

    assert train_ft.SemanticAdaptiveTokenMixer is SemanticAdaptiveTokenMixer
    assert hasattr(train_ft.TrainFinetuneRecipeForNextTokenPrediction, "_countable_labels")


def test_recipe_config_exposes_typed_sdsat_accessor():
    """RecipeConfig.sdsat parses the YAML block into the typed config (None when absent)."""
    from nemo_automodel.recipes._typed_config import RecipeConfig

    class _Raw:
        def __init__(self, d):
            self._d = d

        def get(self, key, default=None):
            return self._d.get(key, default)

    cfg = RecipeConfig(_Raw({"sdsat": {"num_adaptive_tokens": 3, "adaptive_token_id": SAT_ID}}))
    sdsat = cfg.sdsat
    assert isinstance(sdsat, SemanticAdaptiveTokenConfig)
    assert sdsat.num_adaptive_tokens == 3
    assert sdsat.adaptive_token_id == SAT_ID
    assert RecipeConfig(_Raw({})).sdsat is None
