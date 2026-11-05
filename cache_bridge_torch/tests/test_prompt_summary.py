# -*- coding: utf-8 -*-
"""Regression tests for the train/inference prompt-summary contract."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.model import masked_prompt_summary


def test_answer_tokens_do_not_change_summary():
    # Given: two sequences that differ only in their answer-span embeddings.
    prompt = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [50.0, 60.0]]])
    changed_answer = prompt.clone()
    changed_answer[0, 2] = -500.0
    prompt_mask = torch.tensor([[1, 1, 0]])

    # When: the summary is computed with the training prompt mask.
    original_summary = masked_prompt_summary(prompt, prompt_mask)
    changed_summary = masked_prompt_summary(changed_answer, prompt_mask)

    # Then: answer information cannot affect the gate summary.
    assert torch.equal(original_summary, changed_summary)
    assert torch.equal(original_summary, torch.tensor([[[2.0, 3.0]]]))


def test_left_padding_matches_unpadded_summary():
    # Given: the same prompt with and without two left-padding positions.
    prompt = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    padded = torch.tensor([[[99.0, 99.0], [99.0, 99.0],
                            [1.0, 2.0], [3.0, 4.0]]])
    padded_mask = torch.tensor([[0, 0, 1, 1]])

    # When: inference masks left padding and training sees the plain prompt.
    padded_summary = masked_prompt_summary(padded, padded_mask)
    plain_summary = masked_prompt_summary(
        prompt, torch.ones(1, 2, dtype=torch.long))

    # Then: the two paths return exactly the same vector.
    assert torch.equal(padded_summary, plain_summary)


if __name__ == "__main__":
    test_answer_tokens_do_not_change_summary()
    test_left_padding_matches_unpadded_summary()
    print("2 / 2 prompt-summary regression tests passed")
