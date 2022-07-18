# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def get_pooling(method: str) -> Callable[[Tensor, Tensor], Tensor]:
    methods = {
        "cls": cls_pooling,
        "mean": mean_pooling,
        "max": max_pooling,
    }

    pooling = methods.get(method)
    if pooling is None:
        raise ValueError(f"Pooling `method` should be one of {list(methods)}")

    return pooling


def mean_pooling(input_: Tensor, mask: Tensor) -> Tensor:
    mask = mask.unsqueeze(dim=-1).float()
    masked_sum = torch.sum(input_ * mask, dim=1)
    nonzeros = torch.clamp(mask.sum(dim=1), min=1e-9)

    return masked_sum / nonzeros


def max_pooling(input_: Tensor, mask: Tensor) -> Tensor:
    mask = mask.unsqueeze(dim=-1).expand(input_.size()).float()
    input_[mask == 0] = -1e9

    return torch.max(input_, dim=1)[0]


def cummax_pooling(input_: Tensor, mask: Tensor) -> Tensor:
    return max_pooling(input_.cummax(dim=1), mask)


def cls_pooling(input_: Tensor, mask: Tensor) -> Tensor:
    return input_[:, 0]
