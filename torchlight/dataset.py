# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, TypeVar

from torchlight.utils import ModeKeys, PhaseMixin


class Collator(PhaseMixin, metaclass=abc.ABCMeta):
    T = TypeVar("T", bound="Collator")

    def __init__(self, mode: ModeKeys = ModeKeys.TRAIN) -> None:
        self.mode = mode

    def __call__(self, batch: Sequence) -> Any:
        return self.collate_fn(batch)

    def encode(self, example) -> Any:  # pylint: disable=no-self-use
        return example

    def encode_batch(self, batch: Sequence) -> list:
        return list(map(self.encode, batch))

    def collate_train(self, batch: Sequence) -> Any:
        raise NotImplementedError()

    def collate_eval(self, batch: Sequence) -> Any:
        return self.collate_train(batch)

    def collate_predict(self, batch: Sequence) -> Any:
        return self.collate_train(batch)

    def _get_collate_fn(self):
        return {
            ModeKeys.TRAIN: self.collate_train,
            ModeKeys.EVAL: self.collate_eval,
            ModeKeys.PREDICT: self.collate_predict,
        }[self.mode]

    def _collate(self, encoded_batch):
        return self._get_collate_fn()(encoded_batch)

    def collate_fn(self, batch: Sequence) -> Any:
        encoded_batch = self.encode_batch(batch)
        collated = self._collate(encoded_batch)

        return collated
