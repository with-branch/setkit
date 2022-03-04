from typing import List, Callable

from rootflow.training.base import RootflowTrainer


class ClassificationTrainer(RootflowTrainer):
    def train(self) -> None:
        raise NotImplementedError


class FineTuneTrainer(RootflowTrainer):
    def train(self) -> None:
        raise NotImplementedError
