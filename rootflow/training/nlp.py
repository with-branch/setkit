from typing import List, Callable

from rootflow.datasets.base import RootflowDataset
from rootflow.tasks.base import RootflowTrainer


class ClassificationTrainer(RootflowTrainer):
    def __init__(
        self,
        model,
        training_dataset: RootflowDataset,
        validataion_dataset: RootflowDataset,
        metrics: List[Callable] = None,
    ) -> None:
        super().__init__(model, training_dataset, validataion_dataset, metrics)

    def train() -> None:
        raise NotImplementedError


class FineTuneTrainer(RootflowTrainer):
    def __init__(
        self,
        model,
        training_dataset: RootflowDataset,
        validataion_dataset: RootflowDataset,
        metrics: List[Callable] = None,
    ) -> None:
        super().__init__(model, training_dataset, validataion_dataset, metrics)

    def train() -> None:
        raise NotImplementedError
