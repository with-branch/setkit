from typing import Callable, List, Union
from pytorch_lightning import LightningModule

from rootflow.datasets.base import RootflowDataset


class RootflowTrainer(LightningModule):
    def __init__(
        self,
        model,
        training_dataset: RootflowDataset,
        validataion_dataset: RootflowDataset,
        metrics: List[Callable] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validataion_dataset
        self.metrics = metrics

    def train() -> None:
        raise NotImplementedError
