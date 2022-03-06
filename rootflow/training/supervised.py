from typing import Callable, List, Union

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer

from rootflow.datasets.base import RootflowDataLoader
from rootflow.training import metrics


class SupervisedTrainer:
    def __init__(
        self,
        results_directory: str,
        model: torch.nn.Module,
        training_dataset: Union[Dataset, DataLoader],
        validation_dataset: Union[Dataset, DataLoader],
        metrics: Union[Callable, List[Callable]] = None,
    ) -> None:
        self.output_directory = results_directory
        self.model = model
        self.train_loader = self.get_loader(training_dataset)
        self.validation_loader = self.get_loader(validation_dataset)
        self.metrics = metrics

    def get_loader(self, dataset):
        return RootflowDataLoader(dataset)

    def train(self) -> None:
        trainer = Trainer(auto_lr_find=True)
        trainer.tune(self.model, self.train_loader, self.validation_loader)
        trainer.fit(self.model, self.train_loader, self.validation_loader)
