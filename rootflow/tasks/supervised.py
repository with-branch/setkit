from typing import Callable, List, Union

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

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
        self.train_set = training_dataset
        self.val_set = validation_dataset
        self.metrics = metrics

    def train() -> None:
        raise NotImplementedError
