from typing import Callable, List, Union

from torch.nn import Module
from torch.optim import Optimizer, AdamW
from torch.utils.data import Dataset, DataLoader

from rootflow.datasets.base import RootflowDataset


class RootflowTrainer:
    def __init__(
        self,
        results_directory: str,
        model: Module,
        training_dataset: Union[Dataset, RootflowDataset],
        validation_dataset: Union[Dataset, RootflowDataset],
        metrics: List[Callable] = None,
        learning_rate: float = None,
        epochs: int = None,
        checkpoint: int = 500,
        early_stopping: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.results_directory = results_directory
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.metrics = metrics
        self.config = kwargs

    def train(self) -> None:
        raise NotImplementedError
