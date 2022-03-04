from typing import Callable, List, Union

from torch.nn import Module
from torch.optim import Optimizer, AdamW

from rootflow.datasets.base import RootflowDataset


class RootflowTrainer:
    def __init__(
        self,
        results_directory: str,
        model: Module,
        training_dataset: RootflowDataset,
        validataion_dataset: RootflowDataset,
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
        self.validation_dataset = validataion_dataset
        self.metrics = metrics
        self.config = kwargs

    def train(self) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            self.model.parameters(),
        )
