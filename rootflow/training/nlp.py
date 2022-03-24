from typing import List, Callable, Dict, Union, Any, Optional, Tuple

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from rootflow.datasets.base import RootflowDataset
from rootflow.training.base import RootflowTrainer
from rootflow.datasets.base.utils import default_collate_without_key

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


class _ClassificationTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
    ):
        # Calculate the class weights
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        data = inputs.get("data")
        outputs = torch.squeeze(model(**data))
        target = inputs.get("target").float()
        loss = self._loss_fn(target, outputs)
        return (loss, outputs) if return_outputs else loss


class ClassificationTrainer(RootflowTrainer):
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
        **kwargs
    ) -> None:
        super().__init__(
            results_directory,
            model,
            training_dataset,
            validation_dataset,
            metrics,
            learning_rate,
            epochs,
            checkpoint,
            early_stopping,
            **kwargs
        )
        training_args = TrainingArguments(results_directory)
        self.trainer = _ClassificationTrainer(
            self.model,
            args=training_args,
            data_collator=lambda batch: default_collate_without_key(batch, "id"),
            train_dataset=self.training_dataset,
            eval_dataset=self.validation_dataset,
        )

    def train(self) -> None:
        self.trainer.train()


class FineTuneTrainer(RootflowTrainer):
    def train(self) -> None:
        raise NotImplementedError
