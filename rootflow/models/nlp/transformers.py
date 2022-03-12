import gc
from typing import List, Mapping, Union, Dict
import os
import torch
from torch.nn import Module, ModuleDict, ModuleList
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer
from rootflow.models.auto.model import RootflowAutoModel
from rootflow.models.nlp.utils import get_sequence_bookends_recursive, listify_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "True"

from transformers import logging

logging.set_verbosity_error()


class Tokenizer:
    def __init__(
        self, model_name_or_path: str, max_token_length: int, mode: str = "split"
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer = lambda tokenizer_input: tokenizer(
            tokenizer_input, padding="max_length", truncation=False
        )
        if mode == "split":
            # Should bring out the tokenizer function, and just have bookends operate
            # on the output tokens
            self._tokenize_function = (
                lambda tokenization_input: get_sequence_bookends_recursive(
                    self.tokenizer(tokenization_input),
                    max_token_length,
                )
            )
        # (Reminders to implement these modes)
        elif mode == "first":
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
        elif mode == "last":
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")

    def __call__(self, input_strings: Union[List[str], str]) -> Dict[str, torch.Tensor]:
        tokenized = self._tokenize_function(input_strings)
        return listify_tokens(tokenized)


class Transformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        head: Union[Module, ModuleList, ModuleDict] = None,
        num_training_steps: int = 100000,
    ) -> None:
        super().__init__()
        self.num_training_steps = num_training_steps
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.transformer.config
        assert isinstance(
            head, (Module, ModuleList, ModuleDict, None)
        ), "Head must be a torch Module or equivalent!"
        self.head = head
        self.learning_rate = 1e-3

    # Flags instead of type checking would make this faster
    def forward(self, *args, **kwargs):
        transformer_outputs = self.transformer(*args, **kwargs)
        if isinstance(self.head, ModuleDict):
            return {
                task_name: task_head(transformer_outputs)
                for task_name, task_head in self.head.items()
            }
        elif isinstance(self.head, ModuleList):
            return [task_head(transformer_outputs) for task_head in self.head]
        elif isinstance(self.head, Module):
            return self.head(transformer_outputs)
        else:
            return transformer_outputs

    # TODO probably breaks when the number of tasks is greater than one,
    # since torch.sum does not sum over lists
    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        print("training_step")
        print(batch["data"]["input_ids"].shape)
        batch["data"] = self.transformer(**batch["data"])
        print("Made it past the transformer")
        if isinstance(self.head, ModuleDict):
            return torch.sum(
                [
                    task_head.training_step(
                        {
                            "id": batch["id"],
                            "data": batch["data"],
                            "target": batch["target"][task_name],
                        }
                    )
                    for task_name, task_head in self.head.items()
                ]
            )
        elif isinstance(self.head, ModuleList):
            return torch.sum(
                [
                    task_head.training_step(
                        {
                            "id": batch["id"],
                            "data": batch["data"],
                            "target": batch["target"][task_idx],
                        }
                    )
                    for task_idx, task_head in enumerate(self.head)
                ]
            )
        elif isinstance(self.head, Module):
            return self.head.training_step(batch)
        gc.collect()

    # TODO probably breaks when the number of tasks is greater than one,
    # since torch.sum does not sum over lists
    def validation_step(self, batch, *args, **kwargs) -> torch.Tensor:
        print("validation_step")
        print(batch["data"]["input_ids"].shape)
        with torch.no_grad():
            batch["data"] = self.transformer(**batch["data"])
            if isinstance(self.head, ModuleDict):
                return torch.sum(
                    [
                        task_head.training_step(
                            {
                                "id": batch["id"],
                                "data": batch["data"],
                                "target": batch["target"][task_name],
                            }
                        )
                        for task_name, task_head in self.head.items()
                    ]
                )
            elif isinstance(self.head, ModuleList):
                return torch.sum(
                    [
                        task_head.training_step(
                            {
                                "id": batch["id"],
                                "data": batch["data"],
                                "target": batch["target"][task_idx],
                            }
                        )
                        for task_idx, task_head in enumerate(self.head)
                    ]
                )
            elif isinstance(self.head, Module):
                return self.head.training_step(batch)
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, betas=(0.95, 0.999)
        )
        learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": learning_rate_scheduler, "interval": "step"},
        }


class BinaryClassificationHead(LightningModule):
    def __init__(self, input_size, num_classes, dropout: float = 0.1):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(input_size, num_classes)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, features, **kwargs):
        x = features[0][:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        outputs = torch.squeeze(self.forward(batch["data"]), dim=1)
        target = batch["target"].float()
        return self.loss_fn(outputs, target)

    def validation_step(self, batch, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            outputs = torch.squeeze(self.forward(batch["data"]), dim=1)
            target = batch["target"].float()
            return self.loss_fn(outputs, target)


class ClassificationHead(LightningModule):
    def __init__(self, input_size, num_classes, dropout: float = 0.1):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = num_classes
        self.out_proj = torch.nn.Linear(input_size, num_classes)

    def forward(self, features, **kwargs):
        # Needs to first load the correct portion of the
        # model output from the output dictionary (use CLS)
        x = features[0][:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        outputs = self.forward(batch["data"])
        target = batch["target"]
        print(target)
        return 1.0


class AutoTransformer(RootflowAutoModel):
    # TODO Consider refactoring this into a __new__ function instead, so that
    # we can return a normal Transformer. (Just adding a function to transformer
    # also seems like a good option)
    def __new__(
        cls: "AutoTransformer",
        model_name_or_path: str,
        tasks: List[dict],
        num_training_steps: int,
    ) -> Transformer:
        super().__new__(cls, tasks=tasks)
        temp_transformer = AutoModel.from_pretrained(model_name_or_path)
        config = temp_transformer.config
        del temp_transformer
        if len(tasks) == 1:
            head = AutoTransformer.construct_head_from_task(
                task=tasks[0], config=config
            )
        else:
            head = ModuleDict(
                {
                    task["name"]: AutoTransformer.construct_head_from_task(
                        task=task, config=config
                    )
                    for task in tasks
                }
            )
        return Transformer(
            model_name_or_path=model_name_or_path,
            head=head,
            num_training_steps=num_training_steps,
        )

    def construct_head_from_task(task: dict, config: dict) -> Module:
        task_type, task_shape = task["type"], task["shape"]
        if task_type is "classification":
            return ClassificationHead(
                config.hidden_size,
                task_shape,
                dropout=config.hidden_dropout_prob,
            )
        elif task_type is "binary":
            return BinaryClassificationHead(
                config.hidden_size,
                task_shape,
                dropout=config.hidden_dropout_prob,
            )
        elif task_type is "regression":
            raise NotImplementedError("Cannot make a regression head")
        else:
            raise ValueError(f"{task_type} is not a recognized task type")
