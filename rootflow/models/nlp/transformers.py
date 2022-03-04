from typing import List, Mapping, Union, Dict
import os
import torch
from torch.nn import Module, ModuleDict, ModuleList
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer
from rootflow.models.auto.model import RootflowAutoModel
from rootflow.models.nlp.utils import get_sequence_bookends_recursive, listify_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "True"


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
    ) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.transformer.config
        assert isinstance(
            head, (Module, ModuleList, ModuleDict, None)
        ), "Head must be a torch Module or equivalent!"
        self.head = head

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

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        print(batch)
        transformer_outputs = self.transformer(batch)
        if isinstance(self.head, ModuleDict):
            return torch.sum(
                [
                    task_head.training_step(transformer_outputs)
                    for task_head in self.head.values()
                ]
            )
        elif isinstance(self.head, ModuleList):
            return torch.sum(
                [
                    task_head.training_step(transformer_outputs)
                    for task_head in self.head
                ]
            )
        elif isinstance(self.head, Module):
            return self.head.training_step(transformer_outputs)


# Copied from the huggingface RobertaClassificationHead
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
        print(batch)
        self.forward(batch)
        return 1.0


class AutoTransformer(RootflowAutoModel):
    # TODO Consider refactoring this into a __new__ function instead, so that
    # we can return a normal Transformer. (Just adding a function to transformer
    # also seems like a good option)
    def __init__(self, model_name_or_path: str, tasks: List[dict]) -> None:
        super().__init__(tasks=tasks)
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.transformer.config
        self.head = ModuleDict(
            {task["name"]: self.construct_head_from_task(task) for task in tasks}
        )

    def forward(self, *args, **kwargs):
        transformer_outputs = self.transformer(*args, **kwargs)
        return {
            task_name: task_head(transformer_outputs)
            for task_name, task_head in self.head.items()
        }

    def construct_head_from_task(self, task: dict) -> Module:
        task_type, task_shape = task["type"], task["shape"]
        if task_type is "classification":
            return ClassificationHead(
                self.config.hidden_size,
                task_shape,
                dropout=self.config.hidden_dropout_prob,
            )
        elif task_type is "binary" and task_shape == 2:
            return ClassificationHead(
                self.config.hidden_size,
                task_shape,
                dropout=self.config.hidden_dropout_prob,
            )
        elif task_type is "binary":
            raise NotImplementedError("Cannot make a multitarget binary head")
        elif task_type is "regression":
            raise NotImplementedError("Cannot make a regression head")
        else:
            raise ValueError(f"{task_type} is not a recognized task type")
