from typing import List, Mapping, Union, Dict
import torch
from torch.nn import Module, ModuleDict, ModuleList
from transformers import AutoModel, AutoTokenizer
from rootflow.models.nlp.utils import tokenize_bookends


class Tokenizer:
    def __init__(
        self, model_name_or_path: str, max_token_length: int, mode: str = "split"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer = lambda tokenizer_input: self.tokenizer(
            tokenizer_input, padding="max_length", truncation=False
        )
        if mode == "split":
            self._tokenize_function = lambda tokenization_input: tokenize_bookends(
                tokenization_input, max_token_length, tokenizer=self.tokenizer
            )
        # (Reminders to implement these modes)
        elif mode == "first":
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
        elif mode == "last":
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")

    def __call__(self, input_strings: Union[List[str], str]) -> Dict[str, torch.Tensor]:
        return self._tokenize_function(input_strings)


class Transformer(Module):
    def __init__(
        self,
        model_name_or_path: str,
        head: Union[Module, ModuleList, ModuleDict] = None,
    ) -> None:
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


# Copied from the huggingface RobertaClassificationHead
class ClassificationHead(Module):
    def __init__(self, input_size, num_classes, dropout: float = 0.1):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = num_classes
        self.out_proj = torch.nn.Linear(input_size, num_classes)

    def forward(self, features, **kwargs):
        # Needs to first load the correct portion of the
        # model output from the output dictionary
        x = features[0][:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
