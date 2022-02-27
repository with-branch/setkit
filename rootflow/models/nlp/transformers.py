from typing import List, Mapping, Union, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from rootflow.models.nlp.utils import tokenize_bookends


class Tokenizer:
    def __init__(
        self, model_name_or_path: str, max_token_length: int, mode: str = "split"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if mode == "split":
            tokenizer(
                tokenization_input, padding="max_length", truncation=False
            )  # Do something with this
            self._tokenize_function = lambda tokenization_input: tokenize_bookends(
                tokenization_input, max_token_length, tokenizer=self.tokenizer
            )
        elif mode == "first":
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
        elif mode == "last":
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")

    def __call__(self, input_strings: Union[List[str], str]) -> Dict[str, torch.Tensor]:
        return self._tokenize_function(input_strings)


class ClassificationTransformer(torch.nn.Module):
    def __init__(self, model_name_or_path: str, task_shapes: Union[dict, int]) -> None:
        transformer_model_with_head = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=1
            )
        )
        self.transformer = getattr(
            transformer_model_with_head, transformer_model_with_head.base_model_prefix
        )
        self.config = transformer_model_with_head.config
        if isinstance(task_shapes, Mapping):
            self.is_multitask = True
            self.classification_heads = {
                task_name: ClassificationHead(self.config, task_shape)
                for task_name, task_shape in task_shapes.items()
            }
        else:
            self.is_multitask = False
            self.classification_head = ClassificationHead

    def forward(self, *args, **kwargs):
        transformer_outputs = self.transformer(*args, **kwargs)
        if self.is_multitask:
            return {
                task_name: task_head(transformer_outputs)
                for task_name, task_head in self.classification_heads.items()
            }
        else:
            return self.classification_head(transformer_outputs)


# Copied from the huggingface RobertaClassificationHead
class ClassificationHead(torch.nn.Module):
    def __init__(self, config, task_shape):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.task_shape = task_shape
        self.out_proj = torch.nn.Linear(
            config.hidden_size, task_shape  # Multiply out the shape if necessary
        )  # Should be able to reshape to whatever task shape was defined

    def forward(self, features, **kwargs):
        x = features[0][:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
