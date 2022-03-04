from torch.nn import Module, ModuleDict, ModuleList
from typing import Union

from transformers import AutoModel


class TransformerSeq2Seq(Module):
    def __init__(self, model_name_or_path: Union[str, tuple]) -> None:
        super().__init__()
        if isinstance(model_name_or_path, (tuple, list)):
            encoder_type, decoder_type = model_name_or_path
        else:
            encoder_type = model_name_or_path
            decoder_type = model_name_or_path
        self.encoder = AutoModel.from_pretrained(encoder_type)
        self.decoder = AutoModel.from_pretrained(decoder_type)

    def forward(self, x):
        pass
