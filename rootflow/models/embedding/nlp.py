from typing import Union
from transformers import AutoModel
from rootflow.models.embedding.base import Embedder


class TransformerEmbedder(Embedder):
    def __init__(self, model_name_or_path: Union[str, tuple]) -> None:
        if isinstance(model_name_or_path, (tuple, list)):
            encoder_type, decoder_type = model_name_or_path
        else:
            encoder_type = model_name_or_path
            decoder_type = model_name_or_path
        encoder = AutoModel.from_pretrained(encoder_type)
        decoder = AutoModel.from_pretrained(decoder_type)
        super().__init__(encoder=encoder, decoder=decoder)
        raise NotImplementedError
        # TODO The decoder transformer must still be converted into a decoder using the config

    def forward(self, x):
        raise NotImplementedError
