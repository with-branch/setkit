import numpy as np
import torch

from rootflow.models.embedding.base import Embedder

# TODO Generate the layer sizes for the linear embedder in a more disciplined way.
class LinearEmbedder(Embedder):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        intermediate_size = int(np.mean(input_size, embedding_size))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size),
            torch.nn.Linear(input_size, intermediate_size),
            torch.nn.Linear(intermediate_size, embedding_size),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.Linear(embedding_size, intermediate_size),
            torch.nn.Linear(intermediate_size, input_size),
        )
        super().__init__(self.encoder, self.decoder)
