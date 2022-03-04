from typing import Tuple

import torch
from torch import Module

# TODO It may be better long run to change the embedder return type to a dict
# especially when considering the case of nested embedders.
class Embedder(Module):
    def __init__(self, encoder: Module, decoder: Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return (embedding, self.decoder(embedding))
