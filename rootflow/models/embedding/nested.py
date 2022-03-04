from ast import Mod
from torch.nn import Module, ModuleList, ModuleDict

from rootflow.models.embedding.base import Embedder


class NestedEmbedder(Module):
    def __init__(self, outer_embedder: Embedder, inner_embedder: Embedder) -> None:
        super().__init__()
        self.outer = outer_embedder
        self.inner = inner_embedder

    def forward(self, x):
        intermediate_encoding = self.outer.encoder(x)
        embedding = self.inner.encoder(intermediate_encoding)
        intermediate_decoding = self.inner.decoder(embedding)
        return (embedding, self.outer.decoder(intermediate_decoding))
        # Somehow return all hidden states
