from rootflow.training.base import RootflowTrainer


class TransformerEmbeddingTrainer(RootflowTrainer):
    def train(self) -> None:
        raise NotImplementedError
