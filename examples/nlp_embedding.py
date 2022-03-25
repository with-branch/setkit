from asyncio import tasks
from rootflow.datasets.examples import ExampleNLP
from rootflow.models.nlp.transformers import Tokenizer
from rootflow.models.embedding import (
    NestedEmbedder,
    TransformerEmbedder,
    LinearEmbedder,
)
from rootflow.training.embedding import TransformerEmbeddingTrainer
from rootflow.training.metrics import F1, Accuracy

n_epochs = 5

dataset = (
    ExampleNLP()
)  # Automatically downloads the data if necessary (places in the rootflow package)


transformer_layers = TransformerEmbedder("roberta-base")
linear_layers = LinearEmbedder(input_size=768, embedding_size=100)
model = NestedEmbedder(transformer_layers, linear_layers)

tokenizer = Tokenizer(
    "roberta-base",
    transformer_layers.encoder.config.max_position_embeddings,  # This sucks, change
    mode="split",
)
dataset = dataset.map(tokenizer, batch_size=256)
train_set, validation_set = dataset.split()
dataset.summary()

classification_trainer = TransformerEmbeddingTrainer(
    "./results",
    model,
    training_dataset=train_set,
    validation_dataset=validation_set,
    metrics=[F1, Accuracy],
    epochs=n_epochs,
)
classification_trainer.train()
