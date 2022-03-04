from rootflow.datasets.examples import ExampleNLP
from rootflow.datasets.base import DataLoader
from rootflow.models.nlp.transformers import Tokenizer

dataset = ExampleNLP()
tokenizer = Tokenizer("roberta-base", 20)
dataset.map(tokenizer, batch_size=512)
dataloader = DataLoader(dataset, batch_size=128, num_workers=8)

for batch in dataloader:
    print(batch.keys())
    print(batch["data"])
    break
