from rootflow.datasets.examples import ExampleTextCorpus
from rootflow.datasets.base.loader import RootflowDataLoader
from rootflow.models.nlp.transformers import Tokenizer
from tqdm import tqdm

dataset = ExampleTextCorpus()
tokenizer = Tokenizer("roberta-base", 20)
dataset.map(tokenizer, batch_size=128)
dataloader = RootflowDataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)

print(dataset[0])

for batch in tqdm(dataloader):
    break
print(batch)
