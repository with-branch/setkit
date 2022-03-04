from rootflow.datasets.emails.email_corpus import EmailCorpus
from rootflow.datasets.base import DataLoader
from rootflow.models.nlp.transformers import Tokenizer
from tqdm import tqdm

dataset = EmailCorpus()
#tokenizer = Tokenizer("roberta-base", 20)
#dataset.map(tokenizer, batch_size=512)
dataloader = DataLoader(dataset, batch_size=128, num_workers=8)

for batch in tqdm(dataloader):
    tmep = ""
