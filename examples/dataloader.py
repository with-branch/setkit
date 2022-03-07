from rootflow.datasets.emails.email_corpus import EmailCorpus
from rootflow.datasets.base import RootflowDataLoader
from rootflow.models.nlp.transformers import Tokenizer
from tqdm import tqdm

dataset = EmailCorpus(root='/media/dallin/Linux_2/branch/datasets/emails/zarr', download=False)
#tokenizer = Tokenizer("roberta-base", 20)
#dataset.map(tokenizer, batch_size=512)
dataloader = RootflowDataLoader(dataset, batch_size=128, num_workers=8)

for batch in tqdm(dataloader):
    tmep = ""
