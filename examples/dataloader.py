from rootflow.datasets.emails.email_corpus import EmailCorpus
from rootflow.datasets.emails.utils.map_functions import decode_mbox_email_message
from rootflow.datasets.base import RootflowDataLoader
from rootflow.models.nlp.transformers import Tokenizer
from tqdm import tqdm
import mailbox
import time
from ast import literal_eval

def mbox_map (mbox_string):
    mbox_message = mailbox.mboxMessage(mbox_string)
    text, _, _ = decode_mbox_email_message(mbox_message)
    if text == None:
        text = ""
    return text


dataset = EmailCorpus(root='/mnt/3913be04-1a62-4a3d-b5c4-b804c51bfe73/branch/datasets/emails/zarr', download=False)
print(dataset)

for i, data_item in tqdm(enumerate(dataset), total=len(dataset)):
    mbox_message = mailbox.mboxMessage(data_item["data"])
    text, _, _ = decode_mbox_email_message(mbox_message)
    if text == None:
        text = ""
    
    dataset.set(i, text)

print("mapping over the data")
start = time.time()
dataset.map(mbox_map)
end = time.time()
print(f"The total time to map was {end - start}")

dataloader = RootflowDataLoader(dataset, batch_size=128, num_workers=8)

print("loading the data")
for batch in tqdm(dataloader):
    tmep = ""
