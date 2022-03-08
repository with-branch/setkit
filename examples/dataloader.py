from rootflow.datasets.emails.email_corpus import EmailCorpus
from rootflow.datasets.emails.utils.map_functions import decode_mbox_email_message
from rootflow.datasets.base import RootflowDataLoader
from rootflow.models.nlp.transformers import Tokenizer
from tqdm import tqdm
import mailbox

def mbox_map (mbox_string):
    mbox_message = mailbox.mboxMessage(mbox_string)
    return decode_mbox_email_message(mbox_message)


dataset = EmailCorpus(root='/media/dallin/Linux_2/branch/datasets/emails/zarr', download=False)

print("mapping over the data")
dataset.map(mbox_map)

dataloader = RootflowDataLoader(dataset, batch_size=128, num_workers=8)

print("loading the data")
for batch in tqdm(dataloader):
    tmep = ""
