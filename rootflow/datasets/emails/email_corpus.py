import sys  
sys.path.insert(0, '/home/dallin/Branch/rootflow/rootflow/datasets/base')
sys.path.insert(0, '/home/dallin/Branch/rootflow')

from typing import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
    Tuple,
    List,
    Union,
    Any,
    Iterator,
)
from dataset import RootflowDataset, RootflowDataItem
from os.path import exists
from tqdm import tqdm
import json

# importing sys


class DataItem:
    def __init__(self, json_string):
       self.__dict__ = json.loads(json_string)

class EmailCorpus(RootflowDataset):
    BUCKET = "rootflow"
    DATASET_PATH = "datasets/email-notification"

    def __init__(self, prefix: str = "", root: str = None, download: bool = True, tasks: List[dict] = None) -> None:
        self.prefix = self.DATASET_PATH + prefix
        super().__init__(root, download, tasks)

    def download(self, path: str):
        if  not exists(path):
            from google.cloud import storage
            storage_client = storage.Client()

            #grab the name of each file for the prefix
            bucket = storage.Bucket(storage_client, self.bucket) 
            file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)

            #loop through the file names download each one
            print("we are about to download the files")
            for file in tqdm(file_names_iter):
                blob = bucket.get_blob(file)

                print(blob)

            #add the file to the memmap


    def index(self, index):
        return RootflowDataItem(data = "data")

    def prepare_data(self, path: str) -> List["RootflowDataItem"]:
        #return [("id", "data", None)]
        print("prepare_data was called")
        temp = RootflowDataItem(data = "data")
        self.data = []
        self.data.append(temp)
        print(len(self.data))
        return FileNotFoundError

dataset = EmailCorpus()


