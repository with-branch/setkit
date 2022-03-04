import os

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
from rootflow.datasets.base import RootflowDataset, RootflowDataItem
from os.path import exists
from tqdm import tqdm
import json
import numpy as np
from rootflow.datasets.base.utils import (
    map_functions
)

class EmailCorpus(RootflowDataset):
    BUCKET = "rootflow"
    DATASET_PATH = "datasets/email-notification"
    FILE_NAME = "email_data.mmap"
    CHUNK_SIZE = 5000

    def __init__(self, prefix: str = "", root: str = None, download: bool = True, tasks = None) -> None:
        self.prefix = self.DATASET_PATH + prefix
        super().__init__(root, download, tasks)

    def index(self, index: int) -> tuple:
        id, data, _, _, _ = self.data[index]
        if id is None:
            id = f"{type(self).__name__}-{index}"
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, None)

    def download(self, directory: str):
        from google.cloud import storage
        file_path = os.path.join(directory, self.FILE_NAME)
        storage_client = storage.Client()        

        #count how many files we are going to download
        bucket = storage.Bucket(storage_client, name=self.BUCKET) 
        file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)
        num_files = sum(1 for blob in file_names_iter)
        self.data = np.memmap(file_path, dtype=dict, mode='w+', shape=num_files)
        
        print("Downloading files")
        file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)
        #to keep the zarr of resizing so much we insert a chunk at a time
        #dynamic array could be switched with linked list to avoid resizing
        temp_store = []
        store_index = 0
        mmap_initial_index = 0
        for i, file in tqdm(enumerate(file_names_iter), total=num_files, smoothing=.9):
            data = file.download_as_string()
            json_object = json.loads(data)

            #formatt the data
            data_dict = {"from": json_object["data"]["from"], 
                "subject": json_object["data"]["subject"], "mbox": json_object["data"]["mbox"]}
            full_item = {"id": json_object["label_info"]["example_id"], "data": data_dict, 
            "target": json_object["label_info"]["label"], "oracle_id": json_object["label_info"]["oracle_id"], "group_id": json_object["label_info"]["dataset_id"]}

            if len(temp_store) < self.CHUNK_SIZE:
                temp_store.append(full_item)
                store_index += 1
            elif len(temp_store) == self.CHUNK_SIZE and store_index < self.CHUNK_SIZE:
                temp_store[store_index] = full_item
                store_index += 1
            else:
                #insert full chunk into zarr
                self.data[mmap_initial_index:i] = temp_store
                mmap_initial_index = i
                temp_store[0] = full_item
                store_index = 1

            if i == num_files - 1:
                #could have a partial chunk so we insert now
                self.data[mmap_initial_index:i+1] = temp_store[0:store_index]

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        file_path = os.path.join(directory, self.FILE_NAME)
        if exists( file_path ):
            return np.memmap(file_path, dtype=dict, mode='r+')
        else:
            return FileNotFoundError

dataset = EmailCorpus()



