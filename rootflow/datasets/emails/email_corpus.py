from decimal import InvalidContext
from genericpath import isfile
import os
import mailbox
import time

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

from numpy import False_, full
from rootflow.datasets.base import RootflowDataset, RootflowDataItem
from os.path import exists
from tqdm import tqdm
import json
import numcodecs
import zarr
from rootflow.datasets.base.utils import (
    map_functions,
    batch_enumerate
)

class EmailCorpus(RootflowDataset):
    BUCKET = "rootflow"
    DATASET_PATH = "datasets/email-notification"
    FILE_NAME = "emails.zarr"
    CHUNK_SIZE = 50
    DATA_DELIMITER = "$$$data-separator$$$"

    def __init__(self, prefix: str = "", root: str = None, download: bool = None, tasks = None) -> None:
        self.prefix = self.DATASET_PATH + prefix
        super().__init__(root, download, tasks)

    # def map(self, function: Union[Callable, List[Callable]], targets: bool = False, batch_size: int = None) -> Union["RootflowDataset", "RootflowDatasetView"]:
    #     def get_mapped_item(data_item):
    #         if isinstance(data_item, str):
    #             decoded_item = data_item.split(self.DATA_DELIMITER)
    #             mapped_item = RootflowDataItem("")

    #             mapped_item.id = decoded_item[0]
    #             if not targets:
    #                 mapped_item.data = function(decoded_item[1])
    #                 mapped_item.target = None
    #             else:
    #                 mapped_item.data = decoded_item[2]
    #                 mapped_item.target = function(None)

    #             return mapped_item   
    #         elif isinstance(data_item, RootflowDataItem):
    #             if targets:
    #                 data_item.target = function(data_item.target)
    #             else:
    #                 data_item.data = function(data_item.data)
    #             return data_item

    #         return InvalidContext

    #     def get_attribute(data_item):
    #         if isinstance(data_item, str):
    #             decoded_item = data_item.split(self.DATA_DELIMITER)
    #             if targets:
    #                 return None
    #             else:
    #                 return decoded_item[1]
    #         elif isinstance(data_item, RootflowDataItem):
    #             if targets:
    #                 return data_item.target
    #             else:
    #                 return data_item.data

    #     if batch_size is None:
    #         data_in_memory = []
    #         for idx, data_item in tqdm(enumerate(self.data), total=len(self.data)):
    #             data_in_memory.append(get_mapped_item(self.data[idx]))
    #         self.data = data_in_memory
    #     else:
    #         data_in_memory = []
    #         for slice, batch in batch_enumerate(self.data, batch_size):
    #             mapped_batch_data = function(
    #                 [get_attribute(data_item) for data_item in batch]
    #             )
    #             assert isinstance(mapped_batch_data, Sequence) and not isinstance(
    #                 mapped_batch_data, str
    #             ), "Map function does not return a sequence over batch"
    #             assert len(mapped_batch_data) == len(
    #                 batch
    #             ), "Map function does not return batch of same length as input"
    #             for idx, mapped_example_data in zip(
    #                 range(slice.start, slice.stop), mapped_batch_data
    #             ):
    #                 data_item = self.data[idx]
    #                 if isinstance(data_item, str):
    #                     decoded_item = data_item.split(self.DATA_DELIMITER)
    #                     mapped_item = RootflowDataItem("")

    #                     mapped_item.id = decoded_item[0]
    #                     if not targets:
    #                         mapped_item.data = mapped_example_data
    #                         mapped_item.target = None
    #                     else:
    #                         mapped_item.data = decoded_item[2]
    #                         mapped_item.target = mapped_example_data

    #                     data_in_memory.append(mapped_item)  
    #                 elif isinstance(data_item, RootflowDataItem):
    #                     if targets:
    #                         data_item.target = mapped_example_data
    #                     else:
    #                         data_item.data = mapped_example_data
    #                     self.data[idx] = data_item

    #         if len(data_in_memory) > 0:
    #             self.data = data_in_memory

    #     return self

    def download(self, directory: str):
        from google.cloud import storage
        file_path = os.path.join(directory, self.FILE_NAME)
        print(file_path)
        storage_client = storage.Client()

        #count how many files we are going to download
        bucket = storage.Bucket(storage_client, name=self.BUCKET) 
        file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)
        num_files = sum(1 for blob in file_names_iter)

        if  not exists( file_path ):
            store = zarr.NestedDirectoryStore(file_path)
            self.data = zarr.create(shape=num_files, chunks=self.CHUNK_SIZE, 
                store=store, dtype=str)
        else:
            self.data = zarr.open(file_path, mode='a', shape=num_files, chunks=self.CHUNK_SIZE, dtype=str)

        
        print("Downloading files")
        file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)
        #to keep the zarr of resizing so much we insert a chunk at a time
        #dynamic array could be switched with linked list to avoid resizing
        temp_store = []
        store_index = 0
        zarr_initial_index = 0
        bytes_loaded = 0
        for i, file in tqdm(enumerate(file_names_iter), total=num_files, smoothing=.9):
            data = file.download_as_string()
            json_object = json.loads(data)

            #formatt the data
            data_dict = {"from": json_object["data"]["from"], 
                "subject": json_object["data"]["subject"], "mbox": json_object["data"]["mbox"]}
            full_item = {"id": json_object["label_info"]["example_id"], "data": data_dict, 
            "target": json_object["label_info"]["label"], "oracle_id": json_object["label_info"]["oracle_id"], "group_id": json_object["label_info"]["dataset_id"]}
            bytes_loaded += len(full_item)
            # full_item = json_object["label_info"]["example_id"] + self.DATA_DELIMITER + json_object["data"]["from"] + self.DATA_DELIMITER
            # full_item += json_object["data"]["subject"] + self.DATA_DELIMITER + json_object["data"]["mbox"]

            if len(temp_store) < self.CHUNK_SIZE:
                temp_store.append(str(full_item))
                store_index += 1
            elif len(temp_store) == self.CHUNK_SIZE and store_index < self.CHUNK_SIZE:
                temp_store[store_index] = full_item
                store_index += 1
            else:
                #insert full chunk into zarr
                self.data[zarr_initial_index:i] = temp_store
                zarr_initial_index = i
                temp_store[0] = full_item
                store_index = 1

            if i == num_files - 1:
                #could have a partial chunk so we insert now
                self.data[zarr_initial_index:i+1] = temp_store[0:store_index]

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        self.root = directory
        file_path = os.path.join(directory, self.FILE_NAME)
        if exists( file_path ):
            zarr_file = zarr.open(file_path, mode='r+')
            data_in_memeory = []

            for encoded_string in tqdm(zarr_file, total=len(zarr_file)):
                decoded_string = encoded_string.split(self.DATA_DELIMITER)
                data_item = RootflowDataItem(decoded_string[1], id=decoded_string[0], target=None)
                data_in_memeory.append(data_item)

            return data_in_memeory                              
        else:
            return FileNotFoundError

    def set(self, index, new_data):
        self.data[index].data = new_data


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("EmailCorpus()")

    dataset = EmailCorpus(root='/media/dallin/Linux_2/branch/datasets/emails/zarr', download=False)

