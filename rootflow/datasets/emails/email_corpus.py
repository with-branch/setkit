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

    def map(self, function: Union[Callable, List[Callable]], targets: bool = False, batch_size: int = None) -> Union["RootflowDataset", "RootflowDatasetView"]:
        def get_mapped_item(data_item):
            if isinstance(data_item, str):
                decoded_item = data_item.split(self.DATA_DELIMITER)
                mapped_item = RootflowDataItem()

                mapped_item.id = decoded_item[0]
                if not targets:
                    mapped_item.data = function(decoded_item[1])
                    mapped_item.target = None
                else:
                    mapped_item.data = decoded_item[2]
                    mapped_item.target = function(None)

                return mapped_item   
            elif isinstance(data_item, RootflowDataItem):
                if targets:
                    data_item.target = function(data_item.target)
                else:
                    data_item.data = function(data_item.data)
                return data_item

            return InvalidContext

        def get_attribute(data_item):
            if isinstance(data_item, str):
                decoded_item = data_item.split(self.DATA_DELIMITER)
                if targets:
                    return None
                else:
                    return decoded_item[1]
            elif isinstance(data_item, RootflowDataItem):
                if targets:
                    return data_item.target
                else:
                    return data_item.data

        if batch_size is None:
            for idx, data_item in enumerate(self.data):
                self.data[idx] = get_mapped_item(self.data[idx])
        else:
            for slice, batch in batch_enumerate(self.data, batch_size):
                mapped_batch_data = function(
                    [get_attribute(data_item) for data_item in batch]
                )
                assert isinstance(mapped_batch_data, Sequence) and not isinstance(
                    mapped_batch_data, str
                ), "Map function does not return a sequence over batch"
                assert len(mapped_batch_data) == len(
                    batch
                ), "Map function does not return batch of same length as input"
                for idx, mapped_example_data in zip(
                    range(slice.start, slice.stop), mapped_batch_data
                ):
                    data_item = self.data[idx]
                    if isinstance(data_item, str):
                        decoded_item = data_item.split(self.DATA_DELIMITER)
                        mapped_item = RootflowDataItem()

                        mapped_item.id = decoded_item[0]
                        if not targets:
                            mapped_item.data = mapped_example_data
                            mapped_item.target = None
                        else:
                            mapped_item.data = decoded_item[2]
                            mapped_item.target = mapped_example_data

                        self.data[idx] = mapped_item  
                    elif isinstance(data_item, RootflowDataItem):
                        if targets:
                            data_item.target = mapped_example_data
                        else:
                            data_item.data = mapped_example_data
                        self.data[idx] = data_item

        return self

    def index(self, index: int) -> tuple:
        data_item = self.data[index]
        if isinstance(data_item, str):
            # id ZARR_DELIMITER mbox ZARR_DELIMITER label ZARR_DELIMITER oracle_id ZARR_DELIMITER dataset_id
            compiled_string = data_item
            decoded_string = compiled_string.split(self.DATA_DELIMITER)
            id = decoded_string[0]
            data = decoded_string[1]
            target = None 
        elif isinstance(data_item, RootflowDataItem):
            id = data_item.id
            data = data_item.data
            target = data_item.target

        if id is None:
            id = f"{type(self).__name__}-{index}"
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)

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
            return zarr.open(file_path, mode='r')                              
        else:
            return FileNotFoundError

def remove_attachments_from_mbox_string(mbox_string):
        mbox_message = mailbox.mboxMessage(mbox_string)

        start = time.time()
        mbox_message = remove_attachments_from_mbox_message(mbox_message)
        end = time.time()
        tot = end-start

        return mbox_message.as_string(), tot
    
#recursive function that removes both inline and normal attachments
def remove_attachments_from_mbox_message(mbox_message):
    #only messages that are multipart have attachments
    if mbox_message.is_multipart():
        for i, part in enumerate(mbox_message.get_payload()):
            # if part.is_multipart():
            #     for i, sub_part in enumerate(part.get_payload()):
            #         if sub_part.get_content_disposition() in ['inline', 'attachment']:
            #             #remove attachment
            #             mbox_message.get_payload()[i] = ""

            if part.get_content_disposition() in ['inline', 'attachment']:
                #remove attachment
                mbox_message.get_payload()[i] = ""

    return mbox_message

def check_mbox_message_part_for_attachment(mbox_part):
    if mbox_part.get_content_disposition() in ['inline', 'attachment']:
        return True
    
    return False


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("EmailCorpus()")

    dataset = EmailCorpus(root='/media/dallin/Linux_2/branch/datasets/emails/zarr', download=False)

