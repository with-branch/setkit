from decimal import InvalidContext
from genericpath import isfile
import os
import mailbox
import time
import zipfile

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
    ZARR_CLOUD_PATH = "datasets/email-notification-zarr/mbox-no-attachments"
    ZARR_ZIP_NAME = "emails.zip"
    ZARR_NAME = "emails.zarr"
    CHUNK_SIZE = 50
    DATA_DELIMITER = "$$$data-separator$$$"

    def __init__(self, path_to_zarr_in_cloud: str = "", root: str = None, download: bool = None, tasks = None) -> None:
        if path_to_zarr_in_cloud != "":
            self.ZARR_CLOUD_PATH = path_to_zarr_in_cloud
        super().__init__(root, download, tasks)

    def download(self, directory: str):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage.Bucket(storage_client, name=self.BUCKET)

        zipped_zarr_blob = storage.Blob(self.ZARR_CLOUD_PATH + "/" + self.ZARR_ZIP_NAME, bucket)
        path_to_zip  = os.path.join(directory, "emails.zip")

        print("Downloading from Cloud Storage")
        zipped_zarr_blob.download_to_filename(path_to_zip)

        with zipfile.ZipFile(path_to_zip, 'r') as zip_file_ref:
            print("Extracting the zarr file")
            zip_file_ref.extractall(directory)

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        self.root = directory
        file_path = os.path.join(directory, self.ZARR_NAME)
        if exists( file_path ):
            zarr_file = zarr.open(file_path, mode='r+')
            data_in_memeory = []

            #zarr format
            # id ZARR_DELIMITER mbox ZARR_DELIMITER label ZARR_DELIMITER oracle_id 
            for encoded_string in tqdm(zarr_file, total=len(zarr_file)):
                decoded_string = encoded_string.split(self.DATA_DELIMITER)
                data_item = RootflowDataItem(decoded_string[1], id=decoded_string[0], target=None)
                data_in_memeory.append(data_item)

            return data_in_memeory                              
        else:
            raise FileNotFoundError


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("EmailCorpus()")

    dataset = EmailCorpus(root='/mnt/3913be04-1a62-4a3d-b5c4-b804c51bfe73/branch/datasets/emails/zarr', download=False)

