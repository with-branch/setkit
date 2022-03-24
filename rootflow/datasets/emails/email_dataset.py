from decimal import InvalidContext
from genericpath import isfile
import os
import zipfile

from typing import List

from rootflow.datasets.base import RootflowDataset, RootflowDataItem
from os.path import exists
from tqdm import tqdm
import zarr

class EmailDataset(RootflowDataset):
    BUCKET = "rootflow"
    ZARR_CLOUD_PATH = "datasets/emails/zarr/mbox-no-attachments"
    ZARR_ZIP_NAME = "emails.zip"
    ZARR_NAME = "emails.zarr"
    CHUNK_SIZE = 50
    DATA_DELIMITER = "$$$data-separator$$$"

    def __init__(self, low_memory: bool = False, path_to_zarr_in_cloud: str = "", google_credentials: str = None, root: str = None, download: bool = None, tasks = None) -> None:
        if path_to_zarr_in_cloud != "":
            self.ZARR_CLOUD_PATH = path_to_zarr_in_cloud
        self.GOOGLE_CREDENTIALS = google_credentials
        self.LOW_MEMORY = low_memory
        super().__init__(root, download, tasks)

    def download(self, directory: str):
        from google.cloud import storage      
        try:
            if self.GOOGLE_CREDENTIALS != None:
                storage_client = storage.Client.from_service_account_json(self.GOOGLE_CREDENTIALS)
            else:
                storage_client = storage.Client(credentials=self.GOOGLE_CREDENTIALS)                
        except OSError:
            print("The google storage client errored out because it did not have the correct credentials")
            print("You must set the GOOGLE_APPLICATION_CREDENTIALS env variable or pass in the file path to the json file containing a service account key")
            raise OSError
        
        bucket = storage.Bucket(storage_client, name=self.BUCKET)

        zipped_zarr_blob = storage.Blob(self.ZARR_CLOUD_PATH + "/" + self.ZARR_ZIP_NAME, bucket)
        path_to_zip  = os.path.join(directory, "emails.zip")

        print("Downloading from Cloud Storage")
        zipped_zarr_blob.download_to_filename(path_to_zip)

        with zipfile.ZipFile(path_to_zip, 'r') as zip_file_ref:
            print("Extracting the zarr file")
            zip_file_ref.extractall(directory)


