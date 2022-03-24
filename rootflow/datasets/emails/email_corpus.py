import os
import zipfile

from typing import List

from rootflow.datasets.base import RootflowDataset, RootflowDataItem
from os.path import exists
from tqdm import tqdm
import zarr


class EmailCorpus(RootflowDataset):
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
        try:
            from google.cloud import storage
            from google.auth.exceptions import DefaultCredentialsError
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "EmailCorpus requires the use of google cloud storage to download, which is not installed in the current environment.\nTo install, run `pip install --upgrade google-cloud-storage`."
            )

        if self.GOOGLE_CREDENTIALS is None:
            try:
                storage_client = storage.Client(credentials=self.GOOGLE_CREDENTIALS)
            except DefaultCredentialsError:
                raise ValueError(
                    "Could not authenticate google storage client for downloading EmailCorpus.\nGOOGLE_APPLICATION_CREDENTIALS environment variable is not set.\nEither set the GOOGLE_APPLICATION_CREDENTIALS environment variable or set `google_credentials` as the file path to a json file containing an authenticated service account key"
                )
        else:
            try:
                storage_client = storage.Client.from_service_account_json(
                    self.GOOGLE_CREDENTIALS
                )
            except OSError:
                raise OSError(
                    f"Could not authenticate google storage client for downloading EmailCorpus.\nReceived an invalid path for `google_credentials`. Could not find the file {self.GOOGLE_CREDENTIALS}"
                )

        bucket = storage.Bucket(storage_client, name=self.BUCKET)

        zarr_blob_name = self.ZARR_CLOUD_PATH + "/" + self.ZARR_ZIP_NAME
        zipped_zarr_blob = bucket.get_blob(zarr_blob_name)
        path_to_zip = os.path.join(directory, "emails.zip")

        print("Downloading EmailCorpus dataset from Cloud Storage...")
        with open(path_to_zip, "wb") as zip_file:
            with tqdm.wrapattr(
                zip_file, "write", total=zipped_zarr_blob.size
            ) as tqdm_file_obj:
                storage_client.download_blob_to_file(zipped_zarr_blob, tqdm_file_obj)

        with zipfile.ZipFile(path_to_zip, "r") as zip_file_ref:
            print("Extracting EmailCorpus zarr file...")
            zip_file_ref.extractall(directory)

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        self.root = directory
        file_path = os.path.join(directory, self.ZARR_NAME)
        if exists( file_path ):
            if self.LOW_MEMORY:
                zarr_file = zarr.open(file_path, mode='r')
            else:
                zarr_file = zarr.open(file_path, mode='r')
                data_in_memeory = []

                #zarr format
                # id ZARR_DELIMITER mbox ZARR_DELIMITER label ZARR_DELIMITER oracle_id
                print("Loading EmailCorpus dataset...") 
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

    dataset = EmailCorpus(
        root="/mnt/3913be04-1a62-4a3d-b5c4-b804c51bfe73/branch/datasets/emails_zarr/zarr",
        download=True,
        google_credentials="/home/dallin/Branch/service_account/information_gate/potent-zodiac-323320-271d19d4df2e.json",
    )
