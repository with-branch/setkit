from typing import List

import os
from os.path import exists
import zipfile
from tqdm import tqdm
import json
import zarr

from rootflow.datasets.base.dataset import RootflowDatasetView
from rootflow.datasets.base import RootflowDataset, RootflowDataItem
from rootflow import __location__ as ROOTFLOW_LOCATION


class LabeledEmails(RootflowDataset):
    """Labeled emails from the Branch email corpus.

    Inherits from :class:`RootflowDataset`.
    All emails from the branch corpus which were labeled by an oracle. Targets are
    binary classifications indicating if the particular labeler would have liked to
    receive that email as a notification. The specific prompt was as follows:
        "If you received this email today, would you want to receive a notification
        or alert about the email, based on its contents?"
    """

    BUCKET = "rootflow"
    ZARR_CLOUD_PATH = "datasets/email-notification-zarr/mbox-no-attachments"
    ZARR_ZIP_NAME = "emails.zip"
    ZARR_NAME = "emails.zarr"
    CHUNK_SIZE = 50
    DATA_DELIMITER = "$$$data-separator$$$"
    LABEL_ENCODING = {"False": 0, "True": 1}

    def __init__(
        self,
        path_to_zarr_in_cloud: str = "",
        prefix: str = "",
        google_credentials: str = None,
        root: str = None,
        download: bool = None,
        tasks=None,
    ) -> None:
        if path_to_zarr_in_cloud != "":
            self.ZARR_CLOUD_PATH = path_to_zarr_in_cloud
        self.prefix = self.ZARR_CLOUD_PATH + prefix
        self.GOOGLE_CREDENTIALS = google_credentials
        if root is None:
            root = os.path.join(
                ROOTFLOW_LOCATION, "datasets/data", "EmailCorpus", "data"
            )
        super().__init__(root, download, tasks)

    def download(self, directory: str):
        from google.cloud import storage

        try:
            if self.GOOGLE_CREDENTIALS != None:
                storage_client = storage.Client.from_service_account_json(
                    self.GOOGLE_CREDENTIALS
                )
            else:
                storage_client = storage.Client(credentials=self.GOOGLE_CREDENTIALS)
        except OSError:
            print(
                "The google storage client errored out because it did not have the correct credentials"
            )
            print(
                "You must set the GOOGLE_APPLICATION_CREDENTIALS env variable or pass in the file path to the json file containing a service account key"
            )
            raise OSError
        bucket = storage.Bucket(storage_client, name=self.BUCKET)

        zipped_zarr_blob = storage.Blob(
            self.ZARR_CLOUD_PATH + "/" + self.ZARR_ZIP_NAME, bucket
        )
        path_to_zip = os.path.join(directory, "emails.zip")

        print("Downloading from Cloud Storage")
        zipped_zarr_blob.download_to_filename(path_to_zip)

        with zipfile.ZipFile(path_to_zip, "r") as zip_file_ref:
            print("Extracting the zarr file")
            zip_file_ref.extractall(directory)

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        self.root = directory
        file_path = os.path.join(directory, self.ZARR_NAME)
        if exists(file_path):
            zarr_file = zarr.open(file_path, mode="r+")
            data_items = []

            for encoded_string in tqdm(zarr_file, total=len(zarr_file)):
                id, mbox, label, oracle_id, dataset_id = encoded_string.split(
                    self.DATA_DELIMITER
                )
                if label == "":
                    continue

                data = {"mbox": mbox, "oracle_id": oracle_id}
                data_item = RootflowDataItem(
                    data, id=id, target=self.LABEL_ENCODING[label]
                )
                data_items.append(data_item)

            return data_items
        else:
            return FileNotFoundError

    def split_by_oracle_id(self) -> List[RootflowDataset]:
        """Splits dataset into a list of datasets.

        Creates a dataset for each labeler (oracle) in the Branch email corpus. The
        datasets are views, so no data is duplicated.

        Returns:
            List[RootflowDataset] : A list of the datasets, split by oracle_id.
        """
        oracle_indices = {}
        for idx, data_item in self.data:
            oracle_id = data_item.data["oracle_id"]
            oracle_indices[oracle_id].append(idx)
        oracle_datasets = [
            RootflowDatasetView(self, indices) for indices in oracle_indices.values()
        ]
        return oracle_datasets

    def index(self, index: int) -> tuple:
        data_item = self.data[index]
        id, data, target = data_item.id, data_item.data, data_item.target
        data = data["mbox"]
        return (id, data, target)


if __name__ == "__main__":
    dataset = LabeledEmails()
    dataset[0]
