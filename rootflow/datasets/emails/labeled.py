from typing import List

import os
from os.path import exists
from tqdm import tqdm
import zarr

from rootflow.datasets.base.dataset import RootflowDataset, RootflowDatasetView
from rootflow.datasets.base import RootflowDataItem
from rootflow.datasets.emails.email_dataset import EmailDataset


class LabeledEmails(EmailDataset):
    """Labeled emails from the Branch email corpus.

    Inherits from :class:`RootflowDataset`.
    All emails from the branch corpus which were labeled by an oracle. Targets are
    binary classifications indicating if the particular labeler would have liked to
    receive that email as a notification. The specific prompt was as follows:
        "If you received this email today, would you want to receive a notification
        or alert about the email, based on its contents?"
    """
    LABEL_ENCODING = {"False": 0, "True": 1}

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
            raise FileNotFoundError

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
    dataset = LabeledEmails(
        root="/mnt/3913be04-1a62-4a3d-b5c4-b804c51bfe73/branch/datasets/emails_zarr/zarr",
        download=False,
        google_credentials="/home/dallin/Branch/service_account/information_gate/potent-zodiac-323320-271d19d4df2e.json",)
