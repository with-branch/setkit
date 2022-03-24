import os
import zipfile

from typing import List

from rootflow.datasets.base import RootflowDataItem
from os.path import exists
from tqdm import tqdm
import zarr

from rootflow.datasets.emails.email_dataset import EmailDataset


class EmailCorpus(EmailDataset):    

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        self.root = directory
        file_path = os.path.join(directory, self.ZARR_NAME)
        if exists( file_path ):
            if self.LOW_MEMORY:
                return zarr.open(file_path, mode='r')
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
        download=False,
        google_credentials="/home/dallin/Branch/service_account/information_gate/potent-zodiac-323320-271d19d4df2e.json",
    )
