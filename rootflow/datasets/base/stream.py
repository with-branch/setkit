from typing import Iterable, Any, Iterator, List

import logging
import os

from torch.utils.data import IterableDataset
from rootflow.datasets.base.item import DataItem
from rootflow import __location__ as ROOTFLOW_LOCATION


class StreamDataset(IterableDataset):
    def __init__(
        self, root: str = None, download: bool = None, tasks: List[dict] = []
    ) -> None:
        super().__init__()
        self._DEFAULT_DIRECTORY = os.path.join(
            ROOTFLOW_LOCATION, "datasets/data", type(self).__name__, "data"
        )
        if root is None:
            logging.info(
                f"{type(self).__name__} root is not set, using the default data root of {self._DEFAULT_DIRECTORY}"
            )
            root = self._DEFAULT_DIRECTORY

        if download is None:
            try:
                self._addresses = self.prepare_data(root)
            except FileNotFoundError:
                logging.warning(
                    f"Dataset {type(self).__name__} could not be loaded from location '{root}'."
                )
                download = True

        if download is True:
            logging.info(
                f"Downloading {type(self).__name__} data to location '{root}'."
            )
            if not os.path.exists(root):
                os.makedirs(root)
            self.download(root)
            self._addresses = self.prepare_data(root)
        elif download is False:
            try:
                self._addresses = self.prepare_data(root)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not load the data for {type(self).__name__} from '{root}'\nMake sure that the data is located at '{root}'.\nAlso consider setting download to `True`."
                )
        logging.info(f"Loaded {type(self).__name__} from '{root}'.")

    def __iter__(self) -> Iterator[dict]:
        for address in self._addresses:
            next_item = self.fetch_item(address)
            id, data, target = next_item.id, next_item.data, next_item.target
            if id is None:
                id = address
            yield {"id": id, "data": data, "target": target}

    def prepare_data(self, directory: str) -> Iterable[Any]:
        """Prepares data for a rootflow streaming dataset.

        Returns an iterable of addresses which correspond to the locations of data
        items. These could be file paths, urls, uris or any other method of
        location.

        Args:
            directory (str): The dataset's root directory, if necessary.

        Returns:
            Iterable[Any]: An iterable of the data item addresses.
        """
        raise NotImplementedError

    def download(self, directory: str) -> None:
        """Downloads the data for the dataset to a specified directory.

        Args:
            directory (str): Directory to download the data to.
        """
        raise NotImplementedError

    def fetch_item(self, address: Any) -> DataItem:
        raise NotImplementedError
