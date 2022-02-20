from typing import Callable, Tuple, List, Union
import logging
import random
import os
from torch.utils.data import Dataset
import rootflow
from rootflow.datasets.utils import batch_enumerate, map_functions

# TODO:
# - Sort indices before we use them to construct a DatasetView
#   so the items retain their order
# - Filter out the duplicate indices from DatasetView
# - Decide wether setup should be optional


class FunctionalDataset(Dataset):
    def split(
        self, test_proportion: float = 0.1, seed: int = None
    ) -> Tuple["RootflowDatasetView", "RootflowDatasetView"]:
        dataset_length = len(self)
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        n_test = int(dataset_length * test_proportion)
        return (
            RootflowDatasetView(self, indices[n_test:]),
            RootflowDatasetView(self, indices[:n_test]),
        )

    def map(
        self,
        function: Union[Callable, List[Callable]],
        labels: bool = False,
        batch_size: int = None,
    ):
        raise NotImplementedError

    def transform(
        self, function: Union[Callable, List[Callable]], labels: bool = False
    ):
        raise NotImplementedError

    def stats(self):
        pass

    def examples(self, num_examples: int = 5):
        return [self[i] for i in range(num_examples)]

    def describe(self):
        print(self.stats())
        print(self.examples())


class RootflowDataset(FunctionalDataset):
    def __init__(self, root: str = None, download: bool = True) -> None:
        self.DEFAULT_DIRECTORY = os.path.join(
            rootflow.__location__, "datasets/data", type(self).__name__, "data"
        )
        if root is None:
            logging.info(
                f"{type(self).__name__} root is not set, using the default data root of {self.DEFAULT_DIRECTORY}"
            )
            root = self.DEFAULT_DIRECTORY

        try:
            ids, data, labels = self.prepare_data(root)
        except FileNotFoundError as e:
            logging.warning(f"Data could not be loaded from {root}.")
            if download:
                logging.info(
                    f"Downloading {type(self).__name__} data to location {root}."
                )
                if not os.path.exists(root):
                    os.makedirs(root)
                self.download(root)
                ids, data, labels = self.prepare_data(root)
            else:
                raise e
        logging.info(f"Loaded {type(self).__name__} from {root}.")

        self.data = data
        if ids is None:
            self.ids = [str(i) for i, _ in enumerate(self.data)]
        else:
            self.ids = ids
        self.labels = labels

        self.data_transforms = []
        self.label_transforms = []

        self.setup()
        logging.info(f"Setup {type(self).__name__}.")

    def prepare_data(self, path: str) -> Tuple[list, list, list]:
        raise NotImplementedError

    def download(self, path: str):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def transform(
        self, function: Union[Callable, List[Callable]], labels: bool = False
    ):
        if not isinstance(function, (tuple, list)):
            function = [function]
        if labels:
            self.label_transforms += function
        else:
            self.data_transforms += function
        return self

    def map(
        self,
        function: Union[Callable, List[Callable]],
        labels: bool = False,
        batch_size: int = None,
    ):
        # Represents some dangerous interior mutability
        # Does not play well with views (What should we change and not change. Do we allow different parts of the dataset to have different data?)
        # Does not play well with datasets who need to have data be memmaped from disk
        if labels:
            map_target = self.labels
        else:
            map_target = self.data

        if batch_size is None:
            mapping_generator = enumerate(map_target)
        else:
            mapping_generator = batch_enumerate(map_target, batch_size)

        for slice_or_index, batch_or_example in mapping_generator:
            mapped_data = function(batch_or_example)
            map_target[slice_or_index] = mapped_data

        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return RootflowDatasetView(self, data_indices)
        elif isinstance(index, (tuple, list)):
            return RootflowDatasetView(self, index)
        else:
            if self.labels is None:
                label = None
            else:
                label = map_functions(self.labels[index], self.label_transforms)
            return {
                "id": self.ids[index],
                "data": map_functions(self.data[index], self.data_transforms),
                "label": label,
            }


class RootflowDatasetView(FunctionalDataset):
    def __init__(self, dataset: RootflowDataset, view_indices: List[int]) -> None:
        self.dataset = dataset
        self.data_indices = view_indices

    def map(self, function: Callable, labels: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over a dataset view!")

    def transform(
        self, function: Union[Callable, List[Callable]], labels: bool = False
    ):
        return self.dataset.transform(function=function, labels=labels)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return RootflowDatasetView(self, data_indices)
        elif isinstance(index, (tuple, list)):
            return RootflowDatasetView(self, index)
        else:
            return self.dataset[self.data_indices[index]]
