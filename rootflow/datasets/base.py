from typing import Callable, Tuple, List, Union
import logging
import random
import os
from torch.utils.data import Dataset
import rootflow
from rootflow.datasets.utils import batch_enumerate, map_functions

# TODO:
# - Decide if setup should be optional


class FunctionalDataset(Dataset):
    def __init__(self) -> None:
        self.data_transforms = []
        self.target_transforms = []

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
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["RootflowDataset", "RootflowDatasetView"]:
        raise NotImplementedError

    def transform(
        self, function: Union[Callable, List[Callable]], targets: bool = False
    ) -> Union["RootflowDataset", "RootflowDatasetView"]:
        if not isinstance(function, (tuple, list)):
            function = [function]
        if targets:
            self.target_transforms += function
        else:
            self.data_transforms += function
        return self

    def __add__(self, object):
        if not isinstance(object, FunctionalDataset):
            raise AttributeError(f"Cannot add a dataset to {type(object)}")

        return ConcatRootflowDatasetView(self, object)

    def stats(self):
        pass

    def examples(self, num_examples: int = 5):
        return [self[i] for i in range(num_examples)]

    def describe(self):
        print(self.stats())
        print(self.examples())


class RootflowDataset(FunctionalDataset):
    def __init__(self, root: str = None, download: bool = True) -> None:
        super().__init__()
        self.DEFAULT_DIRECTORY = os.path.join(
            rootflow.__location__, "datasets/data", type(self).__name__, "data"
        )
        if root is None:
            logging.info(
                f"{type(self).__name__} root is not set, using the default data root of {self.DEFAULT_DIRECTORY}"
            )
            root = self.DEFAULT_DIRECTORY

        try:
            self.data = self.prepare_data(root)
        except FileNotFoundError as e:
            logging.warning(f"Data could not be loaded from {root}.")
            if download:
                logging.info(
                    f"Downloading {type(self).__name__} data to location {root}."
                )
                if not os.path.exists(root):
                    os.makedirs(root)
                self.download(root)
                self.data = self.prepare_data(root)
            else:
                raise e
        logging.info(f"Loaded {type(self).__name__} from {root}.")

        self.setup()
        logging.info(f"Setup {type(self).__name__}.")

    def prepare_data(self, path: str) -> List["RootflowDataItem"]:
        raise NotImplementedError

    def download(self, path: str):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["RootflowDataset", "RootflowDatasetView"]:
        # Represents some dangerous interior mutability
        # Does not play well with views (What should we change and not change. Do we allow different parts of the dataset to have different data?)
        # Does not play well with datasets who need to have data be memmaped from disk
        if targets:
            map_target = self.targets
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
            id, data, target = self.data[index]
            if id is None:
                id = f"{type(self).__name__}-{index}"
            return {
                "id": id,
                "data": map_functions(data, self.data_transforms),
                "target": map_functions(target, self.target_transforms),
            }


class RootflowDatasetView(FunctionalDataset):
    def __init__(
        self,
        dataset: Union[RootflowDataset, "RootflowDatasetView"],
        view_indices: List[int],
    ) -> None:
        super().__init__()
        self.dataset = dataset
        unique_indices = list(set(view_indices))
        unique_indices.sort()
        self.data_indices = unique_indices

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over a dataset view!")

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return RootflowDatasetView(self, data_indices)
        elif isinstance(index, (tuple, list)):
            return RootflowDatasetView(self, index)
        else:
            data_item = self.dataset[self.data_indices[index]]
            return {
                "id": data_item["id"],
                "data": map_functions(data_item["data"], self.data_transforms),
                "target": map_functions(data_item["target"], self.target_transforms),
            }


class ConcatRootflowDatasetView(FunctionalDataset):
    def __init__(
        self,
        datatset_one: Union[RootflowDataset, "RootflowDatasetView"],
        dataset_two: Union[RootflowDataset, "RootflowDatasetView"],
    ):
        self.dataset_one = datatset_one
        self.dataset_two = dataset_two
        self.transition_point = len(datatset_one)

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over concatenated datasets!")

    def __len__(self):
        return len(self.dataset_one) + len(self.dataset_two)

    def __getitem__(self, index):
        if isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return RootflowDatasetView(self, data_indices)
        elif isinstance(index, (tuple, list)):
            return RootflowDatasetView(self, index)
        else:
            if index < self.transition_point:
                selected_dataset = self.dataset_one
            else:
                selected_dataset = self.dataset_two
                index -= self.transition_point
            data_item = selected_dataset[index]
            return {
                "id": data_item["id"],
                "data": map_functions(data_item["data"], self.data_transforms),
                "target": map_functions(data_item["target"], self.target_transforms),
            }


class RootflowDataItem:
    __slots__ = ("id", "data", "target")

    def __init__(self, data, id=None, target=None) -> None:
        self.data = data
        self.id = id
        if isinstance(target, (tuple, list)):
            target = {
                f"task-{task_idx}": task_target
                for task_idx, task_target in enumerate(target)
            }
        if not isinstance(target, dict) and not target is None:
            target = {"task": target}

        self.target = target

    def __iter__(self):
        return iter((self.id, self.data, self.target))
