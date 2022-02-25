from typing import Callable, Tuple, List, Union
import logging
import random
import os
from torch.utils.data import Dataset
import rootflow
from rootflow.datasets.utils import batch_enumerate, map_functions, get_unique


class FunctionalDataset(Dataset):
    def __init__(self) -> None:
        self.data_transforms = []
        self.target_transforms = []
        self.has_data_transforms = False
        self.has_target_transforms = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, int):
            id, data, target = self._index(index)
            return {"id": id, "data": data, "target": target}
        elif isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return RootflowDatasetView(self, data_indices, sorted=False)
        elif isinstance(index, (tuple, list)):
            return RootflowDatasetView(self, index)

    def _index(self, index) -> tuple:
        raise NotImplementedError

    def split(
        self, test_proportion: float = 0.1, seed: int = None
    ) -> Tuple["RootflowDatasetView", "RootflowDatasetView"]:
        dataset_length = len(self)
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        n_test = int(dataset_length * test_proportion)
        return (
            RootflowDatasetView(self, indices[n_test:], sorted=False),
            RootflowDatasetView(self, indices[:n_test], sorted=False),
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
            self.has_target_transforms = True
        else:
            self.data_transforms += function
            self.has_data_transforms = True
        return self

    def __add__(self, object):
        if not isinstance(object, FunctionalDataset):
            raise AttributeError(f"Cannot add a dataset to {type(object)}")

        return ConcatRootflowDatasetView(self, object)

    def tasks(self):
        raise NotImplementedError

    def task_shapes(self):
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
        pass

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
            attribute = "target"
        else:
            attribute = "data"

        if batch_size is None:
            for idx, example in enumerate(self.data):
                data_item = self.data[idx]
                setattr(data_item, attribute, function(getattr(data_item, attribute)))
        else:
            for slice, batch in batch_enumerate(self.data, batch_size):
                mapped_batch_data = function(
                    [getattr(data_item, attribute) for data_item in batch]
                )
                for idx, mapped_example_data in zip(slice, mapped_batch_data):
                    data_item = self.data[idx]
                    setattr(data_item, attribute, mapped_example_data)

        return self

    def __len__(self) -> int:
        return len(self.data)

    def _index(self, index):
        data_item = self.data[index]
        id, data, target = data_item.id, data_item.data, data_item.target
        if id is None:
            id = f"{type(self).__name__}-{index}"
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)


class RootflowDatasetView(FunctionalDataset):
    def __init__(
        self,
        dataset: Union[RootflowDataset, "RootflowDatasetView"],
        view_indices: List[int],
        sorted=True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        unique_indices = get_unique(view_indices, ordered=sorted)
        self.data_indices = unique_indices

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over a dataset view!")

    def __len__(self):
        return len(self.data_indices)

    def _index(self, index):
        id, data, target = self.dataset._index(self.data_indices[index])
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)


class ConcatRootflowDatasetView(FunctionalDataset):
    def __init__(
        self,
        datatset_one: Union[RootflowDataset, "RootflowDatasetView"],
        dataset_two: Union[RootflowDataset, "RootflowDatasetView"],
    ):
        super().__init__()
        self.dataset_one = datatset_one
        self.dataset_two = dataset_two
        self.transition_point = len(datatset_one)

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over concatenated datasets!")

    def __len__(self):
        return len(self.dataset_one) + len(self.dataset_two)

    def _index(self, index):
        if index < self.transition_point:
            selected_dataset = self.dataset_one
        else:
            selected_dataset = self.dataset_two
            index -= self.transition_point
        id, data, target = selected_dataset._index(index)
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)


class RootflowDataItem:
    __slots__ = ("id", "data", "target")

    def __init__(self, data, id=None, target=None) -> None:
        self.data = data
        self.id = id
        self.target = target  # How do we differentiate between regression, single class or multiclass tasks?

    def __iter__(self):
        return iter((self.id, self.data, self.target))
