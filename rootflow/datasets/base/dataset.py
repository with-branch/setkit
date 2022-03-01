"""Base dataset classes for rootflow

Houses RootflowDataset and the associated classes RootflowDatasetView and
ConcatRootflowDatasetView
"""

from typing import Callable, Mapping, Sequence, Tuple, List, Union
import logging
import os
from rootflow import __location__ as ROOTFLOW_LOCATION
from rootflow.datasets.base.functional import FunctionalDataset
from rootflow.datasets.base.utils import (
    batch_enumerate,
    map_functions,
    get_unique,
    infer_task_from_targets,
)


class RootflowDataset(FunctionalDataset):
    """Abstract class for a rootflow dataset.

    Implments boilerplate downloading, loading and indexing functionality. Extends
    :class:`FunctionalDataset`, and provides all of the same functional API for
    interacting with the dataset. Except in special cases, this functionality does not
    need to be implemented if you are extending RootflowDataset, and will be available
    by default.

    Supported Functionality:
        slicing: Can be sliced to generate a new dataset, supports lists and slices.
        `map` and `transform`: Supports transforms and mapping functions over data.
        filtering and `where`: New datasets can be created using conditional functions.
        addition: New datasets may be created using the addition operator, which will
            concatenate two datasets together.
        statistics and summaries: Dynamically calculate statistics on the dataset, and
            display useful summaries about its contents.

    Only one function is necessary to extend a RootflowDataset. That is the method
    :meth:`prepare_data`. If you wish to dynamically download the dataset, then the
    :meth:`download` method should also be implemented. For any additional steps which
    your dataset needs to perform, you may also implement the :meth:`setup` method.
    """

    def __init__(
        self, root: str = None, download: bool = True, tasks: List[dict] = None
    ) -> None:
        """Creates an instance of a rootflow dataset.

        Attempts to load the dataset from root using the :meth:`prepare_data` method.
        Should this fail to find the given file, it will attempt to download the data
        using the :meth:`download` method, after which it will try once again to load
        the data. If a root is not provided, the dataset will default to using the
        the following path:
            <path to rootflow installation>/datasets/data/<dataset class name>/data

        If the dataset is succesfully loaded, it will then attempt to infer the task
        type, given the data targets, if tasks are not provided.

        Args:
            root (:obj:`str`, optional): Where the data is or should be stored.
            download (:obj:`bool`, optional): Whether the dataset should try to
                download should loading the data fail.
            tasks: (:type:`List[bool]`, optional): Dataset task names, types and shapes.
        """
        super().__init__()
        self.DEFAULT_DIRECTORY = os.path.join(
            ROOTFLOW_LOCATION, "datasets/data", type(self).__name__, "data"
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

        if tasks is None:
            tasks = self._infer_tasks()
            logging.info(f"Tasks not specified, setting automatically")
        self._tasks = tasks

    def prepare_data(self, directory: str) -> List["RootflowDataItem"]:
        """Prepares data for a rootflow dataset

        Loads the data from a directory path and returns a list of
        :class:`RootflowDataItem`s, one for each dataset example in dataset.

        Args:
            directory (str): The directory where we should look for our data.

        Returns:
            List[RootflowDataItem]: The loaded data items.
        """
        raise NotImplementedError

    def download(self, directory: str) -> None:
        """Downloads the data for the dataset to a specified directory.

        Args:
            directory (str): Directory to download the data to.
        """
        raise NotImplementedError

    def setup(self):
        """Performs additional setup steps for the dataset"""
        pass

    def tasks(self):
        """Returns a list of dataset tasks

        Returns a list containing each task for the dataset. The tasks are formatted
        as a dictionary with the following fields:
            {
                "name" : <task name> (str),
                "type" : <task type> (str),
                "shape" : <task shape> (tuple)
            }

        Returns:
            List[dict]: The list of tasks associated with the dataset.
        """
        return self._tasks

    def _infer_tasks(self):
        example_targets = self.index(0)[1]
        if isinstance(example_targets, Mapping):
            tasks = []

            def multitask_generator(task_name):
                for item in self:
                    yield item["target"][task_name]

            for task_name in example_targets.keys():
                generator = multitask_generator(task_name)
                task_type, task_shape = infer_task_from_targets(generator)
                tasks.append(
                    {"name": task_name, "type": task_type, "shape": task_shape}
                )
            return tasks
        else:

            def single_task_generator():
                for item in self:
                    yield item["target"]

            task_type, task_shape = infer_task_from_targets(single_task_generator())
            return [{"name": "task", "type": task_type, "shape": task_shape}]

    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["RootflowDataset", "RootflowDatasetView"]:
        """Maps a function over the dataset.

        Applies some given function(s) to each dataset example contained within the
        dataset. Returns `self` to assist with the functional API, but mutates internal
        state so is not functional at all.

        Args:
            function (Union[Callable, List[Callable]]): The function or functions you
                would like to map over the dataset.
            targets (:obj:`bool`, optional): Whether the functions should be mapped
                over the data item targets, instead of the data.
            batch_size (:obj:`int`, optional): A batch size, if the functions to map
                support or require batches of inputs.
        """
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
        """Gets the length of the dataset."""
        return len(self.data)

    def index(self, index: int) -> tuple:
        """Gets a single data example

        Retrieves a single data example at the given index. Since :meth:`index` is used
        internally, it does not pack the result into a dict, instead returning a tuple.
        (This is prefered for performance)

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple of three items, respectively, the id of the data item, the
                data content of the item, and the target of the data item.
        """
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

    def tasks(self):
        return self.dataset.tasks()

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over a dataset view!")

    def __len__(self):
        return len(self.data_indices)

    def index(self, index):
        id, data, target = self.dataset.index(self.data_indices[index])
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

        self._tasks = self._combine_tasks(datatset_one.tasks(), dataset_two.tasks())

    def tasks(self):
        return self._tasks

    # This function is doing a bit too much
    def _combine_tasks(task_list_one, task_list_two):
        tasks = []

        task_names_one = {task["name"] for task in task_list_one}
        task_names_two = {task["name"] for task in task_list_two}
        combined_names = task_names_one + task_names_two

        for task_name in combined_names:
            if task_name in task_names_one and task_name in task_names_two:
                overlapping_task_one = [
                    task for task in task_list_one if task["name"] == task_name
                ][0]
                overlapping_task_two = [
                    task for task in task_list_one if task["name"] == task_name
                ][0]

                task_type_one = overlapping_task_one["type"]
                task_type_two = overlapping_task_two["type"]
                if not task_type_one == task_type_two:
                    raise ValueError(
                        f"Found two tasks with name {task_name} but types {task_type_one} and {task_type_two}"
                    )

                shape_one = overlapping_task_one["shape"]
                shape_two = overlapping_task_two["shape"]
                if not shape_one == shape_two:
                    raise ValueError(
                        f"Found two tasks with name {task_name} and type {task_type_one} but shapes {shape_one} and {shape_two}"
                    )

                tasks.append(overlapping_task_one)
            elif task_name in task_names_one:
                task = [task for task in task_list_one if task["name"] == task_name][0]
                tasks.append(task)
            elif task_name in task_names_two:
                task = [task for task in task_list_two if task["name"] == task_name][0]
                tasks.append(task)
        return tasks

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over concatenated datasets!")

    def tasks(self):
        return self.dataset_one.tasks()

    def task_shapes(self):
        return self.dataset_two.task_shapes()

    def __len__(self):
        return len(self.dataset_one) + len(self.dataset_two)

    def index(self, index):
        if index < self.transition_point:
            selected_dataset = self.dataset_one
        else:
            selected_dataset = self.dataset_two
            index -= self.transition_point
        id, data, target = selected_dataset.index(index)
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
        # We may want to unpack lists with only a single item for mappings and nested lists as well
        if isinstance(target, Sequence) and not isinstance(target, str):
            target_length = len(target)
            if target_length == 0:
                target = None
            elif target_length == 1:
                target = target[0]
        self.target = target

    def __iter__(self):
        return iter((self.id, self.data, self.target))
