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
    def __init__(
        self, root: str = None, download: bool = True, tasks: List[dict] = None
    ) -> None:
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

    def prepare_data(self, path: str) -> List["RootflowDataItem"]:
        raise NotImplementedError

    def download(self, path: str):
        raise NotImplementedError

    def setup(self):
        pass

    def tasks(self):
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

    def index(self, index):
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
