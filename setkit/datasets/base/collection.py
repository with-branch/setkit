"""Base dataset classes for rootflow

Houses CollectionDataset and the associated classes CollectionDatasetView and
ConcatCollectionDatasetView
"""

from typing import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
    Tuple,
    List,
    Union,
    Any,
    Iterator,
)
import logging
import os
from rootflow import __location__ as ROOTFLOW_LOCATION
from rootflow.datasets.base.functional import (
    FunctionalCollectionDataset as FunctionalDataset,
)
from rootflow.datasets.base.item import DataItem
from rootflow.datasets.base.utils import (
    batch_enumerate,
    map_functions,
    get_unique,
    infer_task_from_targets,
)


class CollectionDataset(FunctionalDataset):
    """Abstract class for a rootflow CollectionDataset.

    Implments boilerplate downloading, loading and indexing functionality. Extends
    :class:`FunctionalDataset`, and provides all of the same functional API for
    interacting with the dataset. Except in special cases, this functionality does not
    need to be implemented if you are extending CollectionDataset, and will be available
    by default.

    Supported Functionality:
        slicing: Can be sliced to generate a new dataset, supports lists and slices.
        `map` and `transform`: Supports transforms and mapping functions over data.
        filtering and `where`: New datasets can be created using conditional functions.
        addition: New datasets may be created using the addition operator, which will
            concatenate two datasets together.
        statistics and summaries: Dynamically calculate statistics on the dataset, and
            display useful summaries about its contents.

    Only one function is necessary to extend a CollectionDataset. That is the method
    :meth:`prepare_data`. If you wish to dynamically download the dataset, then the
    :meth:`download` method should also be implemented. For any additional steps which
    your dataset needs to perform, you may also implement the :meth:`setup` method.
    """

    def __init__(
        self, root: str = None, download: bool = None, tasks: List[dict] = []
    ) -> None:
        """Creates an instance of a rootflow dataset.

        Attempts to load the dataset from root using the :meth:`prepare_data` method.
        Should this fail to find the given file, it will attempt to download the data
        using the :meth:`download` method, after which it will try once again to load
        the data.

        If download is `True` the data will always be downloaded, even if it is already
        present. Alternatively if it is `False`, then no download is allowed,
        regardless. In the case of the default, `None`, the data will be downloaded
        only if :meth:`prepare_data` fails.

        If a root is not provided, the dataset will default to using the
        the following path:
            <path to rootflow installation>/datasets/data/<dataset class name>/data

        If the dataset is succesfully loaded, it will then attempt to infer the task
        type, given the data targets, if tasks are not provided.

        Args:
            root (:obj:`str`, optional): Where the data is or should be stored.
            download (:obj:`bool`, optional): Whether the dataset should download the
                data.
            tasks: (:type:`List[bool]`, optional): Dataset task names, types and shapes.
        """
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
                self.data = self.prepare_data(root)
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
            self.data = self.prepare_data(root)
        elif download is False:
            try:
                self.data = self.prepare_data(root)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not load the data for {type(self).__name__} from '{root}'\nMake sure that the data is located at '{root}'.\nAlso consider setting download to `True`."
                )
        logging.info(f"Loaded {type(self).__name__} from '{root}'.")

        # TODO: Tasks should be normally cached, instead of preloaded. There will be
        # many circumstances where they are not necessary, and this could be annoying
        if tasks is not None and len(tasks) == 0:
            tasks = self._infer_tasks()
            logging.info(f"Tasks not specified, setting automatically")
        self._tasks = tasks

    def prepare_data(self, directory: str) -> List[DataItem]:
        """Prepares data for a rootflow dataset.

        Loads the data from a directory path and returns a list of
        :class:`CollectionDataItem`s, one for each dataset example in dataset.

        Args:
            directory (str): The directory where we should look for our data.

        Returns:
            List[DataItem]: The loaded data items.
        """
        raise NotImplementedError

    def download(self, directory: str) -> None:
        """Downloads the data for the dataset to a specified directory.

        Args:
            directory (str): Directory to download the data to.
        """
        raise NotImplementedError

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
        """Splits targets and infers task information"""
        example_targets = self.index(0)[2]
        if example_targets is None:
            return None
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

    # While splitting out the dataset into three lists, (ids, data, targets) might
    # make batch mapping faster because we can do true slice assignment, it may also
    # decrease the load speed for indexing the dataset (cache locality of the different
    # item members for the given index) TODO: Test this
    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["CollectionDataset", "CollectionDatasetView"]:
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

        Raises:
            AssertionError: If a batched function does not return a list of the same
                length as its inputs.
        """
        assert hasattr(
            function, "__call__"
        ), f"Cannot use a value of type {type(function)} to map over dataset. Parameter `function` must be a callable object."

        if targets:
            attribute = "target"
        else:
            attribute = "data"

        if batch_size is None:
            for idx, data_item in enumerate(self.data):
                setattr(data_item, attribute, function(getattr(data_item, attribute)))
                self.data[idx] = data_item
        else:
            for slice, batch in batch_enumerate(self.data, batch_size):
                mapped_batch_data = function(
                    [getattr(data_item, attribute) for data_item in batch]
                )
                assert isinstance(mapped_batch_data, Sequence) and not isinstance(
                    mapped_batch_data, str
                ), f"Map function {function.__name__} does not return a sequence over batch"
                assert len(mapped_batch_data) == len(
                    batch
                ), f"Map function {function.__name__} does not return batch of same length as input"
                for idx, mapped_example_data in zip(
                    range(slice.start, slice.stop), mapped_batch_data
                ):
                    data_item = self.data[idx]
                    setattr(data_item, attribute, mapped_example_data)
                    self.data[idx] = data_item

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


# TODO Add custom getattr for the dataset views so that if there is a custom
# attribute on a dataset, a view of that dataset will have the same attribute
class CollectionDatasetView(FunctionalDataset):
    """Noncopy subset of a dataset.

    A dataset view is a low cost abstraction which allows for interacting with a
    subset of the dataset without duplication of data. Like :class:`CollectionDataset`
    the view extends :class:`FunctionalDataset`, and provides all of the same
    functional API. (i.e. You can map, transform, take slices, etc)
    """

    def __init__(
        self,
        dataset: FunctionalDataset,
        view_indices: List[int],
        sorted: bool = True,
    ) -> None:
        """Creates an new view of a dataset.

        Args:
            dataset (FunctionalDataset): The dataset which we are taking a view of.
            view_indices (List[int]): Indices corresponding to which data items from
                the dataset we would like to include in the view.
            sorted (:obj:`bool`, optional): Wether to sort the indices so that the
                view maintains ordering when iterating.
        """
        super().__init__()
        self.dataset = dataset
        unique_indices = get_unique(view_indices, ordered=sorted)
        self.data_indices = unique_indices

    def tasks(self) -> List[dict]:
        """Returns a list of dataset tasks.

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
        return self.dataset.tasks()

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over a dataset view!")

    def __len__(self):
        """Returns the length of the view"""
        return len(self.data_indices)

    def index(self, index):
        """Gets a single data example.

        Retrieves a single data example at the given index from the underlying dataset.
        This may be a :class:`CollectionDataset` or another :class:`CollectionDatasetView`,
        potentially even a :class:`ConcatCollectionDatasetView`.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple of three items, respectively, the id of the data item, the
                data content of the item, and the target of the data item.
        """
        id, data, target = self.dataset.index(self.data_indices[index])
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)


class ConcatCollectionDatasetView(FunctionalDataset):
    """Noncopy concatenation of two datasets.

    A concat dataset view is a low cost abstraction which allows for interacting with a
    concatenation of two datasets without duplication of data. Like
    :class:`CollectionDataset` the view extends :class:`FunctionalDataset`, and provides
    all of the same functional API. (i.e. You can map, transform, take slices, etc)
    """

    def __init__(
        self,
        datatset_one: FunctionalDataset,
        dataset_two: FunctionalDataset,
    ):
        """Creates an new concatenated view of two datasets.

        Combines the two given datasets to form a concatenated dataset. Indexing with
        i < len(`dataset_one`) will access the first dataset and indexing
        i >= len(`dataset_one`) will access the second dataset. Creating the combination
        will fail if either of the datasets are not an instance of
        :class:`FunctionalDataset` or if the datasets each have the same task with
        different shapes or types.

        Args:
            dataset_one (FunctionalDataset): The first component of our new dataset.
            dataset_two (FunctionalDataset): The second component of our new dataset.
        """
        assert isinstance(datatset_one, FunctionalDataset) and isinstance(
            dataset_two, FunctionalDataset
        ), f"Cannot concatenate {type(datatset_one)} and {type(dataset_two)}!"
        super().__init__()
        self.dataset_one = datatset_one
        self.dataset_two = dataset_two
        self.transition_point = len(datatset_one)

        self._tasks = ConcatCollectionDatasetView._combine_tasks(
            datatset_one.tasks(), dataset_two.tasks()
        )

    def tasks(self):
        """Returns a list of dataset tasks for the two datasets.

        Returns a list containing each unique task for the datasets. The tasks are
        formatted as a dictionary with the following fields:
            {
                "name" : <task name> (str),
                "type" : <task type> (str),
                "shape" : <task shape> (tuple)
            }

        Returns:
            List[dict]: The list of tasks associated with the datasets.
        """
        return self._tasks

    # TODO This function is doing a bit too much maybe should be refactored
    # and some of the functionality moved into utils
    def _combine_tasks(task_list_one, task_list_two):
        """Returns the unique tasks from two lists of tasks"""
        tasks = []

        task_names_one = {task["name"] for task in task_list_one}
        task_names_two = {task["name"] for task in task_list_two}
        combined_names = task_names_one | task_names_two

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

    def __len__(self):
        """Returns the total length of the concatenated datasets."""
        return len(self.dataset_one) + len(self.dataset_two)

    def index(self, index):
        """Gets a single data example.

        Retrieves a single data example at the given index from the underlying datasets.
        These may be a :class:`CollectionDataset` or another :class:`CollectionDatasetView`,
        potentially even a :class:`ConcatCollectionDatasetView`.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple of three items, respectively, the id of the data item, the
                data content of the item, and the target of the data item.
        """
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
