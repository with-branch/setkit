from typing import (
    Callable,
    List,
    Tuple,
    Dict,
    Any,
    Generator,
    Union,
    Hashable,
    Sequence,
    Iterator,
)

import logging
import os
from setkit import __location__ as SETKIT_LOCATION


class DataItem:
    """A single data example for rootflow datasets.

    A container class for data in rootflow datasets, intended to provide a rigid API
    on which the :class:`FunctionalDataset`s can depened. Behaviorally, it is similar
    to a named tuple, since the only available slots are `id`, `data` and `target`.

    Attributes:
        id (:obj:`Hashable`, optional): A unique id for the dataset example.
        data (Any): The data of the dataset example.
        target (:obj:`Any`, optional): The task target(s) for the dataset example.
    """

    __slots__ = ("id", "data", "target", "metadata")

    def __init__(
        self,
        data: Any,
        target: Any = None,
        id: Hashable = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Creates a new data item.

        Args:
            id (:obj:`Hashable`, optional): A unique id for the dataset example.
            data (Any): The data of the dataset example.
            target (:obj:`Any`, optional): The task target(s) for the dataset example
            metadata (:obj:`Dict[str, Any]`, optional): Additional metadata for the dataset example.
        """
        self.data = data
        self.id = id
        self.target = target
        self.metadata = metadata

    def __getitem__(self, index: int):
        if index == 0:
            return self.id
        elif index == 1:
            return self.data
        elif index == 2:
            return self.target
        elif index == 3:
            return self.metadata
        else:
            raise ValueError(f"Invalid index {index} for CollectionDataItem")

    def __iter__(self) -> Iterator[Tuple[Hashable, Any, Any]]:
        """Returns an iterator to support tuple unpacking

        For example:
            >>> data_item = CollectionDataItem([1, 2, 3], id = 'item', target = 0)
            >>> id, data, target, metadata = data_item
        """
        return iter((self.id, self.data, self.target, self.metadata))


class Dataset:
    def __init__(
        self,
        root: str = None,
        download: bool = None,
        cache_limit: float = None,
        cache: bool = True,
    ) -> None:
        self._transform_functions = ([], [], [])
        self._has_transforms = (False, False, False)
        self.DEFAULT_DIRECTORY = os.path.join(
            SETKIT_LOCATION, "datasets/data", self.__class__.__name__, "data"
        )
        if root is None:
            logging.info(
                f"{type(self).__name__} root is not set, using the default data root of {self.DEFAULT_DIRECTORY}"
            )
            root = self.DEFAULT_DIRECTORY

        self._data_paths = self.get_datapaths(root)

    ## Guarded methods

    def _set_iteration_order(self, order: Dict[int, int]) -> None:
        """Sets the iteration order.

        The dataset set will use this order to drop items from the cache which are
        furthest in the iteration order.

        Args:
            order (Dict[int, int]): A list mapping of a data item index to its order in
                the iteration.
        """
        raise NotImplementedError

    ## Override these methods to implement a new dataset

    def download(self, path: str) -> None:
        """Downloads the data for the dataset to a specified directory.

        Args:
            directory (str): Directory to download the data to.
        """
        raise NotImplementedError

    def get_datapaths(self, location: str) -> List[str]:
        raise NotImplementedError

    def prepare_data(
        self, path: str
    ) -> Union[Generator[DataItem, None, None], DataItem]:
        """Prepares data for a rootflow dataset.

        Loads the data from a directory path and either returns a single data item,
        or returns a generator of data items, depending on how many examples are found
        at the given path.

        Args:
            directory (str): The directory where we should look for our data.

        Returns:
            :class:`DataItem`: A data item for the dataset.

        Yields:
            :class:`DataItem`: A data item for the dataset.
        """
        raise NotImplementedError

    def get_index(self, index: int) -> DataItem:
        """Gets a data item at a given index.

        Args:
            index (int): The index of the data item you would like to get.

        Raises:
            IndexError: If the index is out of bounds.
        """
        raise NotImplementedError

    def set_index(self, index: int, item: DataItem) -> DataItem:
        """Sets a data item at the given index.

        Args:
            index (int): The index of the data item you would like to set.
            item (DataItem): The data item you would like to set.

        Raises:
            IndexError: If the index is out of bounds.
        """
        raise NotImplementedError

    ## Base dataset API

    def split(
        self, splits: Union[float, Sequence[float], Dict[str, float]], seed: int = None
    ) -> Union[Sequence["Dataset"], Dict[str, "Dataset"]]:
        raise NotImplementedError

    def map(
        self,
        function: Union[Callable, List[Callable]],
        map_over: str = "data",
        batch_size: int = None,
    ) -> "Dataset":
        """Maps a function over the dataset.

        Applies some given function(s) to each dataset example contained within the
        dataset. Returns `self` to assist with the functional API, but mutates internal
        state so is not functional at all.

        Args:
            function (Union[Callable, List[Callable]]): The function or functions you
                would like to map over the dataset.
            map_over (:obj:`str`, optional): The slot in the dataset you would like
                to map over. Options are: `data`, `target` and `metadata`. Defaults to
                mapping over the `data` slot.
            batch_size (:obj:`int`, optional): A batch size, if the functions to map
                support or require batches of inputs.

        Raises:
            AssertionError: If a batched function does not return a list of the same
                length as its inputs.
        """
        raise NotImplementedError

    def where(
        self, condition: Callable, map_over: str = "data", batch_size: int = None
    ) -> "Dataset":
        raise NotImplementedError

    def transform(
        self, function: Union[Callable, List[Callable]], map_over: str = "data"
    ) -> "Dataset":
        raise NotImplementedError

    def __add__(self, other: "Dataset") -> "Dataset":
        raise NotImplementedError

    def __sub__(self, other: "Dataset") -> "Dataset":
        raise NotImplementedError

    def ids(self) -> List[Hashable]:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def examples(self, num_examples: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def summary(self, output_width: int = None) -> str:
        # If output_width is None, use the default terminal width
        raise NotImplementedError

    ##  Dunder Methods

    def __getitem__(
        self, index: Union[int, slice, tuple, list]
    ) -> Union[Dict[str, Any], "Dataset"]:
        """Indexes dataset

        If the index specified is an integer, the dataset will call its get_index method,
        pack the result into a dictionary, and return it. However, if the index is
        instead a slice or a list of integers, the dataset will return a view of itself
        with the appropriate indices.

        Args:
            index Union[int, slice, tuple, list]: Specifies the portion of the dataset
                to select.

        Returns:
            Union[dict, DatasetView]: Either a single data item, containing an
                `"id"`, `"data"` and a `"target"`, or a :class:`DatasetView`
                of the desired indices.

        Raises:
            IndexError: If the index is out of bounds.
            TypeError: If the index is not an integer, slice, tuple or list.
        """
        raise NotImplementedError

    def __setitem__(
        self,
        index: Union[int, slice, tuple, list],
        item: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Sets a data item at a given index.

        If the index specified is an integer, the dataset will call its set_index method,
        and attempt to coerce the given dictionary into a :class:`DataItem`. However, if
        the index is instead a slice or a list of integers, the dataset will attempt to
        coerce and assign to the specified slice

        Args:
            index Union[int, slice, tuple, list]: Specifies the portion of the dataset
                to select.
            item Union[Dict[str, Any], Sequence[Dict[str, Any]]]: The data item you
                would like to set.

        Returns:
            Union[dict, DatasetView]: Either a single data item, containing an
                `"id"`, `"data"` and a `"target"`, or a :class:`DatasetView`
                of the replaced values.

        Raises:
            ValueError: If the index is out of bounds.
            ValueError: If the dictionary or list of dictionaries which are given are
                not valid data items.
            TypeError: If the index is not an integer, slice, tuple or list.
            TypeError: If the item is not a dictionary or a list of dictionaries.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Gets the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Should use this method to set the cache order
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.summary()
