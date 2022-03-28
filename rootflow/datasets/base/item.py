from typing import Any, Hashable, Sequence, Iterator, Tuple

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

    __slots__ = ("id", "data", "target")

    # TODO We may want to unpack lists with only a single item for mappings and nested lists as well
    def __init__(self, data: Any, id: Hashable = None, target: Any = None) -> None:
        """Creates a new data item.

        Args:
            id (:obj:`Hashable`, optional): A unique id for the dataset example.
            data (Any): The data of the dataset example.
            target (:obj:`Any`, optional): The task target(s) for the dataset example
        """
        self.data = data
        self.id = id

        if isinstance(target, Sequence) and not isinstance(target, str):
            target_length = len(target)
            if target_length == 0:
                target = None
            elif target_length == 1:
                target = target[0]
        self.target = target

    def __getitem__(self, index: int):
        if index == 0:
            return self.id
        elif index == 1:
            return self.data
        elif index == 2:
            return self.target
        else:
            raise ValueError(f"Invalid index {index} for CollectionDataItem")

    def __iter__(self) -> Iterator[Tuple[Hashable, Any, Any]]:
        """Returns an iterator to support tuple unpacking

        For example:
            >>> data_item = CollectionDataItem([1, 2, 3], id = 'item', target = 0)
            >>> id, data, target = data_item
        """
        return iter((self.id, self.data, self.target))
