from typing import Iterator, Tuple

from torch.utils.data import IterableDataset
from rootflow.datasets.base.item import DataItem


class GeneratorDataset(IterableDataset):
    def __init__(self, length: int) -> None:
        super().__init__()

        self._length = length

    def __len__(self) -> int:
        return self._length

    def split(
        self, validation_proportion: float = 0.1
    ) -> Tuple[
        'GeneratorDataset', 'GeneratorDataset'
    ]: 
        generator_dataset_type = type(self)
        validation_length = int(self.length * validation_proportion)

        return generator_dataset_type(length = self.length - validation_length), generator_dataset_type(validation_length)

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._length):
            next_item = self.yield_item()
            yield {"data": next_item.data, "target": next_item.target}

    def yield_item(self) -> DataItem:
        raise NotImplementedError(
            "To create a new GeneratorDataset the method yeild_item must be implemented."
        )
