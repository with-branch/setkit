from typing import Iterator

from torch.utils.data import IterableDataset
from rootflow.datasets.base.item import DataItem


class GeneratorDataset(IterableDataset):
    def __init__(self, length: int) -> None:
        super().__init__()

        self._length = length

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._length):
            next_item = self.yield_item()
            yield {"data": next_item.data, "target": next_item.target}

    def yield_item(self) -> DataItem:
        raise NotImplementedError(
            "To create a new GeneratorDataset the method yeild_item must be implemented."
        )
