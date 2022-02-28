from typing import Callable, Sequence, Tuple, List, Union
import os
import random
from torch.utils.data import Dataset

import rootflow.datasets.base.dataset as rootflow_datasets
from rootflow.datasets.base.utils import get_nested_data_types
from rootflow.datasets.base.display_utils import (
    format_docstring,
    format_examples_tabular,
    format_statistics,
)


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
            id, data, target = self.index(index)
            return {"id": id, "data": data, "target": target}
        elif isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return rootflow_datasets.RootflowDatasetView(
                self, data_indices, sorted=False
            )
        elif isinstance(index, (tuple, list)):
            return rootflow_datasets.RootflowDatasetView(self, index)

    def __iter__(self):
        for index in range(len(self)):
            id, data, target = self.index(index)
            yield {"id": id, "data": data, "target": target}

    def index(self, index) -> tuple:
        raise NotImplementedError

    def split(
        self, test_proportion: float = 0.1, seed: int = None
    ) -> Tuple[
        "rootflow_datasets.RootflowDatasetView", "rootflow_datasets.RootflowDatasetView"
    ]:
        dataset_length = len(self)
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        n_test = int(dataset_length * test_proportion)
        return (
            rootflow_datasets.RootflowDatasetView(self, indices[n_test:], sorted=False),
            rootflow_datasets.RootflowDatasetView(self, indices[:n_test], sorted=False),
        )

    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union[
        "rootflow_datasets.RootflowDataset", "rootflow_datasets.RootflowDatasetView"
    ]:
        raise NotImplementedError

    def where(
        self,
        filter_function: Callable,
        targets: bool = False,
    ) -> "rootflow_datasets.RootflowDatasetView":
        if targets:
            conditional_attr = "target"
        else:
            conditional_attr = "data"

        filtered_indices = []
        for index, item in enumerate(self):
            if filter_function(item[conditional_attr]):
                filtered_indices.append(index)
        return rootflow_datasets.RootflowDatasetView(self, filtered_indices)

    def transform(
        self, function: Union[Callable, List[Callable]], targets: bool = False
    ) -> Union[
        "rootflow_datasets.RootflowDataset", "rootflow_datasets.RootflowDatasetView"
    ]:
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

        return rootflow_datasets.ConcatRootflowDatasetView(self, object)

    def tasks(self):
        raise NotImplementedError

    def stats(self):
        data_example = self[0]["data"]
        target_example = self[0]["target"]
        tasks = self.tasks()
        if len(tasks) == 1:
            tasks = tasks[0]
        return {
            "length": len(self),
            "data_types": get_nested_data_types(data_example),
            "target_types": get_nested_data_types(target_example),
            "tasks": tasks,
        }

    def examples(self, num_examples: int = 5):
        return [self[i] for i in range(num_examples)]

    def describe(self, output_width: int = None):
        terminal_size = os.get_terminal_size()
        if output_width is None:
            description_width = min(150, terminal_size.columns)
        else:
            description_width = output_width

        print(f"{type(self).__name__}:")
        dataset_doc = type(self).__doc__
        if not dataset_doc is None:
            print(format_docstring(dataset_doc, description_width))
        else:
            print("(No Description)")

        print("\nStats:")
        print(format_statistics(self.stats(), description_width, indent=True))

        print("\nExamples:")
        print(format_examples_tabular(self.examples(), description_width, indent=True))
