"""Base class for rootflow datasets.

Implements the basics of rootflow's functional API for rootflow datasets and
dataset-like objects. (For example RootflowDatasetView)
"""

from typing import Callable, Sequence, Tuple, List, Union
import os
import random
from torch.utils.data import Dataset

import rootflow.datasets.base.collection as rootflow_datasets
from rootflow.datasets.base.utils import get_nested_data_types
from rootflow.datasets.base.display_utils import (
    format_docstring,
    format_examples_tabular,
    format_statistics,
)


class RootflowDataset(Dataset):
    """Abstract class for rootflow's functional dataset API.

    Implements shared behavior for RootflowDataset, RootflowDatasetView, and
    ConcatRootflowDatasetView. This includes things like slicable indexing,
    and formatted display functionality.
    """

    def __init__(self) -> None:
        self._data_transforms = []
        self._target_transforms = []
        self._has_data_transforms = False
        self._has_target_transforms = False

    def __iter__(self):
        """Iterates over dataset

        Yields:
            dict: A dictionary containing an `"id"`, `"data"` and a `"target"`
        """
        raise NotImplementedError

    def split(
        self, validation_proportion: float = 0.1, seed: int = None
    ) -> Tuple[
        "rootflow_datasets.RootflowDatasetView", "rootflow_datasets.RootflowDatasetView"
    ]:
        """Splits the dataset into a train set and a validation set.

        Will return two dataset views of the dataset. Each view is randomly sampled
        from the dataset, and they do not contain any of the same elements.

        Args:
            validation_proportion (float): The proportion of the total dataset size
                to contain in the validation set.
            seed (:obj:`int`, optional): An optional seed for the randomization.

        Returns:
            Tuple[RootflowDatasetView, RootflowDatasetView]: A tuple containing,
                respectively, the train set and the validation set.
        """
        dataset_length = len(self)
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        n_test = int(dataset_length * validation_proportion)
        return (
            rootflow_datasets.RootflowDatasetView(self, indices[n_test:], sorted=False),
            rootflow_datasets.RootflowDatasetView(self, indices[:n_test], sorted=False),
        )

    def transform(
        self, function: Union[Callable, List[Callable]], targets: bool = False
    ) -> Union[
        "rootflow_datasets.RootflowDataset", "rootflow_datasets.RootflowDatasetView"
    ]:
        """Applies a transform to the dataset

        Adds a new transform to the dataset, after all current transforms. The new
        transform will be run each time an item is selected from this dataset.
        (Useful for random augmentation, for example).

        :meth:`transform` returns self to better support a functional API. Keep in
        mind that it is not truly functional, and that the dataset is modified in
        place for space, storage and speed concerns.

        Args:
            function (Union[Callable, List[Callable]]): The transform function or
                list of functions you would like to add.
            targets (:obj:`bool`, optional): Wether this transform should apply
                to the dataset targets.

        Returns:
            Union[RootflowDataset, RootflowDatasetView]: Returns `self`.
        """
        if not isinstance(function, (tuple, list)):
            function = [function]
        if targets:
            self._target_transforms += function
            self._has_target_transforms = True
        else:
            self._data_transforms += function
            self._has_data_transforms = True
        return self

    def tasks(self):
        """Returns dataset target tasks"""
        raise NotImplementedError

    def stats(self):
        """Gets common statistics for dataset.

        Calculates a set of common and useful statistics for the dataset. May attempt
        to decide dynamically which statistics to include, depending on the data and
        target types.

        Returns:
            dict: A dictionary of the collected statistics.
        """
        data_example = self[0]["data"]
        target_example = self[0]["target"]
        tasks = self.tasks()
        if tasks is not None and len(tasks) == 1:
            tasks = tasks[0]
        return {
            "length": len(self),
            "data_types": get_nested_data_types(data_example),
            "target_types": get_nested_data_types(target_example),
            "tasks": tasks,
        }

    def examples(self, num_examples: int = 5) -> List[dict]:
        """Returns multiple examples from the dataset

        Gets a list of examples from the dataset, up to the specified number or the
        size of the dataset, if the requested amount is too large.

        Args:
            num_examples (int): The number of examples to get.

        Returns:
            List[dict]: A list of examples from the dataset.
        """
        num_examples = min(len(self), num_examples)
        return [self[i] for i in range(num_examples)]

    def summary(self, output_width: int = None):
        """Print a formatted summary of the dataset

        Collects the dataset docstring, statistics and some examples, then prints them
        in a formatted summary to the console.

        Args:
            output_width (:obj:`int`, optional): Optional control for the width of the
                formatted output. Defaults to 150 or the console width, whichever is
                smaller.
        """
        terminal_size = os.get_terminal_size()
        if output_width is None:
            description_width = min(150, terminal_size.columns)
        else:
            description_width = output_width

        print(f"{type(self).__name__}:")
        print(format_docstring(type(self).__doc__, description_width, indent=True))

        print("\nStats:")
        print(format_statistics(self.stats(), description_width, indent=True))

        print("\nExamples:")
        print(format_examples_tabular(self.examples(), description_width, indent=True))
