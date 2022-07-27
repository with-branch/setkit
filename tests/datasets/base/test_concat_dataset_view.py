from typing import Tuple
import pytest
from setkit.datasets.base.dataset import (
    RootflowDataItem,
    RootflowDataset,
    ConcatRootflowDatasetView,
)


class DatasetForTesting(RootflowDataset):
    def prepare_data(self, path: str):
        data = [i for i in range(100)]
        targets = [(i % 3) == 1 for i in range(100)]
        ids = [f"data_item-{i}" for i in range(len(data))]
        return [
            RootflowDataItem(data, id=id, target=target)
            for id, data, target in zip(ids, data, targets)
        ]

    def setup(self):
        pass


def test_create_concat_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    assert dataset_view[1]["id"] == "data_item-1"
    assert dataset_view[1]["data"] == 1
    assert dataset_view[1]["target"] == True
    assert dataset_view[len(dataset)]["id"] == "data_item-0"
    assert dataset_view[len(dataset)]["data"] == 0
    assert dataset_view[len(dataset)]["target"] == False
    assert len(dataset_view) == 2 * len(dataset)


def test_slice_concat_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    assert dataset_view[0]["id"] == "data_item-0"
    assert dataset_view[0]["data"] == 0
    assert dataset_view[0]["target"] == False

    dataset_view = dataset_view[2:22:5]
    assert dataset_view[1]["id"] == "data_item-7"
    assert dataset_view[1]["data"] == 7
    assert dataset_view[1]["target"] == True


def test_slice_concat_dataset_view_with_list():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    assert dataset_view[50]["id"] == "data_item-50"
    assert dataset_view[50]["data"] == 50
    assert dataset_view[50]["target"] == False

    indices = [2, 5, 7, 8, 11, 105]
    dataset_view = dataset_view[indices]
    assert dataset_view[5]["id"] == f"data_item-{105 - len(dataset)}"
    assert dataset_view[5]["data"] == 105 - len(dataset)
    assert dataset_view[5]["target"] == ((105 - len(dataset)) % 3 == 1)


def test_map_concat_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    map_function = lambda x: ((x**2) + 1) / 10
    mapped_dataset = dataset_view.map(map_function)
    assert mapped_dataset[4]["data"] == 1.7
    assert dataset.data[4] == 1.7


def test_transform_concat_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    transform_function = lambda x: ((x**2) + 1) / 10
    transformed_dataset = dataset_view.transform(transform_function)
    assert transformed_dataset[4]["data"] == 1.7
    assert dataset.data[4].data == 4


def test_split_concat_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    split_one, split_two = dataset_view.split(seed=42)
    assert split_one[3]["id"] == "data_item-26"
    assert split_one[3]["data"] == 26
    assert split_one[3]["target"] == False
    assert split_two[3]["id"] == "data_item-93"
    assert split_two[3]["data"] == 93
    assert split_two[3]["target"] == False
    assert len(split_one) + len(split_two) == len(dataset_view)


def test_concat_concat_dataset_view_and_dataset():
    dataset = DatasetForTesting()
    dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    concat_result = dataset_view + dataset
    assert concat_result[len(dataset_view)]["id"] == "data_item-0"
    assert concat_result[len(dataset_view)]["data"] == 0
    assert concat_result[len(dataset_view)]["target"] == False
    assert len(concat_result) == len(dataset_view) + len(dataset)


def test_concat_concat_dataset_view_and_dataset_view():
    dataset = DatasetForTesting()
    concat_dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    dataset_view = dataset[5:50:3]
    concat_result = concat_dataset_view + dataset_view
    assert concat_result[len(concat_dataset_view)]["id"] == "data_item-5"
    assert concat_result[len(concat_dataset_view)]["data"] == 5
    assert concat_result[len(concat_dataset_view)]["target"] == False
    assert len(concat_result) == len(concat_dataset_view) + len(dataset_view)


def test_concat_concat_dataset_view_and_concat_dataset_view():
    dataset = DatasetForTesting()
    concat_dataset_view_one = ConcatRootflowDatasetView(dataset, dataset)
    concat_dataset_view_two = ConcatRootflowDatasetView(dataset, dataset)
    concat_result = concat_dataset_view_one + concat_dataset_view_two
    assert concat_result[len(concat_dataset_view_one)]["id"] == "data_item-0"
    assert concat_result[len(concat_dataset_view_one)]["data"] == 0
    assert concat_result[len(concat_dataset_view_one)]["target"] == False
    assert len(concat_result) == len(concat_dataset_view_one) + len(
        concat_dataset_view_two
    )


def test_filter_concat_dataset_view():
    raise NotImplementedError


def test_iter_concat_dataset_view():
    raise NotImplementedError


def test_get_tasks_concat_dataset_view():
    raise NotImplementedError


def test_get_task_shapes_concat_dataset_view():
    raise NotImplementedError


def test_stats_concat_dataset_view():
    raise NotImplementedError


def test_examples_concat_dataset_view():
    raise NotImplementedError


def test_describe_concat_dataset_view():
    raise NotImplementedError
