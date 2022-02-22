from typing import Tuple
import pytest
from rootflow.datasets.base import (
    ConcatRootflowDatasetView,
    RootflowDataset,
    RootflowDatasetView,
)


class TestDataset(RootflowDataset):
    def prepare_data(self, path: str):
        data = [i for i in range(100)]
        labels = [(i % 3) == 1 for i in range(100)]
        ids = [f"data_item-{i}" for i in range(len(data))]
        return (ids, data, labels)

    def setup(self):
        pass


def test_create_dataset_view():
    dataset = TestDataset()
    view_indices = [1, 6, 2, 7, 3, 10, 3]
    dataset_view = RootflowDatasetView(dataset, view_indices)
    assert dataset_view[0]["id"] == "data_item-1"
    assert dataset_view[0]["data"] == 1
    assert dataset_view[0]["label"] == True
    assert len(dataset_view) == 7


def test_create_dataset_view_duplicate_indices():
    dataset = TestDataset()
    view_indices = [1, 6, 2, 7, 3, 10, 3, 6]
    dataset_view = RootflowDatasetView(dataset, view_indices)
    with pytest.raises(IndexError):
        dataset_view[7]


def test_slice_dataset_view():
    dataset = TestDataset()

    dataset_view = dataset[50:100:3]
    assert dataset_view[0]["id"] == "data_item-50"
    assert dataset_view[0]["data"] == 50
    assert dataset_view[0]["label"] == True

    dataset_view = dataset_view[2:22:5]
    assert dataset_view[1]["id"] == "data_item-71"
    assert dataset_view[1]["data"] == 71
    assert dataset_view[1]["label"] == False


def test_slice_dataset_view_with_list():
    dataset = TestDataset()

    dataset_view = dataset[50:100:3]
    assert dataset_view[0]["id"] == "data_item-50"
    assert dataset_view[0]["data"] == 50
    assert dataset_view[0]["label"] == True

    indices = [2, 5, 7, 8, 11]
    dataset_view = dataset_view[indices]
    assert dataset_view[3]["id"] == "data_item-74"
    assert dataset_view[3]["data"] == 74
    assert dataset_view[3]["label"] == True


def test_map_dataset_view():
    dataset = TestDataset()
    dataset_view = dataset[50:100:3]
    map_function = lambda x: ((x**2) + 1) / 10
    mapped_dataset = dataset_view.map(map_function)
    assert mapped_dataset[4]["data"] == 1.7
    assert dataset.data[4] == 1.7


def test_transform_dataset_view():
    dataset = TestDataset()
    dataset_view = dataset[:50]
    transform_function = lambda x: ((x**2) + 1) / 10
    transformed_dataset = dataset_view.transform(transform_function)
    assert transformed_dataset[4]["data"] == 1.7
    assert dataset.data[4] == 4


def test_split_dataset_view():
    dataset = TestDataset()
    dataset_view = dataset[2:88]
    split_one, split_two = dataset_view.split(seed=42)
    assert split_one[3]["id"] == "data_item-61"
    assert split_one[3]["data"] == 61
    assert split_one[3]["label"] == True
    assert split_two[3]["id"] == "data_item-32"
    assert split_two[3]["data"] == 32
    assert split_two[3]["label"] == False


def test_concat_dataset_view_and_dataset():
    dataset = TestDataset()
    dataset_view = dataset[60:]
    concat_result = dataset_view + dataset
    assert concat_result[len(dataset)]["id"] == "data_item-0"
    assert concat_result[len(dataset)]["data"] == 0
    assert concat_result[len(dataset)]["label"] == False
    assert len(concat_result) == 2 * len(dataset)


def test_concat_dataset_view_and_dataset_view():
    dataset = TestDataset()
    dataset_view_one = dataset[5:30]
    dataset_view_two = dataset[40:]
    concat_result = dataset_view_one + dataset_view_two
    assert concat_result[len(dataset_view_one)]["id"] == "data_item-40"
    assert concat_result[len(dataset_view_one)]["data"] == 40
    assert concat_result[len(dataset_view_one)]["label"] == True
    assert len(concat_result) == len(dataset_view_one) + len(dataset_view_two)


def test_concat_dataset_view_and_concat_dataset_view():
    dataset = TestDataset()
    dataset_view = dataset[:40]
    concat_dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    concat_result = dataset_view + concat_dataset_view
    assert concat_result[len(dataset_view)]["id"] == "data_item-0"
    assert concat_result[len(dataset_view)]["data"] == 0
    assert concat_result[len(dataset_view)]["label"] == False
    assert len(concat_result) == len(dataset_view) + len(concat_dataset_view)
