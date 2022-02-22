from typing import Tuple
import pytest
from rootflow.datasets.base import (
    RootflowDataset,
    RootflowDataItem,
    ConcatRootflowDatasetView,
)


class TestDataset(RootflowDataset):
    def prepare_data(self, path: str):
        data = [i for i in range(100)]
        labels = [(i % 3) == 1 for i in range(100)]
        ids = [f"data_item-{i}" for i in range(len(data))]
        return [
            RootflowDataItem(data, id=id, label=label)
            for id, data, label in zip(ids, data, labels)
        ]

    def setup(self):
        pass


def test_create_dataset():
    dataset = TestDataset()
    assert dataset[0]["id"] == "data_item-0"
    assert dataset[0]["data"] == 0
    assert dataset[0]["label"] == False
    assert len(dataset) == 100


def test_create_dataset_no_ids():
    dataset = TestDataset()
    assert dataset[0]["id"] == "0"
    assert dataset[0]["data"] == 0
    assert dataset[0]["label"] == False
    assert len(dataset) == 100


def test_slice_dataset():
    dataset = TestDataset()

    dataset_view = dataset[50:]
    assert dataset_view[0]["id"] == "data_item-50"
    assert dataset_view[0]["data"] == 50
    assert dataset_view[0]["label"] == False

    dataset_view = dataset[17:38]
    assert dataset_view[0]["id"] == "data_item-17"
    assert dataset_view[0]["data"] == 17
    assert dataset_view[0]["label"] == False
    assert dataset_view[-1]["id"] == "data_item-37"
    assert dataset_view[-1]["data"] == 37
    assert dataset_view[-1]["label"] == True

    dataset_view = dataset[20:30:4]
    assert dataset_view[1]["id"] == "data_item-24"
    assert dataset_view[1]["data"] == 24
    assert dataset_view[1]["label"] == False


def test_slice_datset_with_list():
    dataset = TestDataset()

    indices = [2, 3, 6, 12, 26, 32]
    dataset_view = dataset[indices]
    assert dataset_view[2]["id"] == "data_item-6"
    assert dataset_view[2]["data"] == 6
    assert dataset_view[2]["label"] == False

    indices = [7, 3, 6, 55, 12, 26, 31, 2]
    dataset_view = dataset[indices]
    assert dataset_view[6]["id"] == "data_item-31"
    assert dataset_view[6]["data"] == 31
    assert dataset_view[6]["label"] == True


def test_map_dataset():
    dataset = TestDataset()
    map_function = lambda x: ((x**2) + 1) / 10
    mapped_dataset = dataset.map(map_function)
    assert mapped_dataset[4]["data"] == 1.7
    assert mapped_dataset.data[4] == 1.7


def test_transform_dataset():
    dataset = TestDataset()
    transform_function = lambda x: ((x**2) + 1) / 10
    transformed_dataset = dataset.transform(transform_function)
    assert transformed_dataset[4]["data"] == 1.7
    assert transformed_dataset.data[4] == 4


def test_split_dataset():
    dataset = TestDataset()
    split_one, split_two = dataset.split(seed=42)
    assert split_one[3]["id"] == "data_item-56"
    assert split_one[3]["data"] == 56
    assert split_one[3]["label"] == False
    assert split_two[8]["id"] == "data_item-15"
    assert split_two[8]["data"] == 15
    assert split_two[8]["label"] == False


def test_concat_dataset_and_dataset():
    dataset = TestDataset()
    concat_result = dataset + dataset
    assert concat_result[len(dataset)]["id"] == "data_item-0"
    assert concat_result[len(dataset)]["data"] == 0
    assert concat_result[len(dataset)]["label"] == False
    assert len(concat_result) == 2 * len(dataset)


def test_concat_dataset_and_dataset_view():
    dataset = TestDataset()
    dataset_view = dataset[5:30]
    concat_result = dataset + dataset_view
    assert concat_result[len(dataset)]["id"] == "data_item-5"
    assert concat_result[len(dataset)]["data"] == 5
    assert concat_result[len(dataset)]["label"] == False
    assert len(concat_result) == len(dataset) + len(dataset_view)


def test_concat_dataset_and_concat_dataset_view():
    dataset = TestDataset()
    concat_dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    concat_result = dataset + concat_dataset_view
    assert concat_result[len(dataset)]["id"] == "data_item-5"
    assert concat_result[len(dataset)]["data"] == 5
    assert concat_result[len(dataset)]["label"] == False
    assert len(concat_result) == len(dataset) + len(concat_dataset_view)
