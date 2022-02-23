from typing import Tuple
import pytest
from rootflow.datasets.base import (
    ConcatRootflowDatasetView,
    RootflowDataItem,
    RootflowDataset,
    RootflowDatasetView,
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


def test_create_dataset_view():
    dataset = DatasetForTesting()
    view_indices = [1, 6, 2, 7, 3, 10]
    dataset_view = RootflowDatasetView(dataset, view_indices)
    assert dataset_view[0]["id"] == "data_item-1"
    assert dataset_view[0]["data"] == 1
    assert dataset_view[0]["target"] == True
    assert len(dataset_view) == 6


def test_create_dataset_view_duplicate_indices():
    dataset = DatasetForTesting()
    view_indices = [1, 6, 2, 7, 3, 10, 3, 6]
    dataset_view = RootflowDatasetView(dataset, view_indices)
    with pytest.raises(IndexError):
        dataset_view[7]


def test_slice_dataset_view():
    dataset = DatasetForTesting()

    dataset_view = dataset[50:100:3]
    assert dataset_view[0]["id"] == "data_item-50"
    assert dataset_view[0]["data"] == 50
    assert dataset_view[0]["target"] == False

    dataset_view = dataset_view[2:22:5]
    assert dataset_view[1]["id"] == "data_item-71"
    assert dataset_view[1]["data"] == 71
    assert dataset_view[1]["target"] == False


def test_slice_dataset_view_with_list():
    dataset = DatasetForTesting()

    dataset_view = dataset[50:100:3]
    assert dataset_view[0]["id"] == "data_item-50"
    assert dataset_view[0]["data"] == 50
    assert dataset_view[0]["target"] == False

    indices = [2, 5, 7, 8, 11]
    dataset_view = dataset_view[indices]
    assert dataset_view[3]["id"] == "data_item-74"
    assert dataset_view[3]["data"] == 74
    assert dataset_view[3]["target"] == False


def test_map_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = dataset[50:100:3]
    map_function = lambda x: ((x**2) + 1) / 10
    mapped_dataset = dataset_view.map(map_function)
    assert mapped_dataset[4]["data"] == 1.7
    assert dataset.data[4].data == 1.7


def test_transform_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = dataset[:50]
    transform_function = lambda x: ((x**2) + 1) / 10
    transformed_dataset = dataset_view.transform(transform_function)
    assert transformed_dataset[4]["data"] == 1.7
    assert dataset.data[4].data == 4


def test_split_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = dataset[2:88]
    split_one, split_two = dataset_view.split(seed=42)
    assert split_one[3]["id"] == "data_item-5"
    assert split_one[3]["data"] == 5
    assert split_one[3]["target"] == False
    assert split_two[3]["id"] == "data_item-51"
    assert split_two[3]["data"] == 51
    assert split_two[3]["target"] == False
    assert len(split_one) + len(split_two) == len(dataset_view)


def test_concat_dataset_view_and_dataset():
    dataset = DatasetForTesting()
    dataset_view = dataset[60:]
    concat_result = dataset_view + dataset
    assert concat_result[len(dataset_view)]["id"] == "data_item-0"
    assert concat_result[len(dataset_view)]["data"] == 0
    assert concat_result[len(dataset_view)]["target"] == False
    assert len(concat_result) == len(dataset_view) + len(dataset)


def test_concat_dataset_view_and_dataset_view():
    dataset = DatasetForTesting()
    dataset_view_one = dataset[5:30]
    dataset_view_two = dataset[40:]
    concat_result = dataset_view_one + dataset_view_two
    assert concat_result[len(dataset_view_one)]["id"] == "data_item-40"
    assert concat_result[len(dataset_view_one)]["data"] == 40
    assert concat_result[len(dataset_view_one)]["target"] == True
    assert len(concat_result) == len(dataset_view_one) + len(dataset_view_two)


def test_concat_dataset_view_and_concat_dataset_view():
    dataset = DatasetForTesting()
    dataset_view = dataset[:40]
    concat_dataset_view = ConcatRootflowDatasetView(dataset, dataset)
    concat_result = dataset_view + concat_dataset_view
    assert concat_result[len(dataset_view)]["id"] == "data_item-0"
    assert concat_result[len(dataset_view)]["data"] == 0
    assert concat_result[len(dataset_view)]["target"] == False
    assert len(concat_result) == len(dataset_view) + len(concat_dataset_view)
