from setkit.datasets.base.collection import RootflowDataItem
import pytest


def test_create_data_item():
    for data in [20, "a string", {"dict_val": 100}, [1, 2, "string"]]:
        data_item = RootflowDataItem(data)
        assert data_item.data == data
        assert data_item.id == None
        assert data_item.target == None


def test_create_data_item_single_target():
    data_item = RootflowDataItem(20, target=5)
    assert data_item.target == 5


def test_create_data_item_multi_target():
    data_item = RootflowDataItem(1293, target=["target_one", "target_two", 2, 3])
    assert data_item.target["task-0"] == "target_one"
    assert data_item.target["task-1"] == "target_two"
    assert data_item.target["task-2"] == 2
    assert data_item.target["task-3"] == 3


def test_create_data_item_multi_task():
    data_item = RootflowDataItem(
        52, target={"task_one": "task_one_val", "task_two": 1035}
    )
    assert data_item.target["task_one"] == "task_one_val"
    assert data_item.target["task_two"] == 1035


def test_unpack_data_item():
    data_item = RootflowDataItem(20, target=5)
    id, data, target = data_item
    assert id == None
    assert data == 20
    assert target == 5
