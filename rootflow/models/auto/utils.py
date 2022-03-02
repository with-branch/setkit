from typing import List
import torch
import numpy as np


def verify_task(task: dict):
    assert "name" in task, "Found task with no 'name'"
    name = task["name"]
    assert "type" in task, f"Task {name} has no 'type'"
    assert isinstance(task["type"], str), f"Task {name} must have a str 'type'"
    assert "shape" in task, f"Task {name} has no 'shape'"
    assert isinstance(
        task["shape"], (tuple, int, torch.LongTensor, torch.Size, np.ndarray)
    ), f"Task {name} 'shape' must be an int, tuple or similar"


def verify_tasks(tasks: List[dict]):
    assert isinstance(tasks, (tuple, list)), "Tasks must be a tuple or list"
    for task in tasks:
        verify_task(task)
