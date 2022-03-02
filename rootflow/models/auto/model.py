from typing import List

from torch.nn import Module, ModuleDict, ModuleList
from rootflow.models.auto.utils import verify_tasks


class RootflowAutoModel(Module):
    def __init__(self, tasks: List[dict]) -> None:
        super().__init__()
        verify_tasks(tasks)
