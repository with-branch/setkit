from typing import List

import torch
from torch.nn import Module, ModuleDict, ModuleList
from pytorch_lightning import LightningModule

from rootflow.models.auto.utils import verify_tasks


class RootflowAutoModel(LightningModule):
    def __init__(self, tasks: List[dict]) -> None:
        super().__init__()
        verify_tasks(tasks)
