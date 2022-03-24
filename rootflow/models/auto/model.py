from typing import List

import torch
from torch.nn import Module, ModuleDict, ModuleList
from pytorch_lightning import LightningModule

from rootflow.models.auto.utils import verify_tasks


class RootflowAutoModel(LightningModule):
    def __new__(cls, tasks: List[dict], *args, **kwargs):
        verify_tasks(tasks)
