import optuna

from rootflow.datasets.examples import ExampleTabular
from rootflow.models.linear import AutoLinear
from rootflow.tasks import SupervisedTrainer


def transformer_fit(trail: optuna.Trail):
    n_lr_cycles = trail.suggest_int("n_lr_cycles", 1, 10)

    dataset = ExampleTabular()
    train_set, test_set = dataset.split()
    model = AutoLinear(task_shapes=dataset.task_shapes())
