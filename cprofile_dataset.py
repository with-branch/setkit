from rootflow.datasets.examples import ExampleNLP
import cProfile

n_indices = 100000

dataset = ExampleNLP()
dataset_view = dataset
for i in range(10):
    dataset_view = dataset_view[: len(dataset) - (i + 1)]

indices = [i % len(dataset_view) for i in range(n_indices)]


def test():
    for idx in indices:
        dataset_view[idx]


cProfile.run("test()")
