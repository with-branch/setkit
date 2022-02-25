import timeit
import matplotlib.pyplot as plt
from rootflow.datasets.examples import ExampleNLP

n_dataset_views = 20
n_indices = 10000
n_repetitions = 100

dataset = ExampleNLP()
dataset_view = dataset
for i in range(n_dataset_views):
    dataset_view = dataset_view[: len(dataset) - (i + 1)]


def run_test(dataset, n_tests, n_indices):
    indices = [i % len(dataset) for i in range(n_indices)]
    time = timeit.timeit(
        "for i in indices:\n\tdataset[i]",
        number=n_tests,
        globals={"dataset": dataset, "indices": indices},
    ) / (n_tests * n_indices)
    return time * 1e6


view_indexing_speed = []
for n_views in range(n_dataset_views):
    dataset_view = dataset
    for i in range(n_views):
        dataset_view = dataset_view[: len(dataset) - (i + 1)]
    tests = [run_test(dataset_view, n_repetitions, n_indices) for _ in range(2)]
    result = sum(tests) / 2
    view_indexing_speed.append(result)

plt.plot(view_indexing_speed)
plt.show()
