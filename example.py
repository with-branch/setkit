from rootflow.datasets import ExampleDataset
import logging

dataset = ExampleDataset()
print(dataset[0])


my_view = dataset[:200].transform(lambda x : x + 1)