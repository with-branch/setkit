from rootflow.datasets import ExampleDataset
import logging
import time

dataset = ExampleDataset()
print(dataset[0])


first_200_incremented = dataset[:200]  # .transform(lambda x: x + 1, targets=True)
second_200_decremented = dataset[200:400]  # .transform(lambda x: x - 1, targets=True)
leftovers = dataset[400:]

modified_dataset = first_200_incremented + second_200_decremented + leftovers

train_set, test_set = modified_dataset.split()
train_set = train_set  # .transform(lambda x: f"train-{x}")
train_set = train_set  # .transform(lambda x: x * 2, targets=True)

print(f"Length of train: {len(train_set)}")
print(f"Length of test: {len(test_set)}")
print(f"Length of dataset: {len(dataset)}")
print(f"Length of modified dataset: {len(modified_dataset)}")

n_samples = 5
print("-------TRAIN-------")
for i in range(n_samples):
    print(train_set[i])
print("-------TEST-------")
for i in range(n_samples):
    print(test_set[i])

test_lengths = 10000000

for test_dataset in [dataset, train_set]:
    indices = [i % len(test_dataset) for i in range(test_lengths)]
    start_time = time.time()
    for idx in indices:
        test_dataset[idx]
    end_time = time.time()
    print(
        f"Loaded examples at {((end_time - start_time) * 1000) / test_lengths}ms per example"
    )
