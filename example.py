from rootflow.datasets import ExampleDataset
import logging
import time
import cProfile

dataset = ExampleDataset()
print(dataset[0])


first_200_incremented = dataset[:200].transform(lambda x: x + 1, targets=True)
second_200_decremented = dataset[200:400].transform(lambda x: x - 1, targets=True)
leftovers = dataset[400:]

modified_dataset = first_200_incremented + second_200_decremented + leftovers

train_set, test_set = modified_dataset.split()
train_set = train_set.transform(lambda x: f"train-{x}")
train_set = train_set.transform(lambda x: x * 2, targets=True)

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


def create_test(set_to_test, n_to_test=1000000):
    indices = [i % len(set_to_test) for i in range(n_to_test)]

    def test():
        start_time = time.time()
        for idx in indices:
            set_to_test[idx]
        end_time = time.time()
        print(
            f"Loaded examples at {((end_time - start_time) * 1000) / n_to_test}ms per example"
        )

    return test


create_test(dataset)()
create_test(train_set)()
# test = create_test(train_set)
# cProfile.run("test()")
