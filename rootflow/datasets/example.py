import os
import csv
from rootflow.datasets.base import RootflowDataset, RootflowDataItem


class ExampleTabular(RootflowDataset):
    EXAMPLE_DATASET_LENGTH = 1000
    EXAMPLE_DATASET_FILE_NAME = "example.csv"

    def prepare_data(self, path: str):
        with open(os.path.join(path, self.EXAMPLE_DATASET_FILE_NAME)) as data_file:
            csv_data = list(csv.reader(data_file))
        data = []
        for item in csv_data[1:]:
            data.append(
                RootflowDataItem(
                    [int(feature) for feature in item[1:5]],
                    id=item[0],
                    target=int(item[5]),
                )
            )
        return data

    def download(self, path: str):
        ids = [f"example_dataset-{i}" for i in range(self.EXAMPLE_DATASET_LENGTH)]
        data = [
            [  # Just some random functions of i
                int(i**2 / 12) * int(i) + int(i / 10),
                (int(i**2) / (20 * i)) + ((7 * (i % (i % 15))) / int((i**2) / 50)),
                (i % 3) * i / 123,
                i % int((i**2) / 121),
            ]
            for i in range(self.EXAMPLE_DATASET_LENGTH)
        ]
        labels = [i for i in range(self.EXAMPLE_DATASET_LENGTH)]

        data_path = os.path.join(path, self.EXAMPLE_DATASET_FILE_NAME)
        with open(data_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "id",
                    "feature-one",
                    "feature-two",
                    "feature-three",
                    "feature-four",
                    "integer",
                ]
            )
            for id, data, label in zip(ids, data, labels):
                writer.writerow([id, *data, label])


class ExampleNLP(RootflowDataset):
    EXAMPLE_DATASET_LENGTH = 1000
    EXAMPLE_DATASET_FILE_NAME = "example.csv"

    def prepare_data(self, path: str):
        with open(os.path.join(path, self.EXAMPLE_DATASET_FILE_NAME)) as data_file:
            csv_data = list(csv.reader(data_file))
        data = []
        for item in csv_data[1:]:
            data.append(RootflowDataItem(item[1], id=item[0], target=item[2]))
        return data

    def setup(self):
        label_encoding = {"label-0": 0, "label-1": 1}
        for idx, (id, data, label) in enumerate(self.data):
            self.data[idx] = RootflowDataItem(data, id=id, target=label_encoding[label])

    def download(self, path: str):
        ids = [f"example_dataset-{i}" for i in range(self.EXAMPLE_DATASET_LENGTH)]
        data = [
            f"Hello there, my index is {'even' if (i % 2) == 0 else 'odd'}"
            for i in range(self.EXAMPLE_DATASET_LENGTH)
        ]
        labels = [f"label-{i % 2}" for i in range(self.EXAMPLE_DATASET_LENGTH)]

        data_path = os.path.join(path, self.EXAMPLE_DATASET_FILE_NAME)
        with open(data_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["id", "data", "oddness"])
            for data_row in zip(ids, data, labels):
                writer.writerow(data_row)
