import numpy as np

from rootflow.datasets.base import GeneratorDataset, DataItem


class FourierTransform(GeneratorDataset):
    def __init__(self, num_signals=10, sample_rate=500):
        self.num_signals = 10
        self.sample_rate = sample_rate
        super().__init__()

    def yeild_item(self) -> DataItem:
        frequencies = np.random.uniform(1, 20, size=(self.num_signals))
        inputs = np.linspace(0, 50, num=self.sample_rate)
        # Do some more math stuff
        data = [0.1, 1.0, 2.0, 0.1]
        return DataItem(data=data, target=frequencies)
