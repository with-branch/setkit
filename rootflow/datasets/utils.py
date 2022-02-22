from functools import reduce
from typing import Callable, Iterable
from torch.utils.data.dataloader import default_collate


def id_collate(unprocessed_batch):
    batch_without_ids = []
    ids = []
    for id, data, label in unprocessed_batch:
        batch_without_ids.append((data, label))
        ids.append(id)
    processed_data, processed_labels = default_collate(batch_without_ids)
    return (ids, processed_data, processed_labels[0])


def batch(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield iterable[ndx:upper]


def batch_enumerate(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield (slice(ndx, upper), iterable[ndx:upper])


def map_functions(obj: object, function_list: Iterable[Callable]):
    return reduce(lambda o, func: func(o), function_list, obj)
