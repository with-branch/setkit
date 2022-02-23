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
    value = obj
    for function in function_list:
        value = function(value)
    return value


def get_unique(input_iterator, ordered=True):
    if ordered:
        unique = list(set(input_iterator))
        unique.sort()
        return unique
    else:
        seen = set()
        seen_add = seen.add
        return [item for item in input_iterator if not (item in seen or seen_add(item))]
