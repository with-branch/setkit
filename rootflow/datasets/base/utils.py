from typing import Any, Callable, Iterable, Mapping, Sequence, Union, Tuple
import torch
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


def get_nested_data_types(object: Any) -> Union[dict, list, type]:
    if isinstance(object, Sequence) and not isinstance(object, str):
        return [get_nested_data_types(element) for element in object]
    elif isinstance(object, Mapping):
        return {key: get_nested_data_types(value) for key, value in object.items()}
    else:
        return type(object)


def infer_task_from_targets(target_list: list) -> Tuple[str, tuple]:
    if not target_list:
        return None

    first_target = next(target_list)
    if isinstance(first_target, Sequence) and not isinstance(first_target, str):
        first_target_element = next(first_target_element)
        if isinstance(first_target_element, (int, torch.LongTensor)):
            # This needs to be adjusted to work with >1D tensors
            max_element = max([max(target) for target in target_list])
            if max_element > 1:
                return ("multitarget", max_element)
            max_list_sum = max([sum(target) for target in target_list])
            if max_list_sum > 1:
                return ("multitarget", len(first_target))
            else:
                return ("classification", len(first_target))
        elif isinstance(first_target_element, (bool, torch.BoolTensor)):
            pass
        elif isinstance(first_target_element, (float, torch.FloatTensor)):
            return ("regression", (len(first_target), *first_target_element.shape))
    elif isinstance(first_target, (int, torch.LongTensor)):
        max_class_val = max(target_list)
        if max_class_val > 1:
            return ("classification", max_class_val)
        else:
            return ("binary", 2)
    elif isinstance(first_target, (bool, torch.BoolTensor)):
        return ("binary", 2)
    elif isinstance(first_target, (float, torch.FloatTensor)):
        return ("regression", first_target.shape)
    else:
        return (None, None)
