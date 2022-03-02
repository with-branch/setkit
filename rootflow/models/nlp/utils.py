from typing import Any, Callable, Mapping, Sequence, Union, List, Dict
import torch
import numpy as np

# Since we want to use batch padding to speed up the transformers later,
# this function should not include extra padding
def get_sequence_bookends_recursive(
    tokens: Union[
        Union[Sequence, Mapping],
        List[Union[Sequence, Mapping]],
        Dict[Any, Union[Sequence, Mapping]],
    ],
    max_token_length: int,
):
    end_token_length = int(max_token_length / 2)
    start_token_length = max_token_length - end_token_length

    if isinstance(tokens, Sequence) and not isinstance(tokens, str):
        element_example = tokens[0]
        if isinstance(element_example, Sequence) and not isinstance(
            element_example, str
        ):
            return [
                get_sequence_bookends_recursive(element, max_token_length)
                for element in tokens
            ]
        else:
            return get_sequence_bookends(tokens, start_token_length, end_token_length)
    elif isinstance(tokens, Mapping):
        return {
            key: get_sequence_bookends_recursive(token_value, max_token_length)
            for key, token_value in tokens.items()
        }
    else:
        return get_sequence_bookends(tokens, start_token_length, end_token_length)


def get_sequence_bookends(sequence, length_start, length_end):
    start_component = sequence[:length_start]
    end_component = sequence[-length_end:]
    if isinstance(sequence, torch.Tensor):
        return torch.cat((start_component, end_component), dim=0)
    else:
        return np.concatenate((start_component, end_component), axis=0)


def listify_tokens(tokens: Dict[Any, list]) -> List[dict]:
    keys = tokens.keys()
    return [
        {key: value for key, value in zip(keys, value_tuple)}
        for value_tuple in zip(*tokens.values())
    ]
