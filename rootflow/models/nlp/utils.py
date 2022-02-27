from typing import Callable, Mapping, Union, List
import torch

# Since we want to use batch padding to speed up the transformers later,
# this function should not include extra padding
def tokenize_bookends(
    tokenization_input: Union[str, List[str]],
    max_token_length: int,
    tokenizer: Callable,
):
    end_token_length = int(max_token_length / 2)
    start_token_length = max_token_length - end_token_length
    tokens = tokenizer(tokenization_input)

    if isinstance(tokenization_input, (tuple, list)):
        if isinstance(tokens[0], Mapping):
            return [
                {
                    key: get_bookends(data, start_token_length, end_token_length)
                    for key, data in tokenized_item.items()
                }
                for tokenized_item in tokens
            ]
        else:
            return [get_bookends(tokenized_item) for tokenized_item in tokens]
    elif isinstance(tokenization_input, str):
        if isinstance(tokens, Mapping):
            return {
                key: get_bookends(data, start_token_length, end_token_length)
                for key, data in tokens.items()
            }
        else:
            return get_bookends(tokens, start_token_length, end_token_length)


def get_bookends(sequence, length_start, length_end):
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.Tensor(sequence)
    print(sequence)
    return torch.cat((sequence[:length_start], sequence[-length_end:]), dim=0)
