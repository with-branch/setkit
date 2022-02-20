from typing import Callable
import torch


def tokenize_bookends(tokenization_input: str, output_length: int, tokenizer: Callable):
    end_token_length = int(output_length / 2)
    start_token_length = output_length - end_token_length
    tokens = tokenizer(tokenization_input, padding="max_length", truncation=False)
    example_key = list(tokens.keys())[0]
    outputs = [{} for i in range(len(tokens[example_key]))]
    for key, batch in tokens.items():
        for idx, data in enumerate(batch):
            outputs[idx].update(
                {
                    key: get_bookends(
                        torch.tensor(data), start_token_length, end_token_length
                    )
                }
            )
    return outputs


def get_bookends(sequence, length_start, length_end):
    return torch.cat((sequence[:length_start], sequence[-length_end:]), dim=0)


