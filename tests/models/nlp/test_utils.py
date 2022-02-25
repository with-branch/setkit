import torch
from rootflow.models.nlp.utils import tokenize_bookends


def test_tokenize_bookends_single():
    single_string = "This is a very nice string I have here"
    tokenizer_function = lambda input_string: [
        character for character in input_string.encode("utf-8")
    ]
    # fmt: off
    assert tokenize_bookends(single_string, 50, tokenizer_function) == torch.tensor(
        [ 84., 104., 105., 115.,  32., 105., 115.,  32.,  97.,  32., 118., 101.,
        114., 121.,  32., 110., 105.,  99., 101.,  32., 115., 116., 114., 105.,
        110., 103.,  32.,  73.,  32., 104.,  97., 118., 101.,  32., 104., 101.,
        114., 101.]
    )
    # fmt: on
