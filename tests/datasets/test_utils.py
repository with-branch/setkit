from rootflow.datasets.utils import *


def test_batch():
    my_list = [i for i in range(100)]
    for _batch in batch(my_list, batch_size=5):
        assert _batch == [0, 1, 2, 3, 4]
        break
    for _batch in batch(my_list, batch_size=9):
        assert (len(_batch) == 9) or (len(_batch) == 1)
    for _batch in batch(my_list, batch_size=7):
        assert (len(_batch) == 7) or (len(_batch) == 2)


def test_batch_enumerate():
    my_list = [i for i in range(100)]
    for _slice, _batch in batch_enumerate(my_list, batch_size=5):
        assert _slice == slice(0, 5)
        assert _batch == [0, 1, 2, 3, 4]
        break
    for idx, (_slice, _batch) in enumerate(batch_enumerate(my_list, batch_size=9)):
        assert _slice.start == idx * 9
        assert _slice.stop == min((idx + 1) * 9, 100)
        assert (len(_batch) == 9) or (len(_batch) == 1)
    for idx, (_slice, _batch) in enumerate(batch_enumerate(my_list, batch_size=7)):
        assert _slice.start == idx * 7
        assert _slice.stop == min((idx + 1) * 7, 100)
        assert (len(_batch) == 7) or (len(_batch) == 2)


def test_map_functions():
    functions = [
        lambda x: x + 1,
        lambda x: x**2,
        lambda x: str(x),
        lambda x: f"1{x}",
        lambda x: int(x),
        lambda x: x / 8,
    ]
    assert map_functions(5, functions) == 17.0
    assert map_functions(13, functions) == 149.5

    functions = [
        lambda x: f"welcome {x}",
        lambda x: x.replace("e", "..8"),
        lambda x: x.replace(" ", x),
        lambda x: x.split(" ")[1],
        lambda x: f"{x}.",
    ]
    assert map_functions("some input", functions) == "som..8."
    assert map_functions("4tun3", functions) == "4tun34tun3."


def test_get_unique_ordered():
    raise NotImplementedError


def test_get_unique_unordered():
    raise NotImplementedError
