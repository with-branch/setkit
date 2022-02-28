# rootflow
A framework of models, datasets and other utilities for training branch ML models.

# Training
Our plan for training provider is currently in flux. We are looking into `pytorch-lightning` and also considering writing our trainers in native `pytorch`

# Hyperparameter Tuning
We will use Optuna for the hyperparameter tuning.

# Contributing
When contributing there are a couple things to keep in mind. Pull requests and contributions must adhere to the following set of criteria:

- Every function and class has an associated pytest test in the `tests` subfolder.
- Each function and class has a docstring. (This requirement is somewhat looser for private functions and classes or simple, self-explanatory ones. See the appropriate section for more details)
- Code is formatted using the python `black` formatter

As an additional rule of thumb, avoid changing any interfaces or APIs, wherever possible.

### Tests
The organization of the `tests` subfolder should mirror that of the package, expanding a single python file into a directory is acceptable, if the number of tests is large and this would help organization.

### Documentation
Code should be documented according to [PEP 257](https://www.python.org/dev/peps/pep-0257/). Additionaly, we will follow the Google python [docstring conventions](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
Since there is sometimes confusion as to whether an `__init__` docstring should be in the class level documentation or the method level we will stick to the method level. This maintains consistency, and most python type hinting is good about getting the user all of the information they need and handling the `__init__` documentation correctly.
Type hints are allowed and encouraged within the docstrings; In addition, use `:class:` and `:method:` annotations when appropriate.

As demonstration, here is the documentation for the [`rootflow.datasets.examples`](rootflow/datasets/examples.py) `ExampleTabular` dataset.
```python
class ExampleTabular(RootflowDataset):
    """An example rootflow dataset for tabular data.

    Inherits from :class:`RootflowDataset`.
    The data is generated with 4 features calculated off of the targets.
    The targets are the integers range(1000)
    """
...
```
And here is the documentation for the [`rootflow.datasets.base.utils`](rootflow/datasets/base/utils.py) `batch_enumerate` function.
```python
def batch_enumerate(iterable: Iterable, batch_size: int = 1) -> Tuple[slice, list]:
    """Enumerates in batches.

    Enumerates an iterable in consistent length batches, yielding the slice and batch
    for each. Note that the last batch will have size of `len(iterable) % batch_size`
    instead of `batch_size`.

    Args:
        iterable (Iterable): Some iterable which we would like to split into batches.
        batch_size (:obj:`int`, optional): The size of each batch, except the last.

    Yields:
        Tuple[slice, list]: A tuple containing, respectively, the slice corresponding
            to the batch's location in the iterable and a list containing the batch.
    """
...
```

### Formatting
Format your code using the black formatter. This ensures that the codebase is as consistent as possble, and easier to read. To get the black formatter simply run the command
```
pip install black
```
in your rootflow development environment. If you are using VSCode, it is also recommended to set your python formatter. This can be done by navigating to `File/Preferences/Settings`, and then searching `Python Formatting Provider`. Set this to `black`. Consider also enabling the `Format On Save` setting.