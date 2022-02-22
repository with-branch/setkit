# branchml
A framework of models, datasets and other utilities for training branch ML models.

# Training
The branchml framwork will use the FastAI training API, all models and datasets should conform to this.
If possible, NLP models should conform to the huggingface Trainer API as well.

# Hyperparameter Tuning
We will use Optuna for the hyperparameter tuning. Optuna already has some integrations for FastAI

# Contributing
When contributing there are a couple things to keep in mind. First, write tests for your contribution, PRs will not be accepted without tests. Secondly, format your code using the black formatter before submitting a PR or commiting changes. To get the black formatter simply run the command
```
pip install black
```
in your rootflow development environment. If you are using VSCode, it is also recommended to set your python formatter. This can be done by navigating to `File/Preferences/Settings`, and then searching `Python Formatting Provider`. Set this to `black`. Consider also enabling the `Format On Save` setting.

# TODO
Update package dependencies