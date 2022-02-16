# branchml
A framework of models, datasets and other utilities for training branch ML models.

# Training
The branchml framwork will use the FastAI training API, all models and datasets should conform to this.
If possible, NLP models should conform to the huggingface Trainer API as well.

# Hyperparameter Tuning
We will use Optuna for the hyperparameter tuning. Optuna already has some integrations for FastAI

# Contributing
Use black for code formatting
Avoid excessive interior mutability