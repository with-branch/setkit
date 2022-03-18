from rootflow.datasets.examples import ExampleNLP
from rootflow.models.nlp.transformers import (
    Tokenizer,
    Transformer,
    AutoTransformer,
    ClassificationHead,
)
from rootflow.training.nlp import FineTuneTrainer
from rootflow.training import SupervisedTrainer
from rootflow.training.metrics import F1, Accuracy

n_epochs = 20

dataset = ExampleNLP()

model = AutoTransformer(
    "roberta-base", tasks=dataset.tasks(), num_training_steps=n_epochs * len(dataset)
)
tokenizer = Tokenizer(
    "roberta-base", model.config.max_position_embeddings, mode="split"
)
dataset = dataset.map(tokenizer, batch_size=256)
train_set, validation_set = dataset.split()
dataset.summary()

# fine_tune_trainer = FineTuneTrainer(
#     "./results",
#     model,
#     training_dataset=train_set,
#     validataion_dataset=validation_set,
#     metrics=[F1, Accuracy],
# )
# fine_tune_trainer.train()

classification_trainer = SupervisedTrainer(
    "./results",
    model,
    training_dataset=train_set,
    validation_dataset=validation_set,
    metrics=[F1, Accuracy],
    max_epochs=n_epochs,
    gpus=1,
)
classification_trainer.train()
