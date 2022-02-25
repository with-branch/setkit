from rootflow.datasets import ExampleNLP
from rootflow.models.nlp.transformers import Tokenizer, ClassificationTransformer
from rootflow.tasks.nlp import FineTuneTrainer
from rootflow.tasks import SupervisedTrainer
from rootflow.training.metrics import F1, Accuracy

dataset = ExampleNLP()

model = ClassificationTransformer("roberta-base", task_shapes=dataset.task_shapes())
tokenizer = Tokenizer("roberta-base", model.config.max_token_length, mode="split")

dataset = dataset.map(tokenizer, batch_size=256)
train_set, validation_set = dataset.split()

fine_tune_trainer = FineTuneTrainer(
    "./results",
    model,
    training_dataset=train_set,
    metrics=[F1, Accuracy],
)
fine_tune_trainer.train()

classification_trainer = SupervisedTrainer(
    "./results",
    model,
    training_dataset=train_set,
    validation_dataset=validation_set,
    metrics=[F1, Accuracy],
)
classification_trainer.train()
