from rootflow.datasets import ExampleDataset
from rootflow.models.nlp.transformers import Tokenizer, Classifier
from rootflow.tasks.nlp import FineTuneTrainer, ClassificationTrainer
from rootflow.training.metrics import F1, Accuracy

dataset = ExampleDataset() # Automatically downloads the data if necessary (places in the rootflow package)

tokenizer = Tokenizer.from_pretrained("roberta-base", mode="split")
dataset.map(tokenizer, batch_size=256)
train_set, validation_set = dataset.split()

model = Classifier.from_pretrained("roberta-base", num_classes=dataset.num_classes())

fine_tune_trainer = FineTuneTrainer(
    "./results", model, training_dataset=train_set, metrics=[F1, Accuracy]
)
fine_tune_trainer.train()

classification_trainer = ClassificationTrainer(
    "./results",
    model,
    training_dataset=train_set,
    validation_dataset=validation_set,
    metrics=[F1, Accuracy],
)
classification_trainer.train()
