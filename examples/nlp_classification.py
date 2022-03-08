from asyncio import tasks
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

n_epochs = 5

dataset = (
    ExampleNLP()
)  # Automatically downloads the data if necessary (places in the rootflow package)

classification_head = ClassificationHead(input_size=768, num_classes=2, dropout=0.1)
model = Transformer(
    "roberta-base", head=classification_head, num_training_steps=n_epochs * len(dataset)
)
# Or alternatively
model = AutoTransformer(
    "roberta-base", tasks=dataset.tasks(), num_training_steps=n_epochs * len(dataset)
)
print(type(model))

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
    epochs=n_epochs,
)
classification_trainer.train()
