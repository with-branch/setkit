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

dataset = (
    ExampleNLP()
)  # Automatically downloads the data if necessary (places in the rootflow package)

classification_head = ClassificationHead(input_size=768, num_classes=2, dropout=0.1)
model = Transformer("roberta-base", head=classification_head)
# Or alternatively
model = AutoTransformer("roberta-base", tasks=dataset.tasks())

tokenizer = Tokenizer("roberta-base", model.config.hidden_size, mode="split")
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
)
classification_trainer.train()
