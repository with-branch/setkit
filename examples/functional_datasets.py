from rootflow.datasets.examples import ExampleNLP, ExampleTabular

# Creating a new instance of a rootflow dataset will
# automatically download that dataset if you don't have
# the data localy
dataset = ExampleNLP()

# If you don't want to do this, simply disable it
dataset = ExampleNLP(download=False)

# Or if you already have the data in a different location
# just load it from there
dataset = ExampleNLP(root="rootflow/datasets/data/ExampleNLP/data", download=False)

dataset.describe()
