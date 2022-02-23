# rootflow API
The following is a list of considerations, desirable traits, and hypothetical situations we would like `rootflow` to be able to support.

First I would like to explicitly enumerate our ethos. In the order of importance, they are as follows:
- All components of `rootflow` should be compatible with native `pytorch`. (You could choose to use a single `rootflow` component in `pytorch` code, or visa versa)
- Easy to understand and pick up for an experienced `pytorch` developer.
- Performance by default. No extra configuration is necessary to approximate SOTA
- At least average configurability. While we might always be able to support extreme flexibility, all things that you might change from experiment to experiment should be easily configurable.

We don't expect that `rootflow` will comprehensively cover all things that a data scientist might need to do at branch. We do, however, expect that `rootflow` will have at least one useful component for every experiment (hopefully more!), and that component will be easy to use with other `pytorch` code, and apply known best practices.

### Multihead, Multidataset, and Multitask
Current plan is to have models dynamically determine a multi-type state from the dataset `num_classes` function.
If it returns a dictionary, we treat the names as our tasks and create a head for each.

### Custom Optimizer or LR Scheduling.
We will probably allow arbitrary pytorch optimizers to be passed to the trainer.
This implies the same for the LR Scheduler, since it usually takes an optimizer parameter.
These will most likely be separate parameters.

One consideration is the fact that some LR Schedulers update by batch, and others by epoch (Looking at you `StepLR`). I don't know how much we care about supporting this nuance.

### Non-rootflow datasets
Not sure if this should be allowed or not. The only `RootflowDataset` feature we have considered depending on is the `num_classes`, but we can certainly just keep that at the top level API, so that it isn't necessary, the practitioner could provide it.

### Custom DataLoaders
Position will probably be the same as the `Optimizer and LR Scheduling` section.

### LR, Batch Size and Epoch defaults
Other than epochs, these can be found dynamically. Use the well known Learning Rate Range Test from Leslie N Smith for the learning rate, and select batch size such that it is the largest multiple of 2 which does not run out of memory on one of our devices. (Note: `pytorch-lightning` has already implemented something like this.)

As far as the number of epochs. This is a relationship between the dataset, the trainer, and the model. I suggest we should default to infinite epochs (with early stopping)

### Model Arguments
Is passing additional, model specific arguments during training necessary at any point?

I believe not, since the models are constructed separately, and the practitioner is given ample opportunity to set them at this point.

### Rootflow Model Base Class
Clearly the flexibility of allowing pure pytorch models is desirable. These also already have the `eval` and `train` functions, which may be all that is necessary for the trainer to interact with the model.

At this point, I am against having a rootflow model base class, since this will make the package far too inconvenient to use.