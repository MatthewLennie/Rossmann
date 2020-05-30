import callback_handler
import torch
import import_rossman_data
import coloredlogs
from import_rossman_data import RossmanDataset
from callback_handler import Runner
from callback_handler import (
    LRSchedulerCallBack,
    TensorBoardLogCallBack,
    TestValidSwitcherCallBack,
    GPUHandlingCallBacks,
    OptimizerCallBack,
)
import rossman_tabular as rt
import logging
import numpy as np

# import torch.cuda.profiler as profiler
# import pyprof
import torch.cuda.profiler as profiler


def setup_run(
    train_data_obj,
    valid_data_obj,
    cosine_annealing_period,
    lr,
    batch_size,
    layer_sizes,
    dropout,
    betas,
):
    # Logging settings
    coloredlogs.install(level="DEBUG")
    logger = logging.getLogger(__name__)

    # build and train model
    rossman_learner = rt.Learner(
        train_data_obj,
        valid_data_obj,
        cosine_annealing_period,
        lr,
        batch_size,
        layer_sizes,
        dropout,
        betas,
    )
    cb1 = LRSchedulerCallBack()
    cb2 = TestValidSwitcherCallBack()
    cb3 = GPUHandlingCallBacks()
    cb4 = TensorBoardLogCallBack()
    cb5 = OptimizerCallBack()
    example_runner = Runner(rossman_learner, [cb1, cb2, cb3, cb4, cb5])
    example_runner.fit(60)


# Open data objects
train_data_obj: RossmanDataset = RossmanDataset.from_pickle(
    "./data/train_data.pkl"
)
valid_data_obj: RossmanDataset = RossmanDataset.from_pickle(
    "./data/valid_data.pkl"
)


# import_rossman_data.main()
# with torch.autograd.profiler.emit_nvtx():
# callback_handler.main()  # sets up a hyperparameter search of the model

# Logging settings
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

# Hyperparameter Search Range
batch_size = [100000]  # Maxes out the ram
cosine_annealing_period = [10, 2]
layer_sizes = [
    [240, 1000, 50],  # <- Jeremy used this one in the course.
    [240, 1000, 250, 50],
    [240, 150, 80, 40, 10],
    [60, 60, 40, 30, 20, 10],  # <-This one ended up worked well
]
lr = [0.001, 0.0005]
dropout = [0.1, 0.3, 0.4]
betas = [
    [0.9, 0.999],  # The normal default
    [0.99, 0.9999],
    [0.999, 0.99999],
    [0.8, 0.99],
]

# build and train model
for trial in range(60):

    # Randomly choose a set of hyperparameters
    c1 = int(np.random.choice(cosine_annealing_period))
    c2 = np.random.choice(lr)
    c3 = int(np.random.choice(batch_size))
    c4 = np.random.choice(layer_sizes).copy()
    c5 = np.random.choice(dropout)
    c6 = betas[np.random.randint(0, len(betas))]

    setup_run(train_data_obj, valid_data_obj, c1, c2, c3, c4, c5, c6)
    Hparam_string = "Cosine:{}, lr: {}, batchsize: {}, layers:{}, dropout:{}, momentum: {}".format(
        c1, c2, c3, c4, c5, c6
    )

    # print out the Hyperparameters for logging purposes.
    # rossman_learner.writer.add_text("config".format(trial), Hparam_string)
    print(Hparam_string)
    rossman_learner.model.put_activations_into_tensorboard = False

    # rossman_learner.training_loop(450)
    # try:
    #     rossman_learner.training_loop(600)
    # except AssertionError:
    #     print("got NAN activations")
    #     rossman_learner.writer.add_text("failure message", Hparam_string)

    # # delete the model once done with it or watch the GPU ram disappear.
    # del rossman_learner
