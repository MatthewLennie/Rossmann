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
from typing import List, Tuple
import torch.cuda.profiler as profiler


def setup_run(
    train_data_obj: RossmanDataset,
    valid_data_obj: RossmanDataset,
    cosine_annealing_period: int,
    lr: float,
    batch_size: int,
    layer_sizes: List[int],
    dropout: int,
    betas: Tuple[int],
):
    """[Runs a simple random hyperparameter search]

    Args:
        train_data_obj (RossmanDataset): [description]
        valid_data_obj (RossmanDataset): [description]
        cosine_annealing_period (int): [description]
        lr (float): [description]
        batch_size (int): [description]
        layer_sizes (List[int]): [description]
        dropout (int): [description]
        betas (List[int]): [description]
    """

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
    cb1: LRSchedulerCallBack = LRSchedulerCallBack()
    cb2: TestValidSwitcherCallBack = TestValidSwitcherCallBack()
    cb3: GPUHandlingCallBacks = GPUHandlingCallBacks()
    cb4: TensorBoardLogCallBack = TensorBoardLogCallBack()
    cb5: OptimizerCallBack = OptimizerCallBack(
        cosine_annealing_period, lr, betas
    )
    example_runner: Runner = Runner(rossman_learner, [cb1, cb2, cb3, cb4, cb5])
    example_runner.fit(600)
    del example_runner
    del cb1, cb2, cb3, cb4, cb5

    torch.cuda.empty_cache()
    print("ending run")


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
batch_size: List[int] = [250000]  # Maxes out the ram
cosine_annealing_period: List[int] = [10, 5]
layer_sizes: List[List[int]] = [
    # [240, 1000, 50],  # <- Jeremy used this one in the course.
    # [240, 1000, 250, 50],
    [240, 150, 80, 40, 10],
    [60, 60, 40, 30, 20, 10],  # <-This one ended up worked well
]
lr: List[float] = [0.001, 0.0005]
dropout: List[float] = [0.1, 0.3, 0.4]
betas: List[List[float]] = [
    [0.9, 0.999],  # The normal default
    [0.99, 0.9999],
    [0.999, 0.99999],
    [0.8, 0.99],
]

# build and train model
for trial in range(60):

    # Randomly choose a set of hyperparameters
    c1: int = int(np.random.choice(cosine_annealing_period))
    c2: float = np.random.choice(lr)
    c3: int = int(np.random.choice(batch_size))
    c4: List[int] = np.random.choice(layer_sizes).copy()
    c5: float = np.random.choice(dropout)
    c6: List[float] = betas[np.random.randint(0, len(betas))]

    Hparam_string = "Cosine:{}, lr: {}, batchsize: {}, layers:{}, dropout:{}, momentum: {}".format(
        c1, c2, c3, c4, c5, c6
    )

    # print out the Hyperparameters for logging purposes.
    # rossman_learner.writer.add_text("config".format(trial), Hparam_string)
    print(Hparam_string)
    setup_run(train_data_obj, valid_data_obj, c1, c2, c3, c4, c5, c6)
