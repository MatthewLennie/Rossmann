import random
from torch.utils.tensorboard import SummaryWriter
import rossman_tabular as rt
import torch
from import_rossman_data import RossmanDataset
import coloredlogs
import logging
import math


class CallBack:
    _order = 0

    def set_runner(self, run):
        self.run = run

    # gives access to the scope of the runner.
    def __getattr__(self, k):
        return getattr(self.run, k)


class LRSchedulerCallBack(CallBack):
    _order = 1
    # todo LR

    def model_set_up(self):
        pass


class TestValidSwitcherCallBack(CallBack):
    _order = 2

    def after_train_epoch(self):
        # TODO
        self.run.learner.model.eval()
        self.learner.optim.zero_grad()

    def after_validation(self):
        self.run.learner.model.train()


class GPUHandlingCallBacks(CallBack):
    _order = 1

    def model_set_up(self):
        # transfer everything to the device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.learner.model = self.learner.model.to(self.device)
        # print("send to GPU")

    def before_forward(self):
        self.run.cat = self.cat.to(self.device)
        self.run.cont = self.cont.to(self.device)
        # print("cat on:{}".format(self.cat.device))

    def after_forward(self):
        del self.run.cat
        del self.run.cont
        self.run.yb = self.yb[:, 0].to(self.device)

    def after_losses(self):
        del self.run.yb


class OptimizerCallBack(CallBack):
    def cosine_annealing(self, pos):
        pos = pos % 0.25
        peak = 0.1
        start = 0.0001
        end = 0.01

        if pos <= peak:
            return (
                start
                + (1 + math.cos(math.pi * (1 - pos * 4))) * (end - start) / 2
            )
        else:
            end2 = 0.0001
            return (
                end
                + (1 + math.cos(math.pi * (1 - 4 * pos + peak * 4)))
                * (end2 - end)
                / 2
            )

    def model_set_up(self):
        # optimizer
        self.run.learner.loss = torch.nn.MSELoss()
        print("loss set")
        self.cosine_annealing_period = 10
        self.lr = 0.005
        self.betas = tuple([0.9, 0.99])  # for adam.
        self.run.learner.optim = torch.optim.Adam(
            self.run.learner.model.parameters(), lr=self.lr, betas=self.betas
        )
        # self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optim, T_max=self.cosine_annealing_period
        # )

    def before_forward(self):
        new_lr = self.cosine_annealing(self.epoch_fraction)
        for param_group in self.run.learner.optim.param_groups:
            param_group["lr"] = new_lr


class TensorBoardLogCallBack(CallBack):
    _order = 4
    # TODO - record best validation parameters
    # TODO - record hyperparams
    # TODO - record train and validation errors

    # For recording best validation error in tensorboard
    best_validation_error = None

    writer = SummaryWriter("runs/{}".format(random.randint(0, 1e9)))

    def model_set_upTODO(self):
        # hyperparameter logging.
        self.hyperparameters = {}
        self.hyperparameters[
            "cosine_annealing_period"
        ] = self.learner.cosine_annealing_period
        self.hyperparameters["lr"] = self.learner.lr
        self.hyperparameters["hparam/current_epoch"] = 0
        self.hyperparameters["dropout"] = self.dropout

    def after_train_epoch(self):
        self.writer.add_scalar("Training_Loss", self.train_loss, self.epoch)
        self.writer.add_scalar(
            "lr", self.learner.optim.param_groups[0]["lr"], self.epoch
        )

    def after_validation(self):
        self.writer.add_scalar("Validation_Loss", self.valid_loss, self.epoch)


class Runner:
    def __init__(self, learner: rt.Learner, cbs: CallBack = []):
        self.cbs = sorted(cbs, key=lambda k: k._order)
        self.init_cbs()
        self.epoch = 0
        self.learner = learner

        if self("model_set_up"):
            return

    def init_cbs(self):
        for cb in self.cbs:
            cb.set_runner(self)

    def __call__(self, name: str):
        for cb in self.cbs:
            current_func = getattr(cb, name, None)
            if current_func and current_func():
                return True

    def do_batch(
        self, cat: torch.tensor, cont: torch.tensor, yb: torch.tensor
    ):
        self.cat = cat
        self.cont = cont
        del cat, cont
        self.yb = yb
        if self("before_forward"):
            return

        predictions = self.learner.model.forward(self.cat, self.cont)
        if self("after_forward"):
            return

        batch_loss = self.learner.loss(predictions[:, 0], self.yb)
        if self("after_losses"):
            return batch_loss

        batch_loss.backward()
        if self("after_backward"):
            return batch_loss

        self.learner.optim.step()
        if self("after_optim"):
            return batch_loss

        return batch_loss

    def do_all_batches(self, dataloader: torch.utils.data.dataloader):
        epoch_loss = 0
        data_sizes = 0
        for cat, cont, yb in dataloader:
            loss = self.do_batch(cat, cont, yb)
            epoch_loss += loss * yb.shape[0]
            data_sizes += yb.shape[0]
        return epoch_loss / data_sizes

    def fit(self, epochs: int):
        if self("before_train"):
            return
        for epoch in range(epochs):
            self.epoch = epoch
            self.epoch_fraction = float(epoch) / float(epochs)
            print("epoch: {}".format(epoch))
            if self("before_epoch"):
                return
            self.train_loss = self.do_all_batches(self.learner.train_data)
            if self("after_train_epoch"):
                return
            self.valid_loss = self.do_all_batches(self.learner.valid_data)
            if self("after_validation"):
                return


if __name__ == "__main__":
    # example_runner.learn()
    # print(example_runner.epoch)
    print("blah")
    # Logging settings
    coloredlogs.install(level="DEBUG")
    logger = logging.getLogger(__name__)

    # Open data objects
    train_data_obj = RossmanDataset.from_pickle("./data/train_data.pkl")
    valid_data_obj = RossmanDataset.from_pickle("./data/valid_data.pkl")

    # build and train model
    rossman_learner = rt.Learner(
        train_data_obj,
        valid_data_obj,
        10,
        0.01,
        150000,
        [100, 10],
        0.4,
        [0.9, 0.99],
    )
    cb1 = LRSchedulerCallBack()
    cb2 = TestValidSwitcherCallBack()
    cb3 = GPUHandlingCallBacks()
    cb4 = TensorBoardLogCallBack()
    cb5 = OptimizerCallBack()
    example_runner = Runner(rossman_learner, [cb1, cb2, cb3, cb4, cb5])
    example_runner.fit(100)
