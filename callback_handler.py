import rossman_tabular as rt
import torch
from import_rossman_data import RossmanDataset
import coloredlogs
import logging


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
        pass

    def after_optim(self):
        self.learner.optim.zero_grad()


class GPUHandlingCallBacks(CallBack):
    _order = 10

    def model_set_up(self):
        # transfer everything to the device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.learner.model.to(self.device)

    def before_forward(self):
        self.cat = self.cat.to(self.device)
        self.cont = self.cont.to(self.device)

    def after_forward(self):
        del self.cat
        del self.cont
        self.yb = self.yb[2][:, 0].to(self.device)

    def after_losses(self):
        del self.yb


class TensorBoardLogCallBack(CallBack):
    _order = 4
    # TODO - record best validation parameters
    # TODO - record hyperparams
    # TODO - record train and validation errors

    # For recording best validation error in tensorboard
    best_validation_error = None

    def model_set_upTODO(self):
        # hyperparameter logging.
        self.hyperparameters = {}
        self.hyperparameters[
            "cosine_annealing_period"
        ] = self.learner.cosine_annealing_period
        self.hyperparameters["lr"] = self.learner.lr
        self.hyperparameters["hparam/current_epoch"] = 0
        self.hyperparameters["dropout"] = self.dropout


class Runner:
    def __init__(self, learner, cbs=[]):
        self.cbs = sorted(cbs, key=lambda k: k._order)
        self.init_cbs()
        self.epoch = 0
        self.learner = learner
        if self("model_set_up"):
            return

    def init_cbs(self):
        for cb in self.cbs:
            cb.set_runner(self)

    def __call__(self, name):
        for cb in self.cbs:
            current_func = getattr(cb, name, None)
            if current_func and current_func():
                return True

    def do_batch(self, cat, cont, yb):
        self.cat = cat
        self.cont = cont
        self.yb = yb
        if self("before_forward"):
            return

        predictions = self.learn.model.forward(cat, cont)
        if self("after_forward"):
            return

        batch_loss = self.learn.loss(predictions[:, 0], yb)
        if self("after_losses"):
            return batch_loss

        batch_loss.backward()
        if self("after_losses"):
            return batch_loss

        self.learner.optim.step()
        if self("after_optim"):
            return batch_loss

        return batch_loss

    def do_all_batches(self, dataloader):
        for cat, cont, yb in dataloader:
            self.do_batch(cat, cont, yb)

    def fit(self, epochs):
        if self("before_train"):
            return
        for epoch in range(epochs):
            if self("before_epoch"):
                return
            self.do_all_batches(self.learner.train_dataloader)
            if not self("after_train_epoch"):
                return
            self.do_all_batches(self.learner.valid_dataloader)
            if not self("after_validation"):
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
        10000,
        [100, 10],
        0.4,
        [0.9, 0.99],
    )
    cb1 = LRSchedulerCallBack()
    cb2 = TestValidSwitcherCallBack()
    cb3 = GPUHandlingCallBacks()
    cb4 = TensorBoardLogCallBack()
    example_runner = Runner(rossman_learner, [cb1, cb2, cb3, cb4])
    print("sdf")
