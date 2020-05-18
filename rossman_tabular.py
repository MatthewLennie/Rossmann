"""
This file contains the main implementation for the Rossman Kaggle Challenge.
Running as a script runs a random hyperparameter search
"""
import torch
import import_rossman_data as rossman
from import_rossman_data import RossmanDataset
from typing import List
from torch.utils.tensorboard import SummaryWriter
import random
import coloredlogs
import logging
import numpy as np


class TabularRossmanModel(torch.nn.Module):
    """[model class for rossman model, tabular style nn]

    Args:
        torch ([torch.nn.Module]): [inheritance]
    """

    def __init__(
        self,
        embedding_sizes: List[int],
        embedding_depths: List[int],
        layer_sizes: List[int],
        dropout: float,
        writer: torch.utils.tensorboard.writer,
    ):
        """[Sets up the network. Has Categorical embeddings for
        categorical input and simple input for linear layers]

        Args:
            embedding_sizes (List[int]):
            [list of the cardinalities of the categorical variables]
            embeddings_depths (List[int]):
            [the dimension of each dimension]
            cont_vars_sizes (int): [length of the continuous variables.]
            layer_sizes (List[int]):
            [sizes of the linear layers i.e. [5,5,2,1] ->
            a linear layer with 5,5 -> 5,2 -> 2,1]
            dropout (float): percentage dropout
            writer (torch.utils.tensorboard.writer):
            the writer object to create the tensorboard dashboard
        """
        super(TabularRossmanModel, self).__init__()

        self.writer = writer

        # build embeddings for categories
        self.CategoricalEmbeddings = []
        for depth, i in zip(embedding_depths, embedding_sizes):
            self.CategoricalEmbeddings.append(torch.nn.Embedding(i, depth))

        # convert the list of embeddings to a ModuleList so that PyTorch finds
        # it as a paramater for backpropagation...
        self.CategoricalEmbeddings = torch.nn.ModuleList(
            self.CategoricalEmbeddings
        )
        self.EmbeddingDropout = torch.nn.Dropout(dropout)
        self.linear_layers = []
        # cont_vars_sizes + len(embedding_sizes) * self.embedding_depth,
        # build linear layers for continuous variables and cat embeddings
        for in_size, out_size in zip(layer_sizes[:-2], (layer_sizes[1:-1])):
            self.linear_layers.append(torch.nn.Linear(in_size, out_size))
            self.linear_layers.append(torch.nn.ReLU())
            self.linear_layers.append(torch.nn.BatchNorm1d(out_size))
            self.linear_layers.append(torch.nn.Dropout(p=dropout))

        # output layer
        self.linear_layers.append(
            torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        # internal counter
        self.batch = 0
        self.put_activations_into_tensorboard = False

    def forward(
        self, cat_data: torch.tensor, cont_data: torch.tensor
    ) -> torch.tensor:
        """[forward propagation, categorical and continuous data handled seperately]

        Args:
            cat_data (torch.tensor): [categorical variable inputs]
            cont_data (torch.tensor): [continuous variable inputs ]

        Returns:
            torch.tensor: [predictions in normalized form]
        """
        self.batch += 1

        # Get embedding for each Categorical variable.
        cat_outputs = [
            emb(cat_data[:, idx].long())
            for idx, emb in enumerate(self.CategoricalEmbeddings)
        ]
        cat_outputs = self.EmbeddingDropout(torch.cat(cat_outputs, 1))
        # inp ->   Dropout(BatchNorm1(ReLU(linear(inp))))
        x = torch.cat([cat_outputs, cont_data], 1)
        for layer_num, layer in enumerate(self.linear_layers):
            if self.put_activations_into_tensorboard:
                self.writer.add_histogram(
                    "activations/Layer_{}".format(layer_num), x
                )
            x = layer(x)

            # check for nans. Done this way to prevent mem leak
            # still pretty nasty on memory usage
            check = torch.isnan(x).sum() == 0
            assert check
            del check
        return x


class Learner:
    """[Learner class for the Rossmann tabular model]
    """

    def __init__(
        self,
        train_data_obj: RossmanDataset,
        valid_data_obj: RossmanDataset,
        cosine_annealing_period: int,
        lr: float,
        batch_size: int,
        layer_sizes: List[int],
        dropout: float,
        betas: tuple,
    ):
        """[sets up logging, torch device, optimizer and scheduler.]

        Args:
            train_data (torch.utils.data.DataLoader): [training data]
            valid_data (torch.utils.data.DataLoader): [validation data]
            layer_sizes (List[int]): [sizes of the hidden layers]
        Returns:
            [learner]: [learner object]
        """
        self.writer = SummaryWriter("runs/{}".format(random.randint(0, 1e9)))

        # data loaders
        # Don't do like this with a large dataset if you are going
        # to do hyperparameter search
        self.train_data_obj = train_data_obj
        self.valid_data_obj = valid_data_obj
        self.batch_size = batch_size
        self.load_data(train_data_obj, valid_data_obj, self.batch_size)

        self.build_model(layer_sizes, dropout)

    def build_model(self, layer_sizes, dropout):

        # get the cardinality of each categorical variable.
        embedding_sizes = self.get_embedding_sizes(self.train_data_obj)
        # Create the embedding depths based on a simple rule from jeremey
        embedding_depths = [
            min(600, round(1.6 * x ** 0.56)) for x in embedding_sizes
        ]

        # add the non-hidden layer sizes to the array containing the layer sizes
        layer_sizes.insert(0, len(rossman.cont_vars) + sum(embedding_depths))
        layer_sizes.append(2)

        # create model skeleton
        self.model = TabularRossmanModel(
            embedding_sizes,
            embedding_depths,
            layer_sizes,
            dropout,
            self.writer,
        )

        return None

    def load_data(self, train_data_obj, valid_data_obj, batch_size):
        # create data loaders from datasets
        # TODO: test if ASYNC loading is quicker
        self.train_data = torch.utils.data.DataLoader(
            train_data_obj, batch_size=batch_size, pin_memory=True
        )
        self.valid_data = torch.utils.data.DataLoader(
            valid_data_obj, batch_size=batch_size, pin_memory=True
        )

    def initialize_optimizer(self):
        """[creates a clean optimizer and scheduler.
        Can by improved by providing resetting functionality, works for now]
        """
        # For recording best validation error in tensorboard
        self.best_validation_error = None

        # optimizer
        self.loss = torch.nn.MSELoss()
        self.cosine_annealing_period = cosine_annealing_period
        self.lr = lr
        self.betas = tuple(betas)  # for adam.
        self.initialize_optimizer()
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=self.betas
        )
        # self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optim, T_max=self.cosine_annealing_period
        # )

    # def exp_rmspe(
    #     self, pred: torch.tensor, targ: torch.tensor, log: bool = False
    # ) -> torch.tensor:
    #     """[exponential root mean squared percentage error taken from FASTAI code]

    #     Args:
    #         pred (torch.tensor): [predictions]
    #         targ (torch.tensor): [target values]
    #         log (bool, optional): [optional logging flag for tensorboard].
    #          Defaults to False.

    #     Returns:
    #         torch.tensor: [description]
    #     """

    #     pred, targ = torch.exp(pred), torch.exp(targ)
    #     pct_var = (targ - pred) / targ
    #     assert torch.isnan(pct_var).sum() == 0
    #     if log:
    #         self.writer.add_histogram("Losses_percentages", pct_var)
    #     return torch.sqrt((pct_var ** 2).mean())

    # def training_step(
    #     self, input_data: List[torch.tensor], log_grads: bool = False
    # ) -> torch.tensor:
    #     """[summary]

    #     Args:
    #         input_data (List[torch.tensor]): [List containning the categorical
    #         and continuous variables]
    #         log_grads (bool, optional): [whether to log the weights]. Defaults to False.

    #     Returns:
    #         torch.tensor: [returns loss from the batch]
    #     """
    #     cat = input_data[0].to(self.device)
    #     cont = input_data[1].to(self.device)
    #     predictions = self.model.forward(cat, cont)
    #     targ = input_data[2][:, 0].to(self.device)
    #     # TODO include both terms in loss
    #     # batch_loss = self.exp_rmspe(predictions[:, 0], targ)
    #     batch_loss = self.loss(predictions[:, 0], targ)
    #     batch_loss.backward()
    #     if log_grads:
    #         self.dump_model_parameters_to_log()
    #     self.optim.step()
    #     self.schedule.step()
    #     self.optim.zero_grad()
    #     return batch_loss

    # def dump_model_parameters_to_log(self):
    #     """[throws each of the model parameters into the log]
    #     """
    #     for param in self.model.named_parameters():

    #         self.writer.add_histogram(param[0], param[1])
    #         try:
    #             self.writer.add_histogram(
    #                 "Gradient_of_{}".format(param[0]), param[1].grad
    #             )
    #         except NotImplementedError:
    #             logger.debug("Missing Gradient for {}".format(param[0]))

    # def training_loop(self, epochs: int):
    #     """[runs learner over batchs and epochs.]

    #     Args:
    #         epochs (int): [number of epochs]

    #     Returns:
    #         [None]: [None]
    #     """
    #     for current_epoch in range(epochs):
    #         log_grads = True
    #         for batch in self.train_data:
    #             training_batch_loss = self.training_step(batch, log_grads)
    #             log_grads = False

    #         # perform tensorboard logging.
    #         self.writer.add_scalar(
    #             "Training_Loss", training_batch_loss, current_epoch
    #         )
    #         self.writer.add_scalar(
    #             "learning_rate",
    #             self.optim.param_groups[0]["lr"],
    #             current_epoch,
    #         )
    #         self.validation_set(current_epoch)
    #         if (
    #             self.schedule._step_count * 2 % self.cosine_annealing_period
    #             == 0
    #         ):
    #             self.initialize_optimizer()

    #         logger.debug("epoch: {}".format(current_epoch))
    #     # do logging of results.
    #     self.dump_hyperparameters_to_log(current_epoch)
    #     return None

    # def dump_hyperparameters_to_log(self, current_epoch):
    #     self.hyperparameters["hparam/current_epoch"] = current_epoch
    #     self.hyperparameters["hparam/best_epoch"] = self.best_epoch
    #     self.hyperparameters["hparam/batch_size"] = self.batch_size
    #     self.writer.add_hparams(
    #         hparam_dict=self.hyperparameters,
    #         metric_dict={
    #             "hparam/validation_error": self.best_validation_error
    #         },
    #     )
    #     return None

    # def validation_set(self, current_epoch: int):
    #     """[runs forward over validation set with extra logging]

    #     Args:
    #         current_epoch (int): [current epoch for logging purposes]

    #     Returns:
    #         [None]: [None]
    #     """
    #     losses = []
    #     self.model.eval()
    #     for batch in self.valid_data:
    #         predictions = self.model.forward(
    #             batch[0].to(self.device), batch[1].to(self.device)
    #         )

    #         # batch_loss = self.exp_rmspe(
    #         #     predictions[:, 0], batch[2][:, 0].to(self.device), log=True
    #         # )

    #         batch_loss = self.loss(
    #             predictions[:, 0], batch[2][:, 0].to(self.device)
    #         )
    #         losses.append(batch_loss)
    #     current_validation_error = torch.stack(losses).mean()

    #     if (
    #         not self.best_validation_error
    #         or self.best_validation_error > current_validation_error
    #     ):
    #         self.best_validation_error = current_validation_error
    #         self.best_epoch = current_epoch

    #     self.writer.add_scalar(
    #         "Validation_Loss", torch.stack(losses).mean(), current_epoch
    #     )
    #     self.model.train()
    #     return None

    def get_embedding_sizes(self, train_data_obj: RossmanDataset) -> List[int]:
        """[Small helper function just to find the cardinality
        of each categorical variable]

        Args:
            train_data_obj (RossmanDataset): [training data object]

        Returns:
            List[int]: [results list of cardinalities]
        """
        embedding_sizes = []

        # Get embedding Layer sizes based on unique categories.
        # Potential bug if rare classes don't appear in the training set.
        for key in rossman.cat_vars:
            embedding_sizes.append(
                max([len(train_data_obj.data[key].unique()), 2])
            )
        return embedding_sizes


if __name__ == "__main__":
    # sets up a hyperparameter search of the model

    # Logging settings
    coloredlogs.install(level="DEBUG")
    logger = logging.getLogger(__name__)

    # Open data objects
    train_data_obj = RossmanDataset.from_pickle("./data/train_data.pkl")
    valid_data_obj = RossmanDataset.from_pickle("./data/valid_data.pkl")

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
        rossman_learner = Learner(
            train_data_obj, valid_data_obj, c1, c2, c3, c4, c5, c6
        )
        Hparam_string = "Cosine:{}, lr: {}, batchsize: {}, layers:{}, dropout:{}, momentum: {}".format(
            c1, c2, c3, c4, c5, c6
        )

        # print out the Hyperparameters for logging purposes.
        rossman_learner.writer.add_text("config".format(trial), Hparam_string)
        print(Hparam_string)
        rossman_learner.model.put_activations_into_tensorboard = False

        # rossman_learner.training_loop(450)
        try:
            rossman_learner.training_loop(600)
        except AssertionError:
            print("got NAN activations")
            rossman_learner.writer.add_text("failure message", Hparam_string)

        # delete the model once done with it or watch the GPU ram disappear.
        del rossman_learner
