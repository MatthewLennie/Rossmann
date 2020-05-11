import torch
import import_rossman_data as rossman
from import_rossman_data import RossmanDataset
from typing import List
from torch.utils.tensorboard import SummaryWriter
import random
import coloredlogs
import logging
import numpy as np


class tabular_rossman_model(torch.nn.Module):
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
            [sizes of the linear layers i.e. [5,5,2,1] -> a linear layer with 5,5 -> 5,2 -> 2,1]
        """
        super(tabular_rossman_model, self).__init__()

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
        logger.debug("count: {}".format(self.batch))

        # Get embedding for each Categorical variable.
        cat_outputs = [
            emb(cat_data[:, idx].long())
            for idx, emb in enumerate(self.CategoricalEmbeddings)
        ]
        cat_outputs = self.EmbeddingDropout(torch.cat(cat_outputs, 1))

        # inp ->   Dropout(BatchNorm1(ReLU(linear(inp))))
        x = torch.cat([cat_outputs, cont_data], 1)
        for layer in self.linear_layers:
            x = layer(x)
        return x


class learner:
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

        # Create the embedding depths based on a simple rule from jeremey

        embedding_depths = [
            min(600, round(1.6 * x ** 0.56)) for x in embedding_sizes
        ]

        # add the non-hidden layer sizes
        layer_sizes.insert(0, len(rossman.cont_vars) + sum(embedding_depths))
        layer_sizes.append(2)

        # create model skeleton
        self.model = tabular_rossman_model(
            embedding_sizes, embedding_depths, layer_sizes, dropout
        )

        self.best_validation_error = None

        # optimizer
        self.cosine_annealing_period = cosine_annealing_period
        self.lr = lr
        self.initialize_optimizer()

        # hyperparameter logging.
        self.hyperparameters = {}
        self.hyperparameters[
            "cosine_annealing_period"
        ] = self.cosine_annealing_period
        self.hyperparameters["lr"] = self.lr
        self.hyperparameters["hparam/current_epoch"] = 0
        self.hyperparameters["dropout"] = dropout
        # self.hyperparameters["embedding_sizes"] = embedding_sizes
        # transfer everything to the device

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
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
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=self.cosine_annealing_period
        )

    def exp_rmspe(
        self, pred: torch.tensor, targ: torch.tensor, log: bool = False
    ) -> torch.tensor:
        """[exponential root mean squared percentage error taken from FASTAI code]

        Args:
            pred (torch.tensor): [predictions]
            targ (torch.tensor): [target values]
            log (bool, optional): [optional logging flag for tensorboard].
             Defaults to False.

        Returns:
            torch.tensor: [description]
        """

        pred, targ = torch.exp(pred), torch.exp(targ)
        pct_var = (targ - pred) / targ
        assert torch.isnan(pct_var).sum() == 0
        if log:
            self.writer.add_histogram("Losses_percentages", pct_var)
        return torch.sqrt((pct_var ** 2).mean())

    def training_step(
        self, input_data: List[torch.tensor], log_grads: bool = False
    ) -> torch.tensor:
        """[summary]

        Args:
            input_data (List[torch.tensor]): [description]
            log_grads (bool, optional): [description]. Defaults to False.

        Returns:
            torch.tensor: [description]
        """
        cat = input_data[0].to(self.device)
        cont = input_data[1].to(self.device)
        predictions = self.model.forward(cat, cont)
        # TODO include both terms in loss
        batch_loss = self.exp_rmspe(
            predictions[:, 0], input_data[2][:, 0].to(self.device)
        )
        batch_loss.backward()
        if log_grads:
            self.dump_model_parameters_to_log()
        self.optim.step()
        self.schedule.step()
        self.optim.zero_grad()
        return batch_loss

    def dump_model_parameters_to_log(self):
        """[throws each of the model parameters into the log]
        """
        for param in self.model.named_parameters():

            self.writer.add_histogram(param[0], param[1])
            try:
                self.writer.add_histogram(
                    "Gradient_of_{}".format(param[0]), param[1].grad
                )
            except NotImplementedError:
                logger.debug("Missing Gradient for {}".format(param[0]))

    def training_loop(self, epochs: int):
        """[runs learner over batchs and epochs.]

        Args:
            epochs (int): [number of epochs]

        Returns:
            [None]: [None]
        """
        for current_epoch in range(epochs):
            log_grads = True
            for batch in self.train_data:
                training_batch_loss = self.training_step(batch, log_grads)
                log_grads = False
            # perform tensorboard logging.
            self.writer.add_scalar(
                "Training_Loss", training_batch_loss, current_epoch
            )
            self.writer.add_scalar(
                "learning_rate",
                self.optim.param_groups[0]["lr"],
                current_epoch,
            )
            self.validation_set(current_epoch)
            if self.schedule._step_count % self.cosine_annealing_period == 0:
                self.initialize_optimizer()
        # do logging of results.
        self.dump_hyperparameters_to_log(current_epoch)
        return None

    def dump_hyperparameters_to_log(self, current_epoch):
        self.hyperparameters["hparam/current_epoch"] = current_epoch
        self.hyperparameters["hparam/best_epoch"] = self.best_epoch
        self.hyperparameters["hparam/batch_size"] = self.batch_size
        self.writer.add_hparams(
            hparam_dict=self.hyperparameters,
            metric_dict={
                "hparam/validation_error": self.best_validation_error
            },
        )
        return None

    def validation_set(self, current_epoch: int):
        """[runs forward over validation set with extra logging]

        Args:
            current_epoch (int): [current epoch for logging purposes]

        Returns:
            [None]: [None]
        """
        losses = []
        self.model.eval()
        for batch in self.valid_data:
            predictions = self.model.forward(
                batch[0].to(self.device), batch[1].to(self.device)
            )

            batch_loss = self.exp_rmspe(
                predictions[:, 0], batch[2][:, 0].to(self.device), log=True
            )
            losses.append(batch_loss)
        current_validation_error = torch.stack(losses).mean()

        if (
            not self.best_validation_error
            or self.best_validation_error > current_validation_error
        ):
            self.best_validation_error = current_validation_error
            self.best_epoch = current_epoch

        self.writer.add_scalar(
            "Validation_Loss", torch.stack(losses).mean(), current_epoch
        )
        self.model.train()
        return None


def get_embedding_sizes(train_data_obj: RossmanDataset) -> List[int]:
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
    coloredlogs.install(level="DEBUG")
    logger = logging.getLogger(__name__)

    # Open data objects
    train_data_obj = RossmanDataset.from_pickle("./data/train_data.pkl")
    valid_data_obj = RossmanDataset.from_pickle("./data/valid_data.pkl")

    # get the cardinality of each categorical variable.
    embedding_sizes = get_embedding_sizes(train_data_obj)

    # set batch size
    batch_size = [100000, 500000, 10000]
    cosine_annealing_period = [3, 10, 30, 2]
    layer_sizes = [
        [60, 30, 5],
        [60, 40, 30, 4],
        [40, 10],
        [30],
        [60, 60, 40, 30, 20, 10],
        [60, 60, 60, 40, 30, 20, 10],
    ]
    lr = [0.001, 0.01, 0.05, 0.001]
    dropout = [0.1, 0.5, 0.8]
    # build and train model
    for trial in range(30):
        c1 = int(np.random.choice(cosine_annealing_period))
        c2 = np.random.choice(lr)
        c3 = int(np.random.choice(batch_size))
        c4 = np.random.choice(layer_sizes).copy()
        c5 = np.random.choice(dropout)
        rossman_learner = learner(
            train_data_obj, valid_data_obj, c1, c2, c3, c4, c5,
        )
        rossman_learner.writer.add_text(
            "config".format(trial),
            "Cosine:{}, lr: {}, batchsize: {}, layers:{}, dropout:{}".format(
                c1, c2, c3, c4, c5
            ),
        )

        print(
            "Cosine:{}, lr: {}, batchsize: {}, layers:{}, dropout:{}".format(
                c1, c2, c3, c4, c5
            )
        )
        rossman_learner.training_loop(400)

        # delete the model once done with it or watch the GPU ram disappear.
        del rossman_learner
    # rossman_learner = learner(
    #     train_data_obj, valid_data_obj, 3, 0.001, batch_size
    # )
    # rossman_learner.training_loop(2)
