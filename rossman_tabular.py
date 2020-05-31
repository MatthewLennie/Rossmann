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
from FastTensorDataLoader import FastTensorDataLoader


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
            train_data (import_rossman_data:RossmanDataset): [training data]
            valid_data (import_rossman_data:RossmanDataset): [validation data]
            layer_sizes (List[int]): [sizes of the hidden layers]
        Returns:
            [learner]: [learner object]
        """

        # data loaders
        # Don't do like this with a large dataset if you are going
        # to do hyperparameter search
        self.train_data_obj = train_data_obj
        self.valid_data_obj = valid_data_obj
        self.batch_size = batch_size
        self.load_data(train_data_obj, valid_data_obj, self.batch_size)

        self.build_model(layer_sizes, dropout)
        # self.initialize_optimizer()

    def build_model(self, layer_sizes, dropout):

        # Create the embedding depths based on a simple rule from jeremey
        embedding_depths = [
            min(600, round(1.6 * x ** 0.56)) for x in self.embedding_sizes
        ]

        # add the non-hidden layer sizes to the array containing the layer sizes
        layer_sizes.insert(0, len(rossman.cont_vars) + sum(embedding_depths))
        layer_sizes.append(2)

        # create model skeleton
        self.model = TabularRossmanModel(
            self.embedding_sizes,
            embedding_depths,
            layer_sizes,
            dropout,
            # self.writer,
        )

        return None

    def load_data(self, train_data_obj, valid_data_obj, batch_size):
        # create data loaders from datasets
        # get the cardinality of each categorical variable.
        self.embedding_sizes = self.get_embedding_sizes(self.train_data_obj)

        self.train_data = FastTensorDataLoader(
            train_data_obj.x_data_cat,
            train_data_obj.x_data_cont,
            train_data_obj.Y_data,
            batch_size=batch_size,
            shuffle=True,
        )
        self.valid_data = FastTensorDataLoader(
            valid_data_obj.x_data_cat,
            valid_data_obj.x_data_cont,
            valid_data_obj.Y_data,
            batch_size=batch_size,
            shuffle=True,
        )

        # del self.train_data_obj.data
        # del self.valid_data_obj.data

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
