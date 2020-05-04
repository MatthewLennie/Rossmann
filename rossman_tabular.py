import torch
import pandas as pd
import import_rossman_data as rossman
from import_rossman_data import RossmanDataset
from typing import List


class tabular_rossman_model(torch.nn.Module):
    def __init__(self, embedding_sizes: List[int], cont_vars_sizes: int):
        super(tabular_rossman_model, self).__init__()
        # few fully connected layers at the end.
        # remember batch norms
        # dropouts after embeddings.

        # build embeddings for categories
        self.CategoricalEmbeddings = []
        self.embedding_depth = 5
        for i in embedding_sizes:
            self.CategoricalEmbeddings.append(
                torch.nn.Embedding(i, self.embedding_depth)
            )
        self.CategoricalEmbeddings = torch.nn.ModuleList(self.CategoricalEmbeddings)
        # build linear layers for continuous variables and cat embeddings
        self.linear_input_layer = torch.nn.Linear(
            cont_vars_sizes + len(embedding_sizes) * self.embedding_depth,
            cont_vars_sizes,
        )

        self.ReLU1 = torch.nn.ReLU()
        self.linear_layer2 = torch.nn.Linear(cont_vars_sizes, 5)
        self.ReLU2 = torch.nn.ReLU()
        self.linear_layer3 = torch.nn.Linear(5, 2)
        # output
        self.squash = torch.nn.Sigmoid()

    def forward(self, cat_data: torch.tensor, cont_data: torch.tensor) -> torch.tensor:
        cat_outputs = [
            emb(cat_data[:, idx].long())
            for idx, emb in enumerate(self.CategoricalEmbeddings)
        ]
        cat_outputs = torch.cat(cat_outputs, 1)
        # x = self.emb_drop(x)
        l1 = self.linear_input_layer(torch.cat([cat_outputs, cont_data], 1))
        l2 = self.linear_layer2(self.ReLU1(l1))
        l3 = self.linear_layer3(self.ReLU2(l2))
        return self.squash(l3)


class learner:
    def __init__(self):
        self.loss = torch.nn.MSELoss()
        raise (NotImplementedError)
        return None

    def training_step(self):
        raise (NotImplementedError)
        return None

    def training_loop(self):
        raise (NotImplementedError)
        return None

    def validation_set(self):
        raise (NotImplementedError)
        return None


def get_embedding_sizes(train_data_obj: RossmanDataset):
    embedding_sizes = []
    # Get embedding Layer sizes
    for key in rossman.cat_vars:

        embedding_sizes.append(max([len(train_data_obj.data[key].unique()), 2]))

    return embedding_sizes


if __name__ == "__main__":

    # Open data objects
    train_data_obj = RossmanDataset.from_pickle("./data/train_data.pkl")
    valid_data_obj = RossmanDataset.from_pickle("./data/valid_data.pkl")

    embedding_sizes = get_embedding_sizes(train_data_obj)

    batch_size = 5000
    data_loader = torch.utils.data.DataLoader(train_data_obj, batch_size=batch_size)

    model = tabular_rossman_model(embedding_sizes, len(rossman.cont_vars))

    for i in data_loader:
        answer = model.forward(i[0], i[1])
    print("asdf")
