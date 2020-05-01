import torch
import pandas as pd
import import_rossman_data as rossman


class tabular_rossman_model(torch.nn.Module):
    def __init__(self):
        super(tabular_rossman_model, self).__init__()
        # build embeddings for categories
        # build linear layers for continuous variables
        # few fully connected layers at the end.
        # remember batch norms
        # dropouts after embeddings.
        self.squash = torch.nn.Sigmoid()
        self.loss = torch.nn.MSELoss()
        raise (NotImplementedError)

    def forward(self, data):
        raise (NotImplementedError)
        return None


class learner:
    def __init__(self):
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


if __name__ == "__main__":

    # Open data objects
    train_data_obj = rossman.RossmanDataset.from_pickle("./data/train_data.pkl")
    valid_data_obj = rossman.RossmanDataset.from_pickle("./data/valid_data.pkl")

    #
