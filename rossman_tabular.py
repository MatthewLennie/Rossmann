import torch
import pandas as pd
import import_rossman_data as rossman
from import_rossman_data import RossmanDataset
from typing import List
from torch.utils.tensorboard import SummaryWriter
import random
import coloredlogs, logging


class tabular_rossman_model(torch.nn.Module):
    def __init__(self, embedding_sizes: List[int], cont_vars_sizes: int):
        super(tabular_rossman_model, self).__init__()

        # build embeddings for categories
        self.CategoricalEmbeddings = []
        self.embedding_depth = 5
        for i in embedding_sizes:
            self.CategoricalEmbeddings.append(
                torch.nn.Embedding(i, self.embedding_depth)
            )

        # convert the list of embeddings to a ModuleList so that PyTorch finds
        # it as a param.
        self.CategoricalEmbeddings = torch.nn.ModuleList(self.CategoricalEmbeddings)
        self.EmbeddingDropout = torch.nn.Dropout()
        # build linear layers for continuous variables and cat embeddings
        self.linear_input_layer = torch.nn.Linear(
            cont_vars_sizes + len(embedding_sizes) * self.embedding_depth,
            cont_vars_sizes,
        )

        self.Dropout1 = torch.nn.Dropout()
        self.Batch1 = torch.nn.BatchNorm1d(cont_vars_sizes)
        self.ReLU1 = torch.nn.ReLU()

        self.linear_layer2 = torch.nn.Linear(cont_vars_sizes, 5)
        self.Dropout2 = torch.nn.Dropout()
        self.Batch2 = torch.nn.BatchNorm1d(5)
        self.ReLU2 = torch.nn.ReLU()

        self.linear_layer3 = torch.nn.Linear(5, 2)
        self.Dropout1 = torch.nn.Dropout()
        self.squash = torch.nn.Sigmoid()
        self.batch = 0

    def forward(self, cat_data: torch.tensor, cont_data: torch.tensor) -> torch.tensor:
        assert cat_data.device.type == "cuda"
        self.batch += 1
        logger.debug("count: {}".format(self.batch))
        cat_outputs = [
            emb(cat_data[:, idx].long())
            for idx, emb in enumerate(self.CategoricalEmbeddings)
        ]
        cat_outputs = self.EmbeddingDropout(torch.cat(cat_outputs, 1))

        l1 = self.Dropout1(
            self.ReLU1(
                self.linear_input_layer(
                    torch.cat([cat_outputs, self.Batch1(cont_data)], 1)
                )
            )
        )
        l2 = self.Dropout2(self.ReLU2(self.linear_layer2(self.ReLU1(l1))))
        l2 = self.Batch2(l2)
        l3 = self.linear_layer3(self.ReLU2(l2))
        return self.squash(l3)


class learner:
    def __init__(
        self,
        train_data: torch.utils.data.DataLoader,
        valid_data: torch.utils.data.DataLoader,
    ):
        self.writer = SummaryWriter("runs/{}".format(random.randint(0, 1e9)))

        # data loaders
        self.train_data = train_data
        self.valid_data = valid_data

        # create model skeleton
        self.model = tabular_rossman_model(embedding_sizes, len(rossman.cont_vars))
        self.initialize_optimizer()
        self.loss = torch.nn.MSELoss()

        # transfer everything to the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        return None

    def initialize_optimizer(self):
        self.optim = torch.optim.Adam(self.model.parameters())
        # TODO lr scheduler.

    def training_step(self, input_data: List[torch.tensor]) -> torch.tensor:
        cat = input_data[0].to(self.device)
        cont = input_data[1].to(self.device)
        predictions = self.model.forward(cat, cont)
        # TODO include both terms in loss
        batch_loss = self.loss(predictions[:, 0], input_data[2][:, 0].to(self.device))
        batch_loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return batch_loss

    def dump_model_parameters_to_log(self):
        for param in self.model.named_parameters():

            self.writer.add_histogram(param[0], param[1])
            try:
                self.writer.add_histogram(
                    "Gradient_of_{}".format(param[0]), param[1].grad
                )
            except NotImplementedError:
                logger.debug("Missing Gradient for {}".format(param[0]))

    def training_loop(self, epochs: int):
        for current_epoch in range(epochs):
            for batch in self.train_data:
                training_batch_loss = self.training_step(batch)

            # perform tensorboard logging.
            self.writer.add_scalar("Training_Loss", training_batch_loss, current_epoch)
            self.validation_set(current_epoch)
            self.dump_model_parameters_to_log()
        return None

    def validation_set(self, current_epoch: int):
        losses = []
        for batch in self.valid_data:
            predictions = self.model.forward(
                batch[0].to(self.device), batch[1].to(self.device)
            )
            batch_loss = self.loss(predictions[:, 0], batch[2][:, 0].to(self.device))
            losses.append(batch_loss)
        self.writer.add_scalar(
            "Validation_Loss", torch.stack(losses).mean(), current_epoch
        )
        return None


def get_embedding_sizes(train_data_obj: RossmanDataset) -> List[int]:
    embedding_sizes = []
    # Get embedding Layer sizes based on unique categories.
    # Potential bug if rare classes don't appear in the training set.
    for key in rossman.cat_vars:
        embedding_sizes.append(max([len(train_data_obj.data[key].unique()), 2]))
    return embedding_sizes


if __name__ == "__main__":
    coloredlogs.install(level="DEBUG")
    logger = logging.getLogger(__name__)

    # Set up the GPU

    # Open data objects
    train_data_obj = RossmanDataset.from_pickle("./data/train_data.pkl")
    valid_data_obj = RossmanDataset.from_pickle("./data/valid_data.pkl")

    embedding_sizes = get_embedding_sizes(train_data_obj)

    batch_size = 500000
    train_data_loader = torch.utils.data.DataLoader(
        train_data_obj, batch_size=batch_size,  ##pin_memory=True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_data_obj, batch_size=batch_size,  ##pin_memory=True
    )

    # model = tabular_rossman_model(embedding_sizes, len(rossman.cont_vars))

    rossman_learner = learner(train_data_loader, valid_data_loader)
    rossman_learner.training_loop(300)

    # for i in data_loader:
    #     answer = model.forward(i[0], i[1])
    #     rossman_learner.training_step(i)
    # print("asdf")
