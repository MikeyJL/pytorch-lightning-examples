"""A simple example of how pytorch_lightning works.

As per instructions from https://github.com/Lightning-AI/lightning.
"""

from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.nn import Sequential, Linear, ReLU
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class LitAutoEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential(Linear(28 * 28, 128), ReLU(), Linear(128, 3))
        self.decoder = Sequential(Linear(3, 128), ReLU(), Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST("data", download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
tb_logger = TensorBoardLogger(save_dir="examples/basic/logs/")
trainer = Trainer(logger=tb_logger)
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
