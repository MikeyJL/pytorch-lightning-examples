"""A simple example with MNIST dataset.

As per instructions from https://github.com/Lightning-AI/lightning with addtions.

- Sequential network with 2 layers
- TensorBoardLogger
- Early stopping or epochs
"""

from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.nn import Sequential, Linear, ReLU
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class LitAutoEncoder(LightningModule):
    """Lighning module to handle ML steps."""

    def __init__(self):
        """Creates ML model."""

        super().__init__()
        self.encoder = Sequential(Linear(28 * 28, 128), ReLU(), Linear(128, 3))
        self.decoder = Sequential(Linear(3, 128), ReLU(), Linear(128, 28 * 28))

    def configure_optimizers(self):
        """Gradient descent optimizer for forward pass and backpropagation.

        Returns:
            optimizer: The model's optimizer.
        """

        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch):
        """Trains the ML model.

        Args:
            batch: the batch size.

        Returns:
            loss: The loss of the model.
        """

        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = mse_loss(x_hat, x)
        self.log("train_loss", loss)

        return loss

    def forward(self, x):
        """Inference and prediction method.

        Args:
            x: Value used for inference or prediction.

        Returns:
            embedding: The infered/predicted value(s).
        """

        embedding = self.encoder(x)
        return embedding


# Gets and pre-processes data
dataset = MNIST("data", download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

# Instantiates model
autoencoder = LitAutoEncoder()

# Custom logger
tb_logger = TensorBoardLogger(save_dir="examples/basic/logs/")
trainer = Trainer(
    logger=tb_logger, callbacks=[EarlyStopping(monitor="train_loss", mode="min")]
)

# Initiates training
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
