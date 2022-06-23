"""A simple model for the CIFAR-10 dataset."""

from pytorch_lightning.core.lightning import LightningModule
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
from torch.optim.adam import Adam
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms.transforms import ToTensor


class LitAutoEncoder(LightningModule):
    """Lighning module to handle ML steps."""

    def __init__(self) -> None:
        """Creates ML model and defines computations."""

        super().__init__()

        # Model layers
        self.encoder = Sequential(Linear(28 * 28, 128), ReLU(), Linear(128, 3))
        self.decoder = Sequential(Linear(3, 128), ReLU(), Linear(128, 28 * 28))

    def configure_optimizers(self):
        """Gradient descent optimizer for forward pass and backpropagation.

        Returns:
            optimizer: The model's optimizer.
        """

        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = CIFAR10("data", download=True, transform=ToTensor())
