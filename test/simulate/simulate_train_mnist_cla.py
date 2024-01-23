from typing import Optional, Union, Sequence, Dict, List

import lightning.pytorch as pl
import torch.nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST

import tensorneko as neko
from tensorneko import NekoTrainer
from tensorneko.util import get_activation, get_loss


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


class MnistClassifier(neko.NekoModel):

    def __init__(self, name: str, mlp_neurons: List[int], activation: str, dropout_rate: float, loss: str,
        learning_rate: float, weight_decay: float
    ):
        super().__init__(name)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.flatten = torch.nn.Flatten()
        self.mlp = neko.module.MLP(
            neurons=mlp_neurons,
            build_activation=get_activation(activation),
            dropout_rate=dropout_rate
        )
        self.loss_func = get_loss(loss)()
        self.acc_func = Accuracy()

    def forward(self, x):
        # (batch, 28, 28)
        x = self.flatten(x)
        # (batch, 768)
        x = self.mlp(x)
        # (batch, 10)
        return x

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        x, y = batch
        logit = self(x)
        prob = logit.sigmoid()
        loss = self.loss_func(logit, y)
        acc = self.acc_func(prob.max(dim=1)[1], y)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        x, y = batch
        logit = self(x)
        prob = logit.sigmoid()
        loss = self.loss_func(logit, y)
        acc = self.acc_func(prob.max(dim=1)[1], y)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer
        }


if __name__ == '__main__':
    model = MnistClassifier("mnist_mlp_classifier", [784, 1024, 512, 10], "ReLU", 0.5, "CrossEntropyLoss", 1e-4, 1e-4)

    dm = MNISTDataModule()

    trainer = NekoTrainer(log_every_n_steps=50, logger=model.name, precision=32, max_epochs=2,
                          callbacks=[ModelCheckpoint(dirpath="./ckpt",
                                                     save_last=True, filename=model.name + "-{epoch}-{val_acc:.3f}",
                                                     monitor="val_acc", mode="max"
                                                     )])
    trainer.fit(model, dm)
