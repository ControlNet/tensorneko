import numpy as np
import torchvision
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader

import tensorneko
from tensorneko import NekoTrainer
from tensorneko.arch.gan import GAN
from tensorneko.arch.wgan import WGAN
from tensorneko.callback import DisplayMetricsCallback
from tensorneko.callback.earlystop_lr import EarlyStoppingLR


class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GANImpl(GAN):

    def build_generator(self) -> Module:
        return Generator(latent_dim=self.latent_dim, img_shape=(1, 28, 28))

    def build_discriminator(self) -> Module:
        return Discriminator(img_shape=(1, 28, 28))


class WGANImpl(WGAN, GANImpl):
    pass


if __name__ == '__main__':
    model = WGANImpl(128, 0.0002, 0.0003, gp_weight=10.0, d_steps=3)
    trainer = NekoTrainer(log_every_n_steps=100, logger=model.name, precision=32, max_epochs=2,
                          enable_checkpointing=False,
                          callbacks=[DisplayMetricsCallback(), EarlyStoppingLR(0.00025, mode="all")])

    train_loader = DataLoader(
        torchvision.datasets.MNIST('./', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=128, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('./', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=128, shuffle=True)

    with tensorneko.visualization.tensorboard.Server():
        trainer.fit(model, train_loader, test_loader)
