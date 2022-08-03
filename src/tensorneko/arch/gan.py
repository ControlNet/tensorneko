from abc import abstractmethod, ABC
from typing import Optional, Union, Sequence, Dict

import torch
import torchvision
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam

from ..neko_model import NekoModel


class GAN(NekoModel, ABC):
    """
    Vanilla Generative Adversarial Networks (GAN). To use this model, you need to implement `build_generator` and
    `build_discriminator` methods.

    Args:
        latent_dim (``int``): Dimension of latent space.
        g_learning_rate (``float``): Learning rate for generator.
        d_learning_rate (``float``): Learning rate for discriminator.
        g_steps (``int``, optional): Number of steps for training generator per training step. Default: 1.
        d_steps (``int``, optional): Number of steps for training discriminator per training step. Default: 1.
        num_samples (``int``, optional): Number of samples for visualization in TensorBoard. Default: 8.
        grid_cols (``int``, optional): Number of columns for visualization in TensorBoard. Default: num_samples.
        name (``str``, optional): Name of the model. Default: "gan".
        *args: Other arguments for :class:`~tensorneko.neko_model.NekoModel`.
        **kwargs: Other keyword arguments for :class:`~tensorneko.neko_model.NekoModel`.

    Attributes:
        generator (:class:`~torch.nn.Module`): Generator network.
        discriminator (:class:`~torch.nn.Module`): Discriminator network.

    References:
        Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y.
        (2014). Generative Adversarial Nets. Advances in Neural Information Processing Systems, 27.
        https://proceedings.neurips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html

    """

    def __init__(self, latent_dim: int, g_learning_rate: float, d_learning_rate: float,
        g_steps: int = 1, d_steps: int = 1, num_samples: int = 8, grid_cols: Optional[int] = None,
        name: str = "gan", *args, **kwargs
    ):
        super().__init__(name, *args, **kwargs)

        self.latent_dim = latent_dim
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.g_steps = g_steps
        self.d_steps = d_steps
        self.sample_z = None
        self.num_samples = num_samples
        self.grid_cols = grid_cols or num_samples

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.automatic_optimization = False

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    @abstractmethod
    def build_generator(self) -> Module:
        pass

    @abstractmethod
    def build_discriminator(self) -> Module:
        pass

    @staticmethod
    def g_loss_fn(pred, target) -> Tensor:
        return binary_cross_entropy_with_logits(pred, target)

    @staticmethod
    def d_loss_fn(pred, target) -> Tensor:
        return binary_cross_entropy_with_logits(pred, target)

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]

        d_optimizer, g_optimizer = self.optimizers()
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            d_scheduler, g_scheduler = schedulers
        else:
            d_scheduler, g_scheduler = None, None

        # sample noise
        z = torch.randn(x.size(0), self.latent_dim, device=self.device)

        # train discriminator
        d_loss = None
        d_result = None
        for _ in range(self.d_steps):
            d_result = self.d_step(x, z)
            d_loss = d_result["loss"]
            d_optimizer.zero_grad()
            self.manual_backward(d_loss)
            d_optimizer.step()
        if d_scheduler is not None:
            d_scheduler.step()

        # train generator
        g_loss = None
        g_result = None
        for _ in range(self.g_steps):
            g_result = self.g_step(x, z)
            g_loss = g_result["loss"]
            g_optimizer.zero_grad()
            self.manual_backward(g_loss)
            g_optimizer.step()
        if g_scheduler is not None:
            g_scheduler.step()

        return {
            "loss": d_loss + g_loss,
            **{k: v for k, v in d_result.items() if k != "loss"},
            **{k: v for k, v in g_result.items() if k != "loss"},
        }

    def d_step(self, x: Tensor, z: Tensor) -> Dict[str, Tensor]:
        # forward discriminator
        # generate fake images
        fake_image = self.generator(z)
        fake_label = torch.zeros(x.size(0), 1, device=self.device)

        # generate real label
        real_image = x
        real_label = torch.ones(x.size(0), 1, device=self.device)

        # concat fake and real images for training
        d_batch = torch.cat((fake_image, real_image), dim=0)
        d_label = torch.cat((fake_label, real_label), dim=0)

        # calculate discriminator loss
        d_loss = self.d_loss_fn(self.discriminator(d_batch), d_label)
        return {"loss": d_loss, "d_loss": d_loss}

    def g_step(self, x: Tensor, z: Tensor) -> Dict[str, Tensor]:
        # forward generator
        # generate fake images and labels
        fake_image = self.generator(z)
        target_label = torch.ones(x.size(0), 1, device=self.device)  # target label is real.

        # calculate generator loss
        g_loss = self.g_loss_fn(self.discriminator(fake_image), target_label)
        return {"loss": g_loss, "g_loss": g_loss}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:

        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]

        # sample noise
        z = torch.randn(x.size(0), self.latent_dim, device=self.device)

        # test discriminator
        d_result = self.d_step(x, z)
        d_loss = d_result["loss"]

        # test generator
        g_result = self.g_step(x, z)
        g_loss = g_result["loss"]

        return {
            "loss": d_loss + g_loss,
            **{k: v for k, v in d_result.items() if k != "loss"},
            **{k: v for k, v in g_result.items() if k != "loss"},
        }

    def on_validation_epoch_end(self) -> None:
        super().on_epoch_end()
        if self.sample_z is None:
            self.sample_z = torch.randn(self.num_samples, self.latent_dim, device=self.device)
        sample_generated = self.generator(self.sample_z)
        grid = torchvision.utils.make_grid(sample_generated, nrow=self.grid_cols)
        self.log_image("sample_images", grid)

    def configure_optimizers(self):
        d_optimizer = Adam(self.discriminator.parameters(), lr=self.d_learning_rate)
        g_optimizer = Adam(self.generator.parameters(), lr=self.g_learning_rate)
        return [d_optimizer, g_optimizer]
