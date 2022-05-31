from abc import ABC
from typing import Optional, Dict

import torch
from torch import Tensor

from .gan import GAN


class WGAN(GAN, ABC):
    """
    Wasserstein Generative Adversarial Networks (WGAN) or WGAN-GP. To use this model, you need to implement
    `build_generator` and `build_discriminator` methods.

    Args:
        latent_dim (``int``): Dimension of latent space.
        g_learning_rate (``float``): Learning rate for generator.
        d_learning_rate (``float``): Learning rate for discriminator.
        gp_weight (``float``, optional): Weight of gradient penalty if value > 0. Defaults to 10.
        g_steps (``int``, optional): Number of steps for training generator per training step. Default: 1.
        d_steps (``int``, optional): Number of steps for training discriminator per training step. Default: 1.
        num_samples (``int``, optional): Number of samples for visualization in TensorBoard. Default: 8.
        grid_cols (``int``, optional): Number of columns for visualization in TensorBoard. Default: num_samples.
        name (``str``, optional): Name of the model. Default: "wgan".
        *args: Other arguments for :class:`~tensorneko.neko_model.NekoModel`.
        **kwargs: Other keyword arguments for :class:`~tensorneko.neko_model.NekoModel`.

    Attributes:
        generator (:class:`~torch.nn.Module`): Generator network.
        discriminator (:class:`~torch.nn.Module`): Discriminator network.

    References:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein Generative Adversarial Networks. Proceedings of the
        34th International Conference on Machine Learning, 214–223. https://proceedings.mlr.press/v70/arjovsky17a.html

        Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved Training of
        Wasserstein GANs. Advances in Neural Information Processing Systems, 30.
        https://proceedings.neurips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html

    """

    def __init__(self, latent_dim: int, g_learning_rate: float, d_learning_rate: float, gp_weight: float = 10.0,
        g_steps: int = 1, d_steps: int = 1, num_samples: int = 8, grid_cols: Optional[int] = None,
        name: str = "wgan", *args, **kwargs
    ):
        super(WGAN, self).__init__(latent_dim, g_learning_rate, d_learning_rate, g_steps, d_steps, num_samples,
            grid_cols, name, *args, **kwargs)
        self.gp_weight = gp_weight

    @staticmethod
    def d_loss_fn(pred: Tensor, target: Tensor) -> Tensor:
        real_score = pred[target == 1]
        fake_score = pred[target == 0]
        return fake_score.mean() - real_score.mean()

    @staticmethod
    def g_loss_fn(pred: Tensor, target: Tensor) -> Tensor:
        return -pred.mean()

    def gradient_penalty_fn(self, batch_size: int, real_images: Tensor, fake_images: Tensor) -> Tensor:
        alpha = torch.randn(size=(batch_size, 1, 1, 1), device=self.device)
        diff = fake_images - real_images

        # 1. Get the discriminator output for this interpolated image.
        interpolates = torch.autograd.Variable(real_images + alpha * diff, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        # 3. Calculate the norm of the gradients.
        gradients = gradients.view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

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

        # gradient penalty
        if self.training and self.gp_weight > 0:
            gp = self.gradient_penalty_fn(x.size(0), x, fake_image)
            loss = d_loss + self.gp_weight * gp
        else:
            gp = torch.tensor(0., device=self.device)
            loss = d_loss

        return {"loss": loss, "d_loss": d_loss, "gp": gp}
