from abc import abstractmethod, ABC
from typing import Optional, Union, Sequence, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..neko_model import NekoModel
from ..layer import VectorQuantizer


class VQVAE(NekoModel, ABC):
    """
    Vector Quantized Variational Autoencoder (VQVAE). To use this model, you need to implement `build_encoder` and
    `build_decoder` methods.
    
    Args:
        latent_dim (``int``): Dimension of latent space.
        n_embeddings (``int``): Number of embeddings.
        learning_rate (``float``, optional): Learning rate. Default: 0.0003.
        *args: Other arguments for :class:`~tensorneko.neko_model.NekoModel`.
        **kwargs: Other keyword arguments for :class:`~tensorneko.neko_model.NekoModel`.

    Attributes:
        encoder (:class:`~torch.nn.Module`): Encoder.
        decoder (:class:`~torch.nn.Module`): Decoder.
        vector_quantizer (:class:`~tensorneko.layer.vector_quantizer.VectorQuantizer`): Vector quantizer.

    References:
        Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. Retrieved 18 May 2022,
        from https://arxiv.org/abs/1711.00937

    """

    def __init__(self, latent_dim: int, n_embeddings: int, beta: float = 0.25, learning_rate: float = 2e-4,
        name: str = "vqvae", *args, **kwargs
    ):
        super().__init__(name, *args, **kwargs)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vector_quantizer = VectorQuantizer(n_embeddings, latent_dim, beta)

        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z_e = self.encoder(x)
        z_q, embedding_loss = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, embedding_loss

    @abstractmethod
    def build_encoder(self) -> Module:
        pass

    @abstractmethod
    def build_decoder(self) -> Module:
        pass

    @staticmethod
    def rec_loss_fn(pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean() / torch.var(target)

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]
        x_hat, embedding_loss = self(x)
        rec_loss = self.rec_loss_fn(x_hat, x)
        loss = rec_loss + embedding_loss
        return {"loss": loss, "r_loss": rec_loss, "e_loss": embedding_loss}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]
        x_hat, embedding_loss = self(x)
        rec_loss = self.rec_loss_fn(x_hat, x)
        loss = rec_loss + embedding_loss
        return {"loss": loss, "r_loss": rec_loss, "e_loss": embedding_loss}

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        return self(batch)[0]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }
