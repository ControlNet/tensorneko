from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, Dict

from torch import Tensor
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..neko_model import NekoModel


class AutoEncoder(NekoModel, ABC):

    def __init__(self, name: str, learning_rate: float, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.learning_rate = learning_rate
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.loss_fn = MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    @abstractmethod
    def build_encoder(self) -> Module:
        pass

    @abstractmethod
    def build_decoder(self) -> Module:
        pass

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        return {"loss": loss}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }