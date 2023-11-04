from abc import ABC
from typing import Optional, Union, Sequence, Dict

from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, AUROC

from ..neko_model import NekoModel


class BinaryClassifier(NekoModel, ABC):

    def __init__(self, model=None, learning_rate: float = 1e-4, distributed: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.distributed = distributed
        self.loss_fn = BCEWithLogitsLoss()
        self.acc_fn = Accuracy(task="binary")
        self.f1_fn = F1Score(task="binary")
        self.auc_fn = AUROC(task="binary")

    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x):
        return self.model(x)

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = self.loss_fn(y_hat, y)
        prob = y_hat.sigmoid()
        acc = self.acc_fn(prob, y)
        f1 = self.f1_fn(prob, y)
        auc = self.auc_fn(prob, y)
        return {"loss": loss, "acc": acc, "f1": f1, "auc": auc}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        return self.step(batch)

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
