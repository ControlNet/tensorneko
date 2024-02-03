from typing import Optional, Union, Sequence, Dict, Any

from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, AUROC

from ..neko_model import NekoModel
from ..util import Shape


class BinaryClassifier(NekoModel):

    def __init__(self, name, model: Module, learning_rate: float = 1e-4, distributed: bool = False,
        input_shape: Optional[Shape] = None
    ):
        super().__init__(name, input_shape=input_shape, distributed=distributed)
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = BCEWithLogitsLoss()
        self.acc_fn = Accuracy(task="binary")
        self.f1_fn = F1Score(task="binary")
        self.auc_fn = AUROC(task="binary")

    @classmethod
    def from_module(cls, model: Module, learning_rate: float = 1e-4, name: str = "binary_classifier",
        distributed: bool = False, input_shape: Optional[Shape] = None
    ):
        return cls(name, model, learning_rate, distributed, input_shape)

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
        return {"loss": loss, "metric/acc": acc, "metric/f1": f1, "metric/auc": auc}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        return self.step(batch)

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        return self.step(batch)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
