<h1 style="text-align: center">TensorNeko</h1> 

Tensor Neural Engine Kompanion. An util library based on PyTorch and PyTorch Lightning.

## Install

```shell
pip install tensorneko
```

## Neko Layers and Modules

Build an MLP with linear layers. The activation and normalization will be placed in the hidden layers.

784 -> 1024 -> 512 -> 10

```python
import tensorneko as neko
import torch

mlp = neko.module.MLP(
    neurons=[784, 1024, 512, 10],
    build_activation=torch.nn.ReLU,
    build_normalization=[
        lambda: torch.nn.BatchNorm1d(1024),
        lambda: torch.nn.BatchNorm1d(512)
    ],
    dropout_rate=0.5
)
```

Build a Conv2d with activation and normalization.

```python
import tensorneko as neko
import torch

conv2d = neko.layer.Conv2d(
    in_channels=256,
    out_channels=1024,
    kernel_size=(3, 3),
    padding=(1, 1),
    build_activation=torch.nn.ReLU,
    build_normalization=lambda: torch.nn.BatchNorm2d(256),
    normalization_after_activation=False
)
```

#### All modules and layers

layer:

- `Concatenate`
- `Conv2d`
- `Linear`
- `Log`
- `PatchEmbedding2d`
- `PositionalEmbedding`
- `Reshape`

modules:

- `DenseBlock`
- `InceptionModule`
- `MLP`
- `ResidualBlock` and `ResidualModule`
- `AttentionModule`, `TransformerEncoderBlock` and `TransformerEncoder`

## Neko reader

Easily load different modal data.

```python
import tensorneko as neko

# read video (Temporal, Channel, Height, Width)
video_tensor = neko.io.read.video.of("path/to/video.mp4")
# read audio (Channel, Temporal)
audio_tensor = neko.io.read.audio.of("path/to/audio.wav")
# read image (Channel, Height, Width)
image_tensor = neko.io.read.audio.of("path/to/image.png")
# read text 
text_string = neko.io.read.text.of("path/to/text.txt")
```

## Neko preprocessing

```python
import tensorneko as neko

# A video tensor with (120, 3, 720, 1280)
video = neko.io.read.video.of("example/video.mp4")
# Get a resized tensor with (120, 3, 256, 256)
neko.preprocess.resize_video(video, (256, 256))
```

#### All preprocessing utils

- `resize_video`
- `resize_image`

## Neko Model

Build and train a simple model for classifying MNIST with MLP.

```python
from typing import Optional, Union, Sequence, Dict, List

import torch.nn
from torch import Tensor
from torch.optim import Adam
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

import tensorneko as neko
from tensorneko.util import get_activation, get_loss


class MnistClassifier(neko.Model):

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

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        x, y = batch
        logit = self(x)
        prob = logit.sigmoid()
        loss = self.loss_func(prob, y)
        acc = self.acc_func(prob.max(dim=1)[1], y)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        x, y = batch
        logit = self(x)
        prob = logit.sigmoid()
        loss = self.loss_func(prob, y)
        acc = self.acc_func(prob.max(dim=1)[1], y)
        return {"loss": loss, "acc": acc}

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        x, y = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer
        }


model = MnistClassifier("mnist_mlp_classifier", [784, 1024, 512, 10], "ReLU", 0.5, "CrossEntropyLoss", 1e-4, 1e-4)

dm = ...  # The MNIST datamodule from PyTorch Lightning

trainer = neko.Trainer.build(log_every_n_steps=0, gpus=1, logger=model.name, precision=32,
    checkpoint_callback=ModelCheckpoint(dirpath="./ckpt",
        save_last=True, filename=model.name + "-{epoch}-{val_acc:.3f}", monitor="val_acc", mode="max"
    ))

trainer.fit(model, dm)
```

