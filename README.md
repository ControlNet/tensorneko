<h1 style="text-align: center">TensorNeko</h1> 

<div align="center">
    <img src="https://img.shields.io/github/stars/ControlNet/tensorneko?style=flat-square">
    <img src="https://img.shields.io/github/forks/ControlNet/tensorneko?style=flat-square">
    <a href="https://github.com/ControlNet/tensorneko/issues"><img src="https://img.shields.io/github/issues/ControlNet/tensorneko?style=flat-square"></a>
    <a href="https://pypi.org/project/tensorneko/"><img src="https://img.shields.io/pypi/v/tensorneko?style=flat-square"></a>
    <a href="https://pypi.org/project/tensorneko/"><img src="https://img.shields.io/pypi/dm/tensorneko?style=flat-square"></a>
    <img src="https://img.shields.io/github/license/ControlNet/tensorneko?style=flat-square">
</div>

<div align="center">    
    <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/tensorneko?style=flat-square"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.9.0-EE4C2C?style=flat-square&logo=pytorch"></a>
    <a href="https://www.pytorchlightning.ai/"><img src="https://img.shields.io/badge/Pytorch%20Lightning-%3E%3D1.4.0-792EE5?style=flat-square&logo=pytorch-lightning"></a>
</div>

<div align="center">
    <a href="https://github.com/ControlNet/tensorneko/actions"><img src="https://img.shields.io/github/workflow/status/ControlNet/tensorneko/Unittest/dev?label=unittest&style=flat-square"></a>
    <a href="https://github.com/ControlNet/tensorneko/actions"><img src="https://img.shields.io/github/workflow/status/ControlNet/tensorneko/Release/master?label=release&style=flat-square"></a>
</div>

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
import torch.nn

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
import torch.nn

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

layers:

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

## Neko modules

All `tensorneko.layer` and `tensorneko.module` are `NekoModule`. They can be used in 
[fn.py](https://github.com/kachayev/fn.py) pipe operation.

```python
from tensorneko.layer import Linear
from torch.nn import ReLU
import torch

linear0 = Linear(16, 128, build_activation=ReLU)
linear1 = Linear(128, 1)

f = linear0 >> linear1
print(f(torch.rand(16)).shape)
# torch.Size([1])
```

## Neko IO

Easily load and save different modal data.

```python
import tensorneko as neko
from tensorneko.io.text.text_reader import json_data
from typing import List

# read video (Temporal, Channel, Height, Width)
video_tensor, audio_tensor, video_info = neko.io.read.video.of("path/to/video.mp4")
# write video
neko.io.write.video.to("path/to/video.mp4", 
    video_tensor, video_info.video_fps,
    audio_tensor, video_info.audio_fps
)

# read audio (Channel, Temporal)
audio_tensor, sample_rate = neko.io.read.audio.of("path/to/audio.wav")
# write audio
neko.io.write.audio.to("path/to/audio.wav", audio_tensor, sample_rate)

# read image (Channel, Height, Width) with float value in range [0, 1]
image_tensor = neko.io.read.image.of("path/to/image.png")
# write image
neko.io.write.image.to_png("path/to/image.png", image_tensor)
neko.io.write.image.to_jpeg("path/to/image.jpg", image_tensor)

# read plain text
text_string = neko.io.read.text.of("path/to/text.txt")
# write plain text
neko.io.write.text.to("path/to/text.txt", text_string)

# read json as DataFrame
json_df = neko.io.read.text.of_json("path/to/json.json", to_df=True)
# read json as a object
@json_data
class JsonData:
    x: int
    y: int

json_obj: List[JsonData] = neko.io.read.text.of_json("path/to/json.json", cls=List[JsonData])
# read json as a dict or list
json_dict = neko.io.read.text.of_json("path/to/json.json")
# write json
neko.io.write.text.to_json("path/to/json.json", json_dict)

```

## Neko preprocessing

```python
import tensorneko.preprocess.resize
import tensorneko as neko

# A video tensor with (120, 3, 720, 1280)
video = neko.io.read.video.of("example/video.mp4")
# Get a resized tensor with (120, 3, 256, 256)
tensorneko.preprocess.resize.resize_video(video, (256, 256))
```

#### All preprocessing utils

- `resize_video`
- `resize_image`
- `padding_video`
- `padding_audio`

## Neko Visualization

### Variable Web Watcher
Start a web server to watch the variable status when the program (e.g. training, inference, data preprocessing) is running.
```python
import time
from tensorneko.visualization.watcher import *
data_list = ... # a list of data
def preprocessing(d): ...

# initialize the components
pb = ProgressBar("Processing", total=len(data_list))
logger = Logger("Log message")
var = Variable("Some Value", 0)
line_chart = LineChart("Line Chart", "x", "y")
view = View("Data preprocessing").add_all()

t0 = time.time()
# open server when the code block in running.
with Server(view, port=8000):
    for i, data in enumerate(data_list):
        preprocessing(data) # do some processing here
        
        x = time.time() - t0  # time since the start of the program
        y = i # processed number of data
        line_chart.add(x, y)  # add to the line chart
        logger.log("Some messages")  # log messages to the server
        var.value = ...  # keep tracking a variable
        pb.add(1)  # update the progress bar by add 1
```
When the script is running, go to `127.0.0.1:8000` to keep tracking the status.

### Matplotlib wrappers
Display an image of (C, H, W) shape by `plt.imshow` wrapper.
```python
import tensorneko as neko
import matplotlib.pyplot as plt

image_tensor = ...  # an image tensor with shape (C, H, W)
neko.visualization.imshow(image_tensor)
plt.show()
```

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

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
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

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        x, y = batch
        logit = self(x)
        return logit

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer
        }


model = MnistClassifier("mnist_mlp_classifier", [784, 1024, 512, 10], "ReLU", 0.5, "CrossEntropyLoss", 1e-4, 1e-4)

dm = ...  # The MNIST datamodule from PyTorch Lightning

trainer = neko.NekoTrainer(log_every_n_steps=0, gpus=1, logger=model.name, precision=32,
    checkpoint_callback=ModelCheckpoint(dirpath="./ckpt",
        save_last=True, filename=model.name + "-{epoch}-{val_acc:.3f}", monitor="val_acc", mode="max"
    ))

trainer.fit(model, dm)
```

## Neko Notebook Helpers
Here are some helper functions to better interact with Jupyter Notebook.
```python
import tensorneko as neko
# display a video
neko.notebook.Display.video("path/to/video.mp4")
# display an audio
neko.notebook.Display.audio("path/to/audio.wav")
# display a code file
neko.notebook.Display.code("path/to/code.java")
```

## Neko Utilities

`StringGetter`: Get PyTorch class from string.
```python
import tensorneko as neko
activation = neko.util.get_activation("leakyRelu")()
```

`__`: The arguments to pipe operator
```python
from tensorneko.util import __, _
result = __(20) >> (_ + 1) >> (_ * 2) >> __.get
print(result)
# 42
```

`Seed`: The universal seed for `numpy`, `torch` and Python `random`.
```python
from tensorneko.util import Seed
from torch.utils.data import DataLoader

# set seed to 42 for all numpy, torch and python random
Seed.set(42)

# Apply seed to parallel workers of DataLoader
DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=Seed.get_loader_worker_init(),
    generator=Seed.get_torch_generator()
)
```

`dispatch`: Multi-dispatch implementation for Python. 

To my knowledge, 3 popular multi-dispatch libraries still have critical limitations. 
[plum](https://github.com/wesselb/plum) doesn't support static methods, 
[mutipledispatch](https://github.com/mrocklin/multipledispatch) doesn't support Python type annotation syntax and 
[multimethod](https://github.com/coady/multimethod) doesn't support default augments. TensorNeko can do it all.

```python
from tensorneko.util import dispatch

class DispatchExample:

    @staticmethod
    @dispatch
    def go() -> None:
        print("Go0")

    @staticmethod
    @dispatch
    def go(x: int) -> None:
        print("Go1")

    @staticmethod
    @dispatch
    def go(x: float, y: float = 1.0) -> None:
        print("Go2")

@dispatch
def come(x: int) -> str:
    return "Come1"

@dispatch.of(str)
def come(x) -> str:
    return "Come2"
```


Utilities list:
- `reduce_dict_by`
- `summarize_dict_by`
- `generate_inf_seq`
- `compose`
- `listdir`
- `with_printed`
- `with_printed_shape`
- `is_bad_num`
- `ifelse`
- `dict_add`
- `count_parameters`
- `as_list`
- `Configuration`
- `get_activation`
- `get_loss`
- `Seed`
- `dispatch`
- `__`
