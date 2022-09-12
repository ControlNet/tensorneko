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
    <a href="https://www.pytorchlightning.ai/"><img src="https://img.shields.io/badge/Pytorch%20Lightning-%3E%3D1.7.0-792EE5?style=flat-square&logo=pytorch-lightning"></a>
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

To use the library without PyTorch and PyTorch Lightning, you can install the util library (support Python 3.7 ~ 3.10 with limited features) with following command.
```shell
pip install tensorneko_util
```

## Neko Layers, Modules and Architectures

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

#### All architectures, modules and layers

Layers:

- `Aggregation`
- `Concatenate`
- `Conv`, `Conv1d`, `Conv2d`, `Conv3d`
- `GaussianNoise`
- `ImageAttention`, `SeqAttention`
- `MaskedConv2d`, `MaskedConv2dA`, `MaskedConv2dB`
- `Linear`
- `Log`
- `PatchEmbedding2d`
- `PositionalEmbedding`
- `Reshape`
- `Stack`
- `VectorQuantizer`

Modules:

- `DenseBlock`
- `InceptionModule`
- `MLP`
- `ResidualBlock` and `ResidualModule`
- `AttentionModule`, `TransformerEncoderBlock` and `TransformerEncoder`
- `GatedConv`

Architectures:
- `AutoEncoder`
- `GAN`
- `WGAN`
- `VQVAE`

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
from tensorneko.io import json_data
from typing import List

# read video (Temporal, Channel, Height, Width)
video_tensor, audio_tensor, video_info = neko.io.read.video("path/to/video.mp4")
# write video
neko.io.write.video("path/to/video.mp4", 
    video_tensor, video_info.video_fps,
    audio_tensor, video_info.audio_fps
)

# read audio (Channel, Temporal)
audio_tensor, sample_rate = neko.io.read.audio("path/to/audio.wav")
# write audio
neko.io.write.audio("path/to/audio.wav", audio_tensor, sample_rate)

# read image (Channel, Height, Width) with float value in range [0, 1]
image_tensor = neko.io.read.image("path/to/image.png")
# write image
neko.io.write.image("path/to/image.png", image_tensor)
neko.io.write.image("path/to/image.jpg", image_tensor)

# read plain text
text_string = neko.io.read.text("path/to/text.txt")
# write plain text
neko.io.write.text("path/to/text.txt", text_string)

# read json as python dict or list
json_dict = neko.io.read.json("path/to/json.json")
# read json as an object
@json_data
class JsonData:
    x: int
    y: int

json_obj: List[JsonData] = neko.io.read.json("path/to/json.json", cls=List[JsonData])
# write json from python dict/list or json_data decorated object
neko.io.write.json("path/to/json.json", json_dict)
neko.io.write.json("path/to/json.json", json_obj)
```

Besides, the read/write for `mat` and `pickle` files is also supported.


## Neko preprocessing

```python
import tensorneko as neko

# A video tensor with (120, 3, 720, 1280)
video = neko.io.read.video("example/video.mp4").video
# Get a resized tensor with (120, 3, 256, 256)
resized_video = neko.preprocess.resize_video(video, (256, 256))
```

#### All preprocessing utils

- `resize_video`
- `resize_image`
- `padding_video`
- `padding_audio`
- `crop_with_padding`
- `frames2video`

if `ffmpeg` is available, you can use below ffmpeg wrappers.

- `video2frames`
- `merge_video_audio`
- `resample_video_fps`
- `mp32wav`

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

### Tensorboard Server

Simply run tensorboard server in Python script.
```python
import tensorneko as neko

with neko.visualization.tensorboard.Server(port=6006):
    trainer.fit(model, dm)
```

### Matplotlib wrappers
Display an image of (C, H, W) shape by `plt.imshow` wrapper.
```python
import tensorneko as neko
import matplotlib.pyplot as plt

image_tensor = ...  # an image tensor with shape (C, H, W)
neko.visualization.matplotlib.imshow(image_tensor)
plt.show()
```

### Predefined colors
Several aesthetic colors are predefined.

```python
import tensorneko as neko
import matplotlib.pyplot as plt

# use with matplotlib
plt.plot(..., color=neko.visualization.Colors.RED)

# the palette for seaborn is also available
from tensorneko_util.visualization.seaborn import palette
import seaborn as sns
sns.set_palette(palette)
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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer
        }


model = MnistClassifier("mnist_mlp_classifier", [784, 1024, 512, 10], "ReLU", 0.5, "CrossEntropyLoss", 1e-4, 1e-4)

dm = ...  # The MNIST datamodule from PyTorch Lightning

trainer = neko.NekoTrainer(log_every_n_steps=100, gpus=1, logger=model.name, precision=32,
    callbacks=[ModelCheckpoint(dirpath="./ckpt",
        save_last=True, filename=model.name + "-{epoch}-{val_acc:.3f}", monitor="val_acc", mode="max"
    )])

trainer.fit(model, dm)
```

## Neko Callbacks

Some simple but useful pytorch-lightning callbacks are provided.

- `DisplayMetricsCallback`
- `EarlyStoppingLR`: Early stop training when learning rate reaches threshold.

## Neko Notebook Helpers
Here are some helper functions to better interact with Jupyter Notebook.
```python
import tensorneko as neko
# display a video
neko.notebook.display.video("path/to/video.mp4")
# display an audio
neko.notebook.display.audio("path/to/audio.wav")
# display a code file
neko.notebook.display.code("path/to/code.java")
```

## Neko Debug Tools

Get the default values from `ArgumentParser` args. It's convenient to use this in the notebook.
```python
from argparse import ArgumentParser
from tensorneko.debug import get_parser_default_args

parser = ArgumentParser()
parser.add_argument("integers", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--sum", dest="accumulate", action="store_const", const=sum, default=max)
args = get_parser_default_args(parser)

print(args.integers)  # [1, 2, 3]
print(args.accumulate)  # <function sum at ...>
```

## Neko Evaluation

Some metrics function for evaluation are provided.

- `iou_1d`
- `iou_2d`
- `psnr_video`
- `psnr_image`
- `ssim_video`
- `ssim_image`


## Neko Utilities

### Functional

`__`: The arguments to pipe operator. (Inspired from [fn.py](https://github.com/kachayev/fn.py))
```python
from tensorneko.util import __, _
result = __(20) >> (_ + 1) >> (_ * 2) >> __.get
print(result)
# 42
```

`Seq` and `Stream`: A collection wrapper for method chaining with concurrent supporting.
```python
from tensorneko.util import Seq, Stream, _
from tensorneko_util.backend.parallel import ParallelType
# using method chaining
seq = Seq.of(1, 2, 3).map(_ + 1).filter(_ % 2 == 0).map(_ * 2).take(2).to_list()
# return [4, 8]

# using bit shift operator to chain the sequence
seq = Seq.of(1, 2, 3) << Seq.of(2, 3, 4) << [3, 4, 5]
# return Seq(1, 2, 3, 2, 3, 4, 3, 4, 5)

# run concurrent with `for_each` for Stream
if __name__ == '__main__':
    Stream.of(1, 2, 3, 4).for_each(print, progress_bar=True, parallel_type=ParallelType.PROCESS)
```

`Option`: A monad for dealing with data.
```python
from tensorneko.util import return_option

@return_option
def get_data():
    if some_condition:
        return 1
    else:
        return None

def process_data(n: int):
    if condition(n):
        return n
    else:
        return None
    

data = get_data()
data = data.map(process_data).get_or_else(-1)  # if the response is None, return -1
```

`Eval`: A monad for lazy evaluation.
```python
from tensorneko.util import Eval

@Eval.always
def call_by_name_var():
    return 42

@Eval.later
def call_by_need_var():
    return 43

@Eval.now
def call_by_value_var():
    return 44


print(call_by_name_var.value)  # 42
```

### Reactive
This library provides event bus based reactive tools. The API integrates the Python type annotation syntax.

```python
# useful decorators for default event bus
from tensorneko.util import (
    subscribe, # run in the main thread
    subscribe_thread, # run in a new thread
    subscribe_async, # run async
    subscribe_process # run in a new process
)
# Event base type
from tensorneko.util import Event

class LogEvent(Event):
    def __init__(self, message: str):
        self.message = message

# the event argument should be annotated correctly
@subscribe
def log_information(event: LogEvent):
    print(event.message)

@subscribe_thread
def log_information_thread(event: LogEvent):
    print(event.message, "in another thread")

if __name__ == '__main__':
    # emit an event, and then the event handler will be invoked
    # The sequential order is not guaranteed
    LogEvent("Hello world!")
    # one possible output:
    # Hello world! in another thread
    # Hello world!
```

### Multiple Dispatch

`dispatch`: Multi-dispatch implementation for Python. 

To my knowledge, 3 popular multi-dispatch libraries still have critical limitations. 
[plum](https://github.com/wesselb/plum) doesn't support static methods, 
[mutipledispatch](https://github.com/mrocklin/multipledispatch) doesn't support Python type annotation syntax and 
[multimethod](https://github.com/coady/multimethod) doesn't support default argument. TensorNeko can do it all.

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

### Miscellaneous

`StringGetter`: Get PyTorch class from string.
```python
import tensorneko as neko
activation = neko.util.get_activation("leakyRelu")()
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

`Timer`: A timer for measuring the time.
```python
from tensorneko.util import Timer
import time

# use as a context manager with single time
with Timer():
    time.sleep(1)

# use as a context manager with multiple segments
with Timer() as t:
    time.sleep(1)
    t.time("sleep A")
    time.sleep(1)
    t.time("sleep B")
    time.sleep(1)

# use as a decorator
@Timer()
def f():
    time.sleep(1)
    print("f")
```

`Singleton`: A decorator to make a class as a singleton. Inspired from Scala/Kotlin.
```python
from tensorneko.util import Singleton

@Singleton
class MyObject:
    def __init__(self):
        self.value = 0

    def add(self, value):
        self.value += value
        return self.value


print(MyObject.value)  # 0
MyObject.add(1)
print(MyObject.value)  # 1
```

Besides, many miscellaneous functions are also provided.


Functions list (in `tensorneko_util`):
- `generate_inf_seq`
- `compose`
- `listdir`
- `with_printed`
- `ifelse`
- `dict_add`
- `as_list`
- `identity`
- `list_to_dict`
- `get_tensorneko_util_path`

Functions list (in `tensorneko`):
- `reduce_dict_by`
- `summarize_dict_by`
- `with_printed_shape`
- `is_bad_num`
- `count_parameters`
