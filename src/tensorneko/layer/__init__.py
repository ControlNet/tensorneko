# import from modules
from .concatenate import Concatenate
from .conv2d import Conv2d
from .linear import Linear
from .patching import Patching, PatchEmbedding2d
from .positional_embedding import PositionalEmbedding
from .reshape import Reshape


# import from other libraries
from torch.nn import MultiheadAttention as _MultiheadAttention

Attention = _MultiheadAttention
