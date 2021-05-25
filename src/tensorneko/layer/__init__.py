# import from modules
from .conv2d import Conv2d
from .linear import Linear
from .concatenate import Concatenate
from .patching import Patching, PatchEmbedding2d
from .positional_embedding import PositionalEmbedding


# import from other libraries
from torch.nn import MultiheadAttention as _MultiheadAttention

Attention = _MultiheadAttention
