from typing import Optional, Union, Tuple

import torch.nn.functional
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import MultiheadAttention, Linear, Conv2d

from ..neko_module import NekoModule
from ..util import F


class SeqAttention(NekoModule):
    """
    The SeqAttention is the layer taking the input and calculate Q, K and V and feed them into MultiHeadAttention
    layer.

    input (Batch, Temporal, Dim) -> (Q, K, V) -> MultiHeadAttention

    The MultiHeadAttention is proposed by Vaswani, et al. (2017).

    Args:
        embed_dim (``int``): The embedding dim of input sequence.

        num_heads (``bool``): Parallel attention heads.

        dropout (``float``, optional): A Dropout layer on attn_output_weights. Default: 0.0.

        bias (``bool``, optional): Add bias as module parameter. Default: True.

        add_bias_kv (``bool``, optional): Add bias to the key and value sequences at dim=0.

        add_zero_attn (``bool``, optional): Add a new batch of zeros to the key and value sequences at dim=1.

        kdim (``int``, optional): Total number of features in key. Default: None.

        vdim (``int``, optional): Total number of features in value. Default: None.

        return_attention_weights (``bool``, optional): Return both value output and attention weights if True else
            return attention value output only.

    Attributes:
        q_linear (:class:`~torch.nn.Linear`): The PyTorch Linear layer for calculating Q.
        k_linear (:class:`~torch.nn.Linear`): The PyTorch Linear layer for calculating K.
        v_linear (:class:`~torch.nn.Linear`): The PyTorch Linear layer for calculating V.
        attention (:class:`~torch.nn.MultiheadAttention`): The PyTorch MultiheadAttention layer of this module.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
        Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True,
        add_bias_kv: bool = False, add_zero_attn: bool = False, kdim: Optional[int] = None, vdim: Optional[int] = None,
        return_attention_weights: bool = False
    ):
        super().__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.q_linear = Linear(embed_dim, embed_dim)
        self.k_linear = Linear(embed_dim, self.kdim)
        self.v_linear = Linear(embed_dim, self.vdim)
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim,
            batch_first=True)
        self.return_attention_weights = return_attention_weights

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        f = F() >> (map, lambda linear: linear(x)) >> (lambda xs: self.attention(*xs))
        x, weight = f([self.q_linear, self.k_linear, self.v_linear])
        return (x, weight) if self.return_attention_weights else x


class ImageAttention(NekoModule):
    """
    The ImageAttentionModule is the layer taking the input and calculate Q, K and V and feed them into
    MultiHeadAttention layers.

    input (Batch, Channel, Height, Width) -> (Q, K, V) -> MultiHeadAttention

    The main difference of this class to :class:`~tensorneko.layer.attention.SeqAttention` is the input is a batch of
    feature maps.

    The MultiHeadAttention is proposed by Vaswani, et al. (2017).

    Args:
        embed_dim (``int``): The embedding dim of input sequence.

        num_heads (``bool``): Parallel attention heads.

        dropout (``float``, optional): A Dropout layer on attn_output_weights. Default: 0.0.

        bias (``bool``, optional): Add bias as module parameter. Default: True.

        add_bias_kv (``bool``, optional): Add bias to the key and value sequences at dim=0.

        add_zero_attn (``bool``, optional): Add a new batch of zeros to the key and value sequences at dim=1.

        kdim (``int``, optional): Total number of features in key. Default: None.

        vdim (``int``, optional): Total number of features in value. Default: None.

        return_attention_weights (``bool``, optional): Return both value output and attention weights if True else
            return attention value output only.

    Attributes:
        q_linear (:class:`~torch.nn.Conv2d`): The PyTorch Conv2d layer for calculating Q.
        k_linear (:class:`~torch.nn.Conv2d`): The PyTorch Conv2d layer for calculating K.
        v_linear (:class:`~torch.nn.Conv2d`): The PyTorch Conv2d layer for calculating V.
        attention (:class:`~torch.nn.MultiheadAttention`): The PyTorch MultiheadAttention layer of this module.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
        Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True,
        add_bias_kv: bool = False, add_zero_attn: bool = False, kdim: Optional[int] = None, vdim: Optional[int] = None,
        return_attention_weights: bool = False
    ):
        super().__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.q_linear = Conv2d(embed_dim, embed_dim, 1)
        self.k_linear = Conv2d(embed_dim, self.kdim, 1)
        self.v_linear = Conv2d(embed_dim, self.vdim, 1)
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim,
            batch_first=True)
        self.return_attention_weights = return_attention_weights

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        _, _, h, w = x.size()
        f = F() >> (map, lambda linear: linear(x)) >> (map, Rearrange("b c h w -> b (h w) c")) \
            >> (lambda xs: self.attention(*xs))
        x, weight = f([self.q_linear, self.k_linear, self.v_linear])
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return (x, weight) if self.return_attention_weights else x
