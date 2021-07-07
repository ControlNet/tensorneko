from typing import Optional

import torch
from fn import F
from torch import Tensor, zeros
from torch.nn import Parameter, LayerNorm, Linear, MultiheadAttention, GELU, Identity, ModuleList

from . import MLP, ResidualBlock
from ..neko_module import NekoModule
from ..layer import PositionalEmbedding, Concatenate
from ..util import ModuleFactory, compose, Shape


class AttentionModule(NekoModule):
    """
    The AttentionModule is the layer taking the input and calculate Q, K and V and feed them into MultiHeadAttention
    layers.

    input -> (Q, K, V) -> MultiHeadAttention

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
        self.q_linear = Linear(embed_dim, self.kdim)
        self.k_linear = Linear(embed_dim, self.kdim)
        self.v_linear = Linear(embed_dim, self.vdim)
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim,
            batch_first=True)
        self.return_attention_weights = return_attention_weights

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> (map, lambda linear: linear(x)) >> (lambda xs: self.attention(*xs))
        x, weight = f([self.q_linear, self.k_linear, self.v_linear])
        return (x, weight) if self.return_attention_weights else x


class TransformerEncoderBlock(NekoModule):
    """
    The TransformerEncoderBlock is a block in Transformer encoder which is proposed by Vaswani, et al. (2017).

    This TransformerEncoderBlock contains the multi-head attention module and the feed-forward module.

    input -> concat cls token -> add positional encoding -> AttentionModule -> FFN

    Args:
        input_shape (:class:`~tensorneko.util.type.Shape`): The shape of input sequence (N, D). N for length of
            sequence. D for embedding dimension.

        num_heads (``int``): Parallel attention heads.

        has_cls_token (``bool``, optional): The input will concat to a cls token if True. Default ``False``.

        has_pos_encoding (``bool``, optional): The input will add positional encoding if True. Default ``False``.

        linear_drop (``float``, optional): The dropout rate for linear layers. Default ``0.5``.

        attention_drop (``float``, optional): The dropout rate for attention layers. Default ``0.5``.

        build_normalization (``() -> torch.nn.Module``, optional): The normalization builder function for the block.
            Default :class:`~torch.nn.LayerNorm`.

        mlp_ratio (``float``, optional): The MLP ratio, which is the multiplication of hidden layers in FFN. e.g, if the
            embedding is 1024, and the mlp_ratio is 4.0. The FFN will be 1024 -> 4096 -> 1024. Default ``4.0``.

        build_mlp_activation (``() -> torch.nn.Module``, optional): The activation builder for FFN.
            Default :class:`~torch.nn.GELU`.

    Attributes:
        cls_token (:class:`~torch.nn.Parameter`): The trainable cls token parameters.
        pos_emb_layer (:class:`~tensorneko.layer.PositionalEmbedding`): The positional embedding layer.
        attn_module (:class:`~tensorneko.module.ResidualBlock`): The attention module with residual connection.
        feedforward_module (:class:`~tensorneko.module.ResidualBlock`): The MLP module with residual connection.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
        Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

    """

    def __init__(self, input_shape: Shape, num_heads: int, has_cls_token: bool = False,
        has_pos_encoding: bool = True,
        linear_drop: float = 0.5, attention_drop: float = 0.5,
        build_normalization: Optional[ModuleFactory] = LayerNorm,
        mlp_ratio: float = 4.0, build_mlp_activation: Optional[ModuleFactory] = GELU
    ):
        super().__init__()
        # n: number of patches, d: embedding dimension
        n, d = input_shape
        # num of head
        self.num_head = num_heads
        # positional embedding
        self.has_cls_token = has_cls_token
        if self.has_cls_token:
            # prepare for class token
            self.cls_token = Parameter(zeros(1, d))
            self.token_concat = Concatenate(dim=0)
            input_shape = (n + 1, d)
        self.has_pos_encoding = has_pos_encoding
        if self.has_pos_encoding:
            self.pos_emb_layer = PositionalEmbedding(input_shape=input_shape, dropout_rate=linear_drop)

        # set normalization builder
        if build_normalization is LayerNorm:
            build_normalization = F(LayerNorm, input_shape)
        elif build_normalization is None:
            build_normalization = Identity

        # multi-head attention module with residual connection and normalization
        self.attn_module = ResidualBlock(
            sub_module_layers=(build_normalization, F(AttentionModule, d, self.num_head, attention_drop)),
            tail_module_layers=None
        )

        self.feedforward_module = ResidualBlock((
            build_normalization,
            F(MLP, [d, int(d * mlp_ratio), d],
                build_activation=build_mlp_activation, dropout_rate=linear_drop
            ))
        )

    def forward(self, x: Tensor) -> Tensor:
        f = F()
        if self.has_cls_token:
            f = f >> (map, lambda tokens: self.token_concat([self.cls_token, tokens])) >> list >> torch.stack
        if self.has_pos_encoding:
            f = f >> self.pos_emb_layer
        f = f >> self.attn_module >> self.feedforward_module
        return f(x)


class TransformerEncoder(NekoModule):
    """
    The TransformerEncoder repeatedly generate :class:`TransformerEncoderBlock` with specified times.

    Args:
        input_shape (:class:`~tensorneko.util.type.Shape`): The shape of input sequence (N, D). N for length of
            sequence. D for embedding dimension.

        num_heads (``int``): Parallel attention heads.

        has_cls_token (``bool``, optional): The input will concat to a cls token if True. Default ``False``.

        linear_drop (``float``, optional): The dropout rate for linear layers. Default ``0.5``.

        attention_drop (``float``, optional): The dropout rate for attention layers. Default ``0.5``.

        build_normalization (``() -> torch.nn.Module``, optional): The normalization builder function for the block.
            Default :class:`~torch.nn.LayerNorm`.

        mlp_ratio (``float``, optional): The MLP ratio, which is the multiplication of hidden layers in FFN. e.g, if the
            embedding is 1024, and the mlp_ratio is 4.0. The FFN will be 1024 -> 4096 -> 1024. Default ``4.0``.

        build_mlp_activation (``() -> torch.nn.Module``, optional): The activation builder for FFN.
            Default :class:`~torch.nn.GELU`.

        pos_encoding (``str``, optional): The option of where you want to add positional encoding. ``"all"`` for all
            blocks. ``"first"`` for first block only. ``"none"`` for no adding. Default ``"all"``.

        repeat (``int``, optional): The repeat time of TransformerEncoderBlock.

    Attributes:
        blocks (:class:`~torch.nn.ModuleList`): The blocks list in the module.

    """

    def __init__(self, input_shape: Shape, num_heads: int, has_cls_token: bool = False, linear_drop: float = 0.5,
        attention_drop: float = 0.5, build_normalization: Optional[ModuleFactory] = LayerNorm, mlp_ratio: float = 4.0,
        build_mlp_activation: Optional[ModuleFactory] = GELU, pos_encoding: str = "all",
        repeat: int = 1
    ):
        super().__init__()
        if pos_encoding == "all":
            has_pos_encoding = [True] * repeat
        elif pos_encoding == "first":
            has_cls_token = [True] + (repeat - 1) * [False]
        elif pos_encoding == "none":
            has_cls_token = [False] * repeat

        def build_block(i):
            return TransformerEncoderBlock(
                input_shape, num_heads, has_cls_token,
                has_pos_encoding[i],
                linear_drop, attention_drop,
                build_normalization, mlp_ratio, build_mlp_activation
            )

        self.blocks = ModuleList([build_block(i) for i in range(repeat)])
        self.repeat = repeat

    def forward(self, x: Tensor) -> Tensor:
        return compose(self.blocks)(x)
