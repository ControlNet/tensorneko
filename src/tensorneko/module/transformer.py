from typing import Optional

import torch
from torch import Tensor, zeros
from torch.nn import Parameter, LayerNorm, GELU, Identity, ModuleList

from . import MLP, ResidualBlock
from ..layer import PositionalEmbedding, Concatenate, SeqAttention
from ..neko_module import NekoModule
from ..util import ModuleFactory, compose, Shape, F


class TransformerEncoderBlock(NekoModule):
    """
    The TransformerEncoderBlock is a block in Transformer encoder which is proposed by Vaswani, et al. (2017).

    This TransformerEncoderBlock contains the multi-head attention module and the feed-forward module.

    input -> concat cls token -> add positional encoding -> AttentionModule -> FFN

    Args:
        input_shape (:class:`~tensorneko.util.type.Shape`): The shape of input sequence (N, D). N for length of
            sequence. D for embedding dimension.

        num_heads (``int``): Parallel attention heads.

        add_cls_token (``bool``, optional): The input will concat to a cls token if True. Default ``False``.

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

    def __init__(self, input_shape: Shape, num_heads: int, add_cls_token: bool = False,
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
        self.add_cls_token = add_cls_token
        if self.add_cls_token:
            # prepare for class token
            self.cls_token = Parameter(zeros(1, d))
            self.token_concat = Concatenate(dim=0)
            input_shape = (n + 1, d)
        else:
            self.cls_token = None
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
            sub_module_layers=(build_normalization, F(SeqAttention, d, self.num_head, attention_drop)),
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
        if self.add_cls_token:
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

        add_cls_token (``bool``, optional): The input will concat to a cls token if True. Default ``False``.

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

    def __init__(self, input_shape: Shape, num_heads: int, add_cls_token: bool = False, linear_drop: float = 0.5,
        attention_drop: float = 0.5, build_normalization: Optional[ModuleFactory] = LayerNorm, mlp_ratio: float = 4.0,
        build_mlp_activation: Optional[ModuleFactory] = GELU, pos_encoding: str = "all",
        repeat: int = 1
    ):
        super().__init__()
        if pos_encoding == "all":
            has_pos_encoding = [True] * repeat
        elif pos_encoding == "first":
            has_pos_encoding = [True] + (repeat - 1) * [False]
        elif pos_encoding == "none":
            has_pos_encoding = [False] * repeat
        else:
            raise ValueError(f"pos_encoding must be one of 'all', 'first', 'none'. But got {pos_encoding}")

        def build_block(i):
            return TransformerEncoderBlock(
                input_shape, num_heads, add_cls_token if i == 0 else False,
                has_pos_encoding[i],
                linear_drop, attention_drop,
                build_normalization, mlp_ratio, build_mlp_activation
            )

        self.blocks = ModuleList([build_block(i) for i in range(repeat)])
        self.repeat = repeat

    def forward(self, x: Tensor) -> Tensor:
        return compose(self.blocks)(x)
