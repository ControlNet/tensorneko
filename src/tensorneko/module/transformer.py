from typing import Iterator, Optional

from fn import F
from torch import Tensor, zeros
from torch.nn import Module, Parameter, LayerNorm, Sequential, Linear, MultiheadAttention, GELU, Identity, ModuleList

from . import MLP, ResidualBlock
from ..layer import PositionalEmbedding, Concatenate
from ..util import ModuleFactory, compose


class AttentionModule(Module):
    """
    The attention module.
    x -> QKV -> MultiheadAttention
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
        add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
        return_attention_weights=False
    ):
        super().__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.q_linear = Linear(embed_dim, self.kdim)
        self.k_linear = Linear(embed_dim, self.kdim)
        self.v_linear = Linear(embed_dim, self.vdim)
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.return_attention_weights = return_attention_weights

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> (map, lambda linear: linear(x)) >> list >> (lambda xs: self.attention(*xs))
        x, weight = f([self.q_linear, self.k_linear, self.v_linear])
        return (x, weight) if self.return_attention_weights else x


class TransformerEncoderBlock(Module):

    def __init__(self, input_shape: Iterator[int], num_head: int, has_cls_token: bool = False,
        linear_drop: float = 0.5, attention_drop: float = 0.5,
        build_normalization: Optional[ModuleFactory] = LayerNorm,
        mlp_ratio: int = 4, build_mlp_activation: Optional[ModuleFactory] = GELU
    ):
        super().__init__()
        # n: number of patches, d: embedding dimension
        n, d = input_shape
        # num of head
        self.num_head = num_head
        # positional embedding
        self.has_cls_token = has_cls_token
        if self.has_cls_token:
            # prepare for class token
            self.cls_token = Parameter(zeros(1, 1, d))
            self.token_concat = Concatenate(dim=1)
            input_shape = (n + 1, d)
        self.pos_emb_layer = PositionalEmbedding(input_shape=input_shape, drop_rate=linear_drop)

        # set normalization builder
        if build_normalization is LayerNorm:
            build_normalization = F(LayerNorm, input_shape)
        elif build_normalization is None:
            build_normalization = Identity

        # multihead attention module with residual connection and normalization
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
            f = f >> (lambda tokens: self.token_concat([self.cls_token, tokens]))
        f = f >> self.pos_emb_layer >> self.attn_module >> self.feedforward_module
        return f(x)


class TransformerEncoder(Module):

    def __init__(self, input_shape: Iterator[int], num_head: int, has_cls_token: bool = False, linear_drop: float = 0.5,
        attention_drop: float = 0.5, build_normalization: Optional[ModuleFactory] = LayerNorm, mlp_ratio: int = 4,
        build_mlp_activation: Optional[ModuleFactory] = GELU, repeat: int = 1
    ):
        super().__init__()
        build_block = F(TransformerEncoderBlock, input_shape, num_head, has_cls_token, linear_drop, attention_drop,
            build_normalization, mlp_ratio, build_mlp_activation
        )
        self.blocks = ModuleList([build_block() for _ in range(repeat)])
        self.repeat = repeat

    def forward(self, x: Tensor) -> Tensor:
        return compose(self.blocks)(x)
