from abc import ABC, abstractmethod

import torch
from torch import Tensor, zeros
from torch.nn import Parameter, Dropout

from ..neko_module import NekoModule
from ..util import Shape, F, _


class PositionalEmbedding(NekoModule):
    """
    Trainable positional embedding for transformer/self-attention layer.

    The trainable positional embedding is used in Transformer (Vaswani, et al., 2017).

    Args:
        input_shape (:class:`~tensorneko.util.type.Shape`): The input shape of the sequence.
            Format: (Length of sequence, Embedding dimension).

        dropout_rate (``int``, optional): The dropout rate after the positional embedding. Default ``0``.

    Attributes:
        emb (:class`~torch.nn.Parameter`): The trainable positional embedding parameters.

        dropout (:class`~torch.nn.Dropout`): The PyTorch dropout module in the layer.

        trainable (``bool``): The trainable flag.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
        Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

    """

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = Parameter(zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.
        if self.use_dropout:
            self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> _ + self.emb
        if self.use_dropout:
            f = f >> self.dropout
        return f(x)

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value


class AbstractNonTrainablePositionalEmbedding(PositionalEmbedding, ABC):
    """
    Abstract class for non-trainable positional embedding. Inherit this class to define your own positional embedding.

    The method `make_embedding` should be defined to generate the positional embedding as a Tensor with shape same to
        argument `input_shape`.

    Args:
        input_shape (:class:`~tensorneko.util.type.Shape`): The input shape of the sequence.
            Format: (Length of sequence, Embedding dimension).

        dropout_rate (``int``, optional): The dropout rate after the positional embedding. Default ``0``.

    Attributes:
        emb (:class`~torch.nn.Parameter`): The trainable positional embedding parameters.

        dropout (:class`~torch.nn.Dropout`): The PyTorch dropout module in the layer.

    """

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    @abstractmethod
    def make_embedding(self) -> Tensor:
        pass


class SinCosPositionalEmbedding(AbstractNonTrainablePositionalEmbedding):
    """2D sine-cosine position embedding"""

    def make_embedding(self) -> Tensor:
        """
        Generate the positional embedding as a Tensor with shape same to argument `input_shape`.

        Pytorch version of
            https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31

        Returns:
            :class:`~torch.Tensor`: The positional embedding.

        References:
            Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
            Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

        """

        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode='trunc') / d_hid)

        sinusoid_table = torch.stack([get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()
