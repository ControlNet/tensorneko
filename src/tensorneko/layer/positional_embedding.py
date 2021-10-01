from fn import F, _
from torch import Tensor, zeros
from torch.nn import Parameter, Dropout

from ..neko_module import NekoModule
from ..util import Shape


class PositionalEmbedding(NekoModule):
    """
    Trainable positional embedding for transformer/self-attention layer.

    The trainable positional embedding is used in Transformer (Vaswani, et al., 2017).

    Args:
        input_shape (:class:`~tensorneko.util.type.Shape`): The input shape of the sequence.

        dropout_rate (``int``, optional): The dropout rate after the positional embedding. Default ``0``.

    Attributes:
        emb (:class`~torch.nn.Parameter`): The trainable positional embedding parameters.

        dropout (:class`~torch.nn.Dropout`): The PyTorch dropout module in the layer.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
        Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

    """

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__()
        self.emb = Parameter(zeros(1, *input_shape))
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.
        if self.use_dropout:
            self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> _ + self.emb
        if self.use_dropout:
            f = f >> self.dropout
        return f(x)
