from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Embedding

from ..neko_module import NekoModule


class VectorQuantizer(NekoModule):
    """
    The vector quantization layer.

    Args:
        n_embeddings (``int``): The number of embeddings.
        embedding_dim (``int``): The dimension of embeddings.
        beta (``float``): The beta value for embedding loss.

    Attributes:
        embedding (:class:`~torch.nn.Embedding`): The embedding PyTorch layer.

    References:
        Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. Retrieved 18 May 2022,
        from https://arxiv.org/abs/1711.00937

    """

    def __init__(self, n_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embedding = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = Embedding(self.n_embedding, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embedding, 1 / self.n_embedding)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_index = self.predict_indexes(z)

        z_q = self.indexing(z_index)
        z_q = z + (z_q - z).detach()

        embedding_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        return z_q, embedding_loss

    def predict_indexes(self, z: Tensor) -> Tensor:
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # calculate distance from each z to each entry
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))

        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.view(z.shape[:3])

    def indexing(self, z_index: Tensor) -> Tensor:
        # z_index: (b, h, w)
        z_q = self.embedding(z_index)  # (b, h, w, c)
        return rearrange(z_q, "b h w c -> b c h w")
