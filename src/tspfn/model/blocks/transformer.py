from torch import Tensor, nn

from tspfn.model.blocks.feed_forward import SwiGLUNet
from tspfn.model.blocks.multihead_attention import Attention
from tspfn.model.blocks.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim_q: int,
        embedding_dim_k: int,
        embedding_dim_v: int,
        total_embedding_dim: int,
        n_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        *,
        is_causal: bool = True,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.attention = Attention(
            embedding_dim_q=embedding_dim_q,
            embedding_dim_k=embedding_dim_k,
            embedding_dim_v=embedding_dim_v,
            total_embedding_dim=total_embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = SwiGLUNet(external_dim=embedding_dim_q, internal_dim=ff_hidden_dim)
        self.ffn_norm = RMSNorm(embedding_dim_q, norm_eps)
        self.attention_norm = RMSNorm(embedding_dim_q, norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        """for now the q/k/v will always be the same"""
        x = self.attention_norm(x)
        h = x + self.attention(query=x, key=x, value=x, is_causal=self.is_causal)
        return h + self.feed_forward(self.ffn_norm(h))
