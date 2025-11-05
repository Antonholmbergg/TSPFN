import torch
import torch.nn.functional as F  # noqa
from torch import nn


class Attention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        embedding_dim_q (int): Size of embedding dim for query
        embedding_dim_k (int): Size of embedding dim for key
        embedding_dim_v (int): Size of embedding dim for value
        total_embedding_dim (int): Total embedding dim of combined heads post input projection. Each head
            has dim total_embedding_dim // n_heads
            n_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        embedding_dim_q: int,
        embedding_dim_k: int,
        embedding_dim_v: int,
        total_embedding_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self._qkv_same_embed_dim = embedding_dim_q == embedding_dim_k == embedding_dim_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(embedding_dim_q, total_embedding_dim * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(embedding_dim_q, total_embedding_dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(embedding_dim_k, total_embedding_dim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(embedding_dim_v, total_embedding_dim, bias=bias, **factory_kwargs)
        output_dim = embedding_dim_q
        self.out_proj = nn.Linear(total_embedding_dim, output_dim, bias=bias, **factory_kwargs)
        if total_embedding_dim % n_heads != 0:
            msg = "Embedding dim is not divisible by n_heads"
            raise ValueError(msg)
        self.dim_head = total_embedding_dim // n_heads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal=True,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``embedding_dim_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: True

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, embedding_dim_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )
        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, total_embedding_dim) -> (N, L_t, n_heads, dim_head) -> (N, n_heads, L_t, dim_head)
        query = query.unflatten(-1, [self.n_heads, self.dim_head]).transpose(1, 2).contiguous()
        # (N, L_s, total_embedding_dim) -> (N, L_s, n_heads, dim_head) -> (N, n_heads, L_s, dim_head)
        key = key.unflatten(-1, [self.n_heads, self.dim_head]).transpose(1, 2).contiguous()
        # (N, L_s, total_embedding_dim) -> (N, L_s, n_heads, dim_head) -> (N, n_heads, L_s, dim_head)
        value = value.unflatten(-1, [self.n_heads, self.dim_head]).transpose(1, 2).contiguous()
        if query.is_contiguous() is False:
            print("query is not contigous")
        if key.is_contiguous() is False:
            print("key is not contigous")
        if value.is_contiguous() is False:
            print("value is not contigous")

        # Step 3. Run SDPA
        # (N, n_heads, L_t, dim_head)
        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout, is_causal=is_causal)
        # (N, n_heads, L_t, dim_head) -> (N, L_t, n_heads, dim_head) -> (N, L_t, total_embedding_dim)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, total_embedding_dim) -> (N, L_t, output_dim)
        return self.out_proj(attn_output)


if __name__ == "__main__":
    attention = Attention(
        embedding_dim_q=64,
        embedding_dim_k=64,
        embedding_dim_v=64,
        total_embedding_dim=512,
        n_heads=8,
        dropout=0.0,
    )
    attention = torch.compile(attention)
    inp = torch.nested.nested_tensor(
        tensor_list=[torch.rand(100, 64), torch.rand(10, 64), torch.rand(1000, 64)], layout=torch.jagged
    )
    # inp = torch.rand(2, 1000, 64)
    out = attention(inp, inp, inp)
    print(inp, out)
