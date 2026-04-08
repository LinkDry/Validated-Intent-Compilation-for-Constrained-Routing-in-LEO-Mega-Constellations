"""Bilinear destination-conditioned actor with separate src/dst projections."""

import torch
import torch.nn as nn


class BilinearActor(nn.Module):
    """Produces [N, N, K] next-hop logits via bilinear product.

    logits[i, j, k] = src_proj(emb[i]) @ W_k @ dst_proj(emb[j]) + bias[k]

    Separate projections break role aliasing between router and destination.
    """

    def __init__(self, embed_dim: int = 64, num_actions: int = 4):
        super().__init__()
        self.src_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dst_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W = nn.Parameter(torch.randn(num_actions, embed_dim, embed_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_actions))

    def forward(self, node_emb: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        """Compute per-destination next-hop logits.

        Args:
            node_emb: [N, D] node embeddings from GAT encoder
            neighbor_mask: [N, K] float, 1.0 if valid, 0.0 otherwise

        Returns:
            logits: [N, N, K] next-hop logits
        """
        src = self.src_proj(node_emb)  # [N, D]
        dst = self.dst_proj(node_emb)  # [N, D]

        logits = torch.einsum('id, kdf, jf -> ijk', src, self.W, dst)
        logits = logits + self.bias

        mask = neighbor_mask.unsqueeze(1)  # [N, 1, K]
        logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits
