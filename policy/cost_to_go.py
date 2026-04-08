"""Cost-to-go prediction head for routing via distance field distillation."""

import torch
import torch.nn as nn


class CostToGoHead(nn.Module):
    """Predicts V(i,j) = estimated shortest-path cost from node i to destination j.

    Next-hop is derived greedily: argmin_k [edge_delay(i,k) + V(nbr_k, j)]

    Uses MLP on concatenated src/dst embeddings for higher capacity than bilinear.
    """

    def __init__(self, embed_dim: int = 128, rank: int = 64, dist_mean: float = 60.0):
        super().__init__()
        self.src_proj = nn.Sequential(
            nn.Linear(embed_dim, rank),
            nn.LayerNorm(rank),
            nn.ReLU(),
        )
        self.dst_proj = nn.Sequential(
            nn.Linear(embed_dim, rank),
            nn.LayerNorm(rank),
            nn.ReLU(),
        )
        # MLP scorer on [src; dst; src*dst]
        self.scorer = nn.Sequential(
            nn.Linear(rank * 3, rank * 2),
            nn.ReLU(),
            nn.Linear(rank * 2, rank),
            nn.ReLU(),
            nn.Linear(rank, 1),
        )
        # Initialize final layer bias to mean distance so predictions start reasonable
        with torch.no_grad():
            self.scorer[-1].bias.fill_(dist_mean)
            # Small weights so initial variation is small around the mean
            self.scorer[-1].weight.mul_(0.01)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Predict cost-to-go for all (src, dst) pairs.

        Args:
            node_emb: [N, D] node embeddings from GAT encoder

        Returns:
            cost: [N, N] predicted cost-to-go (in ms scale)
        """
        N = node_emb.shape[0]
        src = self.src_proj(node_emb)  # [N, R]
        dst = self.dst_proj(node_emb)  # [N, R]

        src_exp = src.unsqueeze(1).expand(N, N, -1)
        dst_exp = dst.unsqueeze(0).expand(N, N, -1)

        pair_feat = torch.cat([src_exp, dst_exp, src_exp * dst_exp], dim=-1)  # [N, N, 3R]
        cost = self.scorer(pair_feat).squeeze(-1)  # [N, N]
        return cost

    def derive_nexthop(
        self,
        cost: torch.Tensor,
        neighbor_table: torch.Tensor,
        edge_delays: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Derive next-hop table greedily from cost-to-go predictions."""
        nbr_idx = neighbor_table.clamp(min=0)  # [N, K]
        nbr_cost = cost[nbr_idx]  # [N, K, N]
        candidate = edge_delays.unsqueeze(-1) + nbr_cost  # [N, K, N]
        inv_mask = (neighbor_mask == 0).unsqueeze(-1)
        candidate = candidate.masked_fill(inv_mask, float('inf'))
        slot_table = candidate.argmin(dim=1)  # [N, N]
        return slot_table
