"""GNN routing policy with cost-to-go prediction head."""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from typing import Dict, Tuple

from .gat_encoder import GATEncoder
from .cost_to_go import CostToGoHead


class GNNRoutingPolicy(nn.Module):
    """Routing policy that predicts cost-to-go V(i,j) and derives next-hop greedily.

    next_hop(i,j) = argmin_k [ edge_delay(i,k) + V(nbr_k, j) ]
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        rank: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = GATEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
        )
        self.cost_head = CostToGoHead(embed_dim=hidden_dim, rank=rank)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass: predict cost-to-go.

        Returns:
            cost: [N, N] predicted cost-to-go
        """
        node_emb = self.encoder(data)  # [N, D]
        cost = self.cost_head(node_emb)  # [N, N]
        return cost

    def get_routing_table(
        self,
        data: Data,
        neighbor_table: torch.Tensor,
        neighbor_delays: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next-hop slot table via greedy cost-to-go minimization.

        Args:
            data: PyG Data
            neighbor_table: [N, K] int64
            neighbor_delays: [N, K] float
            neighbor_mask: [N, K] float

        Returns:
            slot_table: [N, N] int64
            cost: [N, N] float predicted cost-to-go
        """
        cost = self.forward(data)
        slot_table = self.cost_head.derive_nexthop(
            cost, neighbor_table, neighbor_delays, neighbor_mask,
        )
        return slot_table, cost

    @staticmethod
    def obs_to_pyg(obs: Dict[str, np.ndarray], device: torch.device):
        """Convert env observation to tensors.

        Returns:
            data: PyG Data
            neighbor_table: [N, K] int64
            neighbor_delays: [N, K] float
            neighbor_mask: [N, K] float
        """
        num_edges = int(obs["num_edges"])

        data = Data(
            x=torch.tensor(obs["node_features"], dtype=torch.float32, device=device),
            edge_index=torch.tensor(
                obs["edge_index"][:, :num_edges], dtype=torch.long, device=device,
            ),
            edge_attr=torch.tensor(
                obs["edge_attr"][:num_edges], dtype=torch.float32, device=device,
            ),
        )
        neighbor_table = torch.tensor(obs["neighbor_table"], dtype=torch.long, device=device)
        neighbor_delays = torch.tensor(obs["neighbor_delays"], dtype=torch.float32, device=device)
        neighbor_mask = torch.tensor(obs["action_mask"], dtype=torch.float32, device=device)

        return data, neighbor_table, neighbor_delays, neighbor_mask
