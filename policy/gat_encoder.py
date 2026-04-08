"""GAT graph encoder for LEO routing."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder that produces per-node embeddings.

    Input: PyG Data with node features and edge index
    Output: per-node embeddings [N, hidden_dim]
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        edge_feat_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            # GAT with edge_dim support
            self.gat_layers.append(
                GATConv(
                    in_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """Encode graph into per-node embeddings.

        Args:
            data: PyG Data with x [N, node_feat_dim], edge_index [2, E],
                  edge_attr [E, edge_feat_dim]

        Returns:
            Node embeddings [N, hidden_dim]
        """
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, data.edge_index, edge_attr=edge_attr)
            x = norm(x + residual)
            x = torch.relu(x)
            x = self.dropout(x)

        return x
