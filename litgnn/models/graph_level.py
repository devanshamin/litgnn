from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import pool as global_pooling
from torch_geometric.typing import Adj

from litgnn import models


class GraphLevelGNN(nn.Module):

    def __init__(
        self,
        model_cls: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_conv_layers: int,
        communicator_name: str,
        num_ffn_layers: int = 1,
        dropout: float = 0.0,
        pooling_func_name: str = "global_add_pool"
    ) -> None:
        
        super().__init__()
        assert hasattr(global_pooling, pooling_func_name), "Invalid pooling function!"
        self.pooling = getattr(global_pooling, pooling_func_name)

        assert hasattr(models, model_cls), f"Model class '{model_cls}' not found!"
        self.model = getattr(models, model_cls)(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=num_conv_layers,
            communicator_name=communicator_name,
            dropout=dropout
        )
        ffns = []
        if num_ffn_layers > 1:
            for _ in range(num_ffn_layers - 1):
                ffns.extend([
                    nn.ReLU(), 
                    nn.Dropout(p=dropout), 
                    nn.Linear(hidden_channels, hidden_channels)
                ])
        ffns.extend([nn.Dropout(p=dropout), nn.Linear(hidden_channels, out_channels)])
        self.seq = nn.Sequential(*ffns)

    def forward(
        self, 
        x: Tensor, 
        edge_index: Adj, 
        edge_attr: Tensor,
        batch: Tensor
    ) -> Tensor:

        x = self.model(x, edge_index, edge_attr) # num_atoms x hidden_channels 
        # global_pooling on x to get molecule embeddings
        x = self.pooling(x, batch) # batch_size x hidden_channels
        return self.seq(x) # batch_size x out_channels
