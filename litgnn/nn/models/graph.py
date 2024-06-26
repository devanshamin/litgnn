import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import pool as global_pooling
from torch_geometric.typing import Adj

from litgnn.nn import models


class GraphLevelGNN(nn.Module):

    def __init__(
        self,
        model_cls: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_conv_layers: int,
        num_ffn_layers: int = 1,
        dropout: float = 0.0,
        pooling_func_name: str = "global_mean_pool",
        **model_kwargs
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
            dropout=dropout,
            **model_kwargs
        )
        ffns = []
        if num_ffn_layers > 1:
            for _ in range(num_ffn_layers - 1):
                ffns.extend([
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                ])
        ffns.extend([nn.Dropout(p=dropout), nn.Linear(hidden_channels, out_channels)])
        self.seq_out = nn.Sequential(*ffns)

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Tensor
    ) -> Tensor:

        out = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        if out.shape[0] == x.shape[0]:
            # out -> num_atoms x hidden_channels
            # Apply global pooling on `out` to get molecule embeddings
            out = self.pooling(out, batch) # batch_size x hidden_channels

        return self.seq_out(out) # batch_size x out_channels
