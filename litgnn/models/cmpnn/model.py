from typing import Literal, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.typing import Adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MaxAggregation, SumAggregation
from torch_geometric.nn.inits import reset


class CMPNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        communicator_name: str,
        dropout: float = 0.0,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        # Additional conv layer for computing the final atom embedding
        self.num_layers = num_layers + 1
        self.communicator_name = communicator_name

        # Can be transformed into pre-tower ffn layers similar to `GIN`
        self.atom_input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU()
        )
        self.bond_input_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU()
        )
        
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers): 
            self.convs.append(
                GCNEConv(hidden_channels, hidden_channels, communicator_name, dropout)
            )

        self.communicator = NodeEdgeMessageCommunicator(
            name=communicator_name,
            hidden_channels=hidden_channels
        )

        # Can be transformed into post-tower ffn layers similar to `GIN`
        self.lin = nn.Linear(hidden_channels * 2, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        
        reset(self.atom_input_proj)
        reset(self.bond_input_proj)
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self, 
        x: Tensor, 
        edge_index: Adj, 
        edge_attr: Tensor
    ) -> Tensor:
        
        x_proj = self.atom_input_proj(x)
        h_atom = x_proj.clone()
        h_bond = self.bond_input_proj(edge_attr)
        
        for layer in self.convs[:-1]:
            h_atom, h_bond = layer(x=h_atom, edge_attr=h_bond, edge_index=edge_index)

        # Kth layer aggregation
        aggr_atom_message, _ = self.convs[-1](h_atom, h_bond, edge_index)
        h_atom = self.communicator(aggr_atom_message, h_atom)
        h_atom = self.lin(torch.cat([h_atom, x_proj], 1)) # Skip connection
        return h_atom


class GCNEConv(MessagePassing):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        communicator_name: str, 
        dropout: float = 0.0
    ) -> None:
        
        super().__init__(
            aggr=[SumAggregation(), MaxAggregation()], 
            aggr_kwargs=dict(mode='message_booster'),
            flow="target_to_source"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.communicator_name = communicator_name
        self.communicator = NodeEdgeMessageCommunicator(
            name=communicator_name,
            hidden_channels=in_channels
        )
        self.seq = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the module."""

        super().reset_parameters()
        reset(self.communicator)
        reset(self.seq)

    def forward(
        self, 
        x: Tensor, 
        edge_attr: Tensor, 
        edge_index: Adj
    ) -> Tuple[Tensor, Tensor]:

        x = self.propagate(
            edge_index, 
            x=x, 
            edge_attr=edge_attr, 
            # Aggregation is done on the `edge_attr` based on `edge_index_i`
            # So the first dimension is not always equal to num_atoms 
            # i.e., edge_index_i.unique().shape != x.size(0)
            # The output should be of shape (num_atoms x hidden_channels)
            # `x.size(0)` will get assigned to `dim_size` that is passed to the
            # `aggregate` method
            size=[x.size(0), None] 
        )
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr
    
    def message(self, edge_attr: Tensor) -> Tensor:

        return edge_attr
    
    def update(self, message: Tensor, x: Tensor) -> Tensor:

        return self.communicator(message, x)

    def edge_update(
        self, 
        x: Tensor, 
        edge_attr: Tensor, 
        edge_index_i: Tensor,
        edge_index_j: Tensor
    ) -> Tensor:
        
        # For example,
        # Atom_0 -[Bond_0]-> Atom_1
        # Atom_0 <-[Bond_1]- Atom_1
        # Bond_0 = Atom_0 - Bond_1
        bond_embed = x[edge_index_i] - edge_attr[edge_index_j]
        return self.seq(bond_embed)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, '
            f"communicator_name='{self.communicator_name}')"
        )


class NodeEdgeMessageCommunicator(nn.Module):

    def __init__(
        self, 
        name: Literal["inner_product", "gru", "mlp"], 
        hidden_channels: int
    ) -> None:
        
        super().__init__()
        assert name in ("inner_product", "gru", "mlp"), f"Invalid communicator '{name}'!"
        self.name = name
        self.hidden_channels = hidden_channels
        self.communicator = None

        if name == "gru":
            self.communicator = nn.GRUCell(hidden_channels, hidden_channels)
        elif name == "mlp":
            self.communicator = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU()
            )
    
    def forward(self, message: Tensor, hidden_state: Tensor) -> Tensor:

        if self.name == "inner_product":
            # print(hidden_state.shape, message.shape)
            out = hidden_state * message
        elif self.name == "gru":
            out = self.communicator(hidden_state, message)
        elif self.name == "mlp":
            message = torch.cat((hidden_state, message), dim=1)
            out = self.communicator(message)

        return out

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.hidden_channels}, '
            f"name='{self.name}')"
        )
