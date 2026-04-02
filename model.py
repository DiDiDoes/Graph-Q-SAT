import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from torch_geometric.data import Data
from torch_geometric.utils import scatter


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int = 1):
    """
    MLP with ReLU activations.
    num_hidden_layers=1 matches the paper's GN core setting.
    """
    layers = []
    last = in_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(nn.ReLU())
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class EdgeModel(nn.Module):
    """
    e'_ij = phi_e([e_ij, v_i, v_j, u_graph_of_edge])
    """
    def __init__(self, edge_in, node_in, global_in, hidden_dim, edge_out, num_hidden_layers=1):
        super().__init__()
        self.mlp = _mlp(
            in_dim=edge_in + 2 * node_in + global_in,
            hidden_dim=hidden_dim,
            out_dim=edge_out,
            num_hidden_layers=num_hidden_layers,
        )
        self.norm = nn.LayerNorm(edge_out)

    def forward(self, src, dst, edge_attr, u_per_edge):
        out = self.mlp(torch.cat([edge_attr, src, dst, u_per_edge], dim=-1))
        return self.norm(out)


class NodeModel(nn.Module):
    """
    v'_i = phi_v([v_i, sum_{k->i} e'_ki, u_graph_of_node])
    """
    def __init__(self, node_in, edge_in, global_in, hidden_dim, node_out, num_hidden_layers=1):
        super().__init__()
        self.mlp = _mlp(
            in_dim=node_in + edge_in + global_in,
            hidden_dim=hidden_dim,
            out_dim=node_out,
            num_hidden_layers=num_hidden_layers,
        )
        self.norm = nn.LayerNorm(node_out)

    def forward(self, x, edge_index, edge_attr, u_per_node):
        row, col = edge_index  # row = source, col = target
        agg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce="sum")
        out = self.mlp(torch.cat([x, agg, u_per_node], dim=-1))
        return self.norm(out)


class GlobalModel(nn.Module):
    """
    u' = phi_u([u, avg_edges(e'), avg_vars(v')])

    The paper specifies:
    - edge -> global: average
    - variable(node) -> global: average
    """
    def __init__(self, global_in, edge_in, node_in, hidden_dim, global_out, num_hidden_layers=1):
        super().__init__()
        self.mlp = _mlp(
            in_dim=global_in + edge_in + node_in,
            hidden_dim=hidden_dim,
            out_dim=global_out,
            num_hidden_layers=num_hidden_layers,
        )
        self.norm = nn.LayerNorm(global_out)

    def forward(self, u, edge_attr, x, batch, edge_batch, var_mask):
        num_graphs = u.size(0)

        edge_aggr = scatter(edge_attr, edge_batch, dim=0, dim_size=num_graphs, reduce="mean")

        # Aggregate only variable nodes to the global, as described in the paper.
        if var_mask.any():
            var_x = x[var_mask]
            var_batch = batch[var_mask]
            node_aggr = scatter(var_x, var_batch, dim=0, dim_size=num_graphs, reduce="mean")
        else:
            node_aggr = x.new_zeros(num_graphs, x.size(-1))

        out = self.mlp(torch.cat([u, edge_aggr, node_aggr], dim=-1))
        return self.norm(out)


class GraphNetBlock(nn.Module):
    """
    One graph-network block in Battaglia et al. style:
      edge update -> node update -> global update
    """
    def __init__(
        self,
        node_in,
        edge_in,
        global_in,
        node_out,
        edge_out,
        global_out,
        hidden_dim=64,
        num_hidden_layers=1,
    ):
        super().__init__()
        self.edge_model = EdgeModel(edge_in, node_in, global_in, hidden_dim, edge_out, num_hidden_layers)
        self.node_model = NodeModel(node_in, edge_out, global_in, hidden_dim, node_out, num_hidden_layers)
        self.global_model = GlobalModel(global_in, edge_out, node_out, hidden_dim, global_out, num_hidden_layers)

    def forward(self, x, edge_index, edge_attr, u, batch, var_mask):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        row, _ = edge_index
        edge_batch = batch[row]

        u_per_edge = u[edge_batch]
        edge_attr = self.edge_model(x[row], x[edge_index[1]], edge_attr, u_per_edge)

        u_per_node = u[batch]
        x = self.node_model(x, edge_index, edge_attr, u_per_node)

        u = self.global_model(u, edge_attr, x, batch, edge_batch, var_mask)
        return x, edge_attr, u


class GraphQSat(nn.Module):
    """
    Reproduction of the paper's Graph-Q-SAT GNN model.

    Input:
        data.x         : [N, 2] node features
        data.edge_index: [2, E] directed edges
        data.edge_attr : [E, 2] edge features
        data.batch     : [N] optional graph ids for batched PyG data

    Output:
        qs: [num_variable_nodes, 2] Q-values for true/false assignment for each variable node
    """
    def __init__(
        self,
        node_input_dim: int = 2,
        edge_input_dim: int = 2,
        global_input_dim: int = 1,   # "empty" global in paper; we use a learned-zero placeholder width 1
        encoder_dim: int = 32,
        core_node_dim: int = 64,
        core_edge_dim: int = 64,
        core_global_dim: int = 32,
        decoder_dim: int = 32,
        message_passing_steps: int = 4,
    ):
        super().__init__()
        self.message_passing_steps = message_passing_steps
        self.global_input_dim = global_input_dim

        # Encoder: independent graph network / per-entity encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, encoder_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, encoder_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, encoder_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_dim),
        )

        # Core: receives concatenation of encoder latents and previous core state
        self.core = GraphNetBlock(
            node_in=encoder_dim + core_node_dim,
            edge_in=encoder_dim + core_edge_dim,
            global_in=encoder_dim + core_global_dim,
            node_out=core_node_dim,
            edge_out=core_edge_dim,
            global_out=core_global_dim,
            hidden_dim=64,
            num_hidden_layers=1,
        )

        # Decoder: only node outputs are needed for Q-values
        self.node_decoder = nn.Sequential(
            nn.Linear(core_node_dim, decoder_dim),
            nn.ReLU(),
            nn.LayerNorm(decoder_dim),
        )
        self.q_head = nn.Linear(decoder_dim, 2)  # Q(true), Q(false)

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        var_mask = x[:, 0] > x[:, 1]

        # Keep the stored graph unidirectional, but duplicate each literal edge
        # internally so variables and clauses can both receive messages.
        reverse_edge_index = edge_index[[1, 0]]
        mp_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        mp_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # Paper says global input is empty and only used for message passing.
        # We instantiate it as a zero vector per graph.
        u0 = x.new_zeros((num_graphs, self.global_input_dim))

        # Encode
        x_enc = self.node_encoder(x)           # [N, 32]
        e_enc = self.edge_encoder(mp_edge_attr)   # [2E, 32]
        u_enc = self.global_encoder(u0)        # [B, 32]

        # First core input: encoder output concatenated with zeros
        x_lat = torch.cat([x_enc, x.new_zeros(x.size(0), 64)], dim=-1)
        e_lat = torch.cat([e_enc, mp_edge_attr.new_zeros(mp_edge_attr.size(0), 64)], dim=-1)
        u_lat = torch.cat([u_enc, u0.new_zeros(num_graphs, 32)], dim=-1)

        # Process: 4 message-passing iterations in the paper
        for _ in range(self.message_passing_steps):
            x_core, e_core, u_core = self.core(x_lat, mp_edge_index, e_lat, u_lat, batch, var_mask)

            # Recurrent concat with encoder outputs
            x_lat = torch.cat([x_enc, x_core], dim=-1)
            e_lat = torch.cat([e_enc, e_core], dim=-1)
            u_lat = torch.cat([u_enc, u_core], dim=-1)

        # Decode node states -> 2 Q-values/node
        node_h = self.node_decoder(x_core)

        # Variable nodes only participate in action selection
        qs = self.q_head(node_h)

        return qs, var_mask

    @staticmethod
    def split_candidate_logits(data: Data, qs: torch.Tensor, var_mask: torch.Tensor) -> List[torch.Tensor]:
        batch = getattr(data, "batch", None)
        if batch is None:
            return [qs[var_mask].flatten()]

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        var_qs = qs[var_mask]
        var_batch = batch[var_mask]
        return [
            var_qs[var_batch == graph_idx].flatten()
            for graph_idx in range(num_graphs)
        ]

    @torch.no_grad()
    def select_action(self, data: Data):
        """
        Returns:
            action: int, the index of the candidate to branch on.

        The paper selects the argmax over all variable-node Q-values.
        """
        qs, var_mask = self.forward(data)
        candidate_logits = self.split_candidate_logits(data, qs, var_mask)[0]
        return candidate_logits.argmax(dim=-1).item()


# ---- Example usage ----
if __name__ == "__main__":
    from cnf import CNFLoader
    cnf_file = "data/uf50-218/uf50-01.cnf"
    loader = CNFLoader(cnf_file)
    loader.load_cnf()
    loader.build_vcg()
    data = loader.graph

    model = GraphQSat()

    qs, var_mask = model(data)
    print("qs shape:", qs.shape)
    print("qs:", qs)

    action = model.select_action(data)
    print("selected action:", action)
