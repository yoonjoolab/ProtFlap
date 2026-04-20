import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# =========================
# ACTIVATION FUNCTION
# =========================
def get_activation_fn(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == 'LeakyReLU':
        return nn.LeakyReLU()
    elif name == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


# =========================
# REGRESSION MODEL (RAW RMSF)
# =========================
class NodeMLP_GCN(nn.Module):
    def __init__(self, in_node_feats, hidden_dim=256, num_gcn_layers=5,
                 dropout=0.3, use_residual=True, use_batch_norm=False,
                 activation='ReLU', use_bias=True):

        super().__init__()

        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.activation_fn = get_activation_fn(activation)

        # =========================
        # NODE FEATURE MLP
        # =========================
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_feats, 64, bias=use_bias),
            self.activation_fn,
            nn.Linear(64, 128, bias=use_bias),
            self.activation_fn,
            nn.Linear(128, 128, bias=use_bias),
            self.activation_fn,
            nn.Linear(128, 256, bias=use_bias),
            self.activation_fn,
            nn.Linear(256, hidden_dim, bias=use_bias),
            self.activation_fn,
            nn.Dropout(dropout)
        )

        # =========================
        # GCN LAYERS (WITH EDGE WEIGHTS)
        # =========================
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, bias=use_bias)
            for _ in range(num_gcn_layers)
        ])

        if use_batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(num_gcn_layers)
            ])
        else:
            self.bn_layers = None

        # =========================
        # OUTPUT LAYER
        # =========================
        self.out = nn.Linear(hidden_dim, 1)  # RAW RMSF

    # =========================
    # FORWARD
    # =========================
    def forward(self, x, edge_index, edge_attr=None, batch=None):

        # NODE EMBEDDING
        x = self.node_mlp(x)

        # GCN LAYERS
        for i, gcn in enumerate(self.gcn_layers):
            x_res = x

            if edge_attr is not None:
                x = gcn(x, edge_index, edge_weight=edge_attr)
            else:
                x = gcn(x, edge_index)

            if self.bn_layers is not None:
                x = self.bn_layers[i](x)

            x = self.activation_fn(x)

            if self.use_residual:
                x = x + x_res

        # RAW OUTPUT (per residue)
        preds = self.out(x).squeeze(-1)  # shape: [num_nodes]

        return preds, None
