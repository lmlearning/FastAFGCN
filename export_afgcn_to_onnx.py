# export_onnx.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
import torch.nn as nn
import numpy as np

# AFGCN model (PyG version)
class AFGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim))
        for _ in range(1, num_layers):
            self.convs.append(GraphConv(hid_dim, hid_dim))
        self.fc = nn.Linear(hid_dim, out_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h0 = x
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dp(h)
            h = h + h0
        return self.fc(h).squeeze(-1)  # logits

# Load trained model
model = AFGCN(in_dim=128, hid_dim=128, out_dim=1, num_layers=4)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# Dummy inputs (replace 100 with your typical node count)
num_nodes = 100
x_dummy = torch.empty((num_nodes, 128))
nn.init.xavier_uniform_(x_dummy)
edge_index_dummy = torch.randint(0, num_nodes, (2, 300))

# Export to ONNX
torch.onnx.export(
    model,
    (x_dummy, edge_index_dummy),
    "afgcn.onnx",
    input_names=["x", "edge_index"],
    output_names=["logits"],
    dynamic_axes={"x": {0: "num_nodes"}, "edge_index": {1: "num_edges"}},
    opset_version=17,
)
print("✅ Exported to afgcn.onnx")
