import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class MetabolicGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, heads=4):
        super(MetabolicGNN, self).__init__()
        
        # GAT Convolutions
        # First layer: Input features -> Hidden channels * heads
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.6)
        
        # Second layer: Hidden channels * heads -> Hidden channels (averaging heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        
        # Classification Head (per node or per graph? Plan says community detection/node labeling)
        # Assuming we want to classify nodes (e.g. which pathway a node belongs to) 
        # OR graph-level classification. "Community Detection" usually implies Node classification.
        
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # x: Node features [num_nodes, num_node_features]
        # edge_index: Graph connectivity [2, num_edges]

        # 1. First Transformer / Attention Block
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        # 2. Second Transformer Block
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        
        # 3. Output Projection
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)

def get_model(num_features, num_classes):
    return MetabolicGNN(num_features, num_classes)
