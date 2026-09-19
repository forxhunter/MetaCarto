import torch
import torch.optim as optim
from experiments.gnn_model import get_model
from experiments.data_loader import create_mock_data

def train_one_epoch(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    
    # Simple node classification task
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

import os
import glob
from src.parsing import parse_kgml
from experiments.gnn_model import get_model
from torch_geometric.data import Data
import networkx as nx

def load_real_data(data_dir="data/kegg"):
    """
    Loads all XML files from data_dir, parses them, and returns a list of PyG Data objects.
    """
    files = glob.glob(os.path.join(data_dir, "*.xml"))
    print(f"Found {len(files)} training files in {data_dir}")
    dataset = []
    
    for f in files:
        G = parse_kgml(f)
        if G is None or len(G.nodes) < 5:
            continue
            
        # Convert NetworkX G to PyG Data
        # 1. Node Features (Mock one-hot for now. Real world: Use Chemistry/MACCS)
        x = []
        node_map = {n: i for i, n in enumerate(G.nodes)}
        
        for n, data in G.nodes(data=True):
            # Simple feature: 1 if reaction, 0 if metabolite
            feat = [1.0] if data.get('type') == 'reaction' else [0.0] 
            # Pad to match model input dim
            x.append(feat + [0.0]*9) 
            
        x = torch.tensor(x, dtype=torch.float)
        
        # 2. Edges
        edge_index = []
        for u, v in G.edges:
            if u in node_map and v in node_map:
                edge_index.append([node_map[u], node_map[v]])
                edge_index.append([node_map[v], node_map[u]])
                
        if not edge_index:
             continue
             
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 3. Y Labels (Ground Truth Groups)
        # For unsupervised/clustering, we might self-supervise. 
        # Here, let's create dummy labels for testing flow or use a heuristic.
        # Placeholder: Random labels
        y = torch.randint(0, 5, (len(G.nodes),))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
        
    return dataset

def run_training():
    print("Initializing Training Loop...")
    
    # 1. Load Data
    dataset = load_real_data()
    if not dataset:
        print("No data found. Please run 'python fetch_kegg.py --org eco' first.")
        # Fallback to mock for demonstration if empty
        from experiments.data_loader import create_mock_data
        print("Falling back to mock data...")
        dataset = [create_mock_data(10) for _ in range(5)]
    else:
        print(f"Loaded {len(dataset)} real graphs.")

    # 2. Setup Model
    num_features = 10
    num_classes = 5 # Predicted Subsystems
    model = get_model(num_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()
    
    # 3. Train
    model.train()
    for epoch in range(10):
        total_loss = 0
        for data in dataset:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y) # Warning: data.y needs to be set correctly
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss / len(dataset):.4f}")
            
    print("Training complete. Model saved to 'gnn_model.pth'")
    torch.save(model.state_dict(), "gnn_model.pth")

if __name__ == "__main__":
    run_training()
