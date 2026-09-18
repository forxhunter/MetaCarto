import cobra
import networkx as nx

def load_sbml_model(file_path):
    """
    Loads an SBML model using COBRApy.
    """
    try:
        model = cobra.io.read_sbml_model(file_path)
        return model
    except Exception as e:
        print(f"Error loading SBML model: {e}")
        return None

def build_bipartite_graph(model):
    """
    Converts a COBRA model into a NetworkX bipartite graph.
    Nodes: Metabolites (0), Reactions (1)
    Edges: Stoichiometry
    """
    G = nx.Graph()
    
    # Add Metabolite Nodes
    for met in model.metabolites:
        G.add_node(met.id, type="metabolite", name=met.name, bipartite=0)
        
    # Add Reaction Nodes
    for rxn in model.reactions:
        G.add_node(rxn.id, type="reaction", name=rxn.name, bipartite=1)
        
        # Add Edges
        for met, coeff in rxn.metabolites.items():
            G.add_edge(rxn.id, met.id, weight=coeff)
            
    return G

def convert_to_pyg(G):
    """
    Converts a NetworkX graph to PyTorch Geometric Data object.
    (Placeholder for now - simple conversion)

    torch and torch_geometric are imported lazily: they are only needed by the
    GNN experiments, and importing them at module scope made this module -- and
    therefore every driver that loads an SBML model -- fail to import on any
    environment without torch_geometric installed.
    """
    import torch
    from torch_geometric.data import Data

    # Create mapping from node ID to index
    node_mapping = {n: i for i, n in enumerate(G.nodes())}
    
    # Edges
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_index.append([node_mapping[v], node_mapping[u]]) # Undirected
        edge_attr.append(data.get("weight", 0))
        edge_attr.append(data.get("weight", 0))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Node features (Mock features for now: 1-hot encoding of type)
    x = []
    for n, data in G.nodes(data=True):
        if data["type"] == "metabolite":
            x.append([1, 0])
        else:
            x.append([0, 1])
            
    x = torch.tensor(x, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

import xml.etree.ElementTree as ET

def parse_kgml(file_path):
    """
    Parses a KEGG KGML (XML) file to extract the metabolic network structure.
    Returns a NetworkX graph representing the ground truth layout (if coords exist).
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        G = nx.Graph()
        
        # 1. Parse Entries (Nodes)
        # Entry types: compound, enzyme, map, group
        entries = {}
        for entry in root.findall('entry'):
            id = entry.get('id')
            name = entry.get('name')
            type_ = entry.get('type')
            
            # Extract graphics for coordinates (Ground Truth)
            graphics = entry.find('graphics')
            x = graphics.get('x')
            y = graphics.get('y')
            
            if x and y:
                entries[id] = {'name': name, 'type': type_, 'x': float(x), 'y': float(y)}
                G.add_node(name, type=type_, x=float(x), y=float(y), kegg_id=id)
                
        # 2. Parse Relations (Edges) - protein-protein or regulation
        for rel in root.findall('relation'):
            entry1 = rel.get('entry1')
            entry2 = rel.get('entry2')
            type_ = rel.get('type')
            
            # Use names if possible, else IDs
            u = entries.get(entry1, {}).get('name')
            v = entries.get(entry2, {}).get('name')
            
            if u and v:
                G.add_edge(u, v, type=type_)
                
        # 3. Parse Reactions (Metabolic Edges)
        for rxn in root.findall('reaction'):
            rxn_name = rxn.get('name')
            
            substrates = [sub.get('name') for sub in rxn.findall('substrate')]
            products = [prod.get('name') for prod in rxn.findall('product')]
            
            # Add reaction node
            if rxn_name not in G:
                G.add_node(rxn_name, type='reaction')
            
            for s in substrates:
                G.add_edge(s, rxn_name, relation='substrate')
            for p in products:
                G.add_edge(rxn_name, p, relation='product')
                
        return G
    except Exception as e:
        print(f"Error parsing KGML {file_path}: {e}")
        return None
