from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np

def get_smiles_from_annotation(metabolite):
    """
    Attempts to retrieve SMILES string from metabolite annotations.
    (This is highly dependent on the SBML annotation format, e.g., KEGG, ChEBI)
    """
    annotation = metabolite.annotation
    # Placeholder logic: Check for 'smiles' key specific to some databases
    # Real implementation would query PubChem/ChEBI via API if not found
    if 'smiles' in annotation:
        if isinstance(annotation['smiles'], list):
            return annotation['smiles'][0]
        return annotation['smiles']
    return None

def get_maccs_fingerprint(smiles):
    """
    Converts a SMILES string to a MACCS fingerprint.
    Returns a numpy array.
    """
    if not smiles:
        return np.zeros(167) # MACCS keys are 167 bits
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
        
    fp = MACCSkeys.GenMACCSKeys(mol)
    # Convert RDKit BitVect to numpy array
    return np.array(fp)

def augment_graph_with_chemistry(G, model):
    """
    Iterates through the graph nodes, finds metabolite counterparts, and adds
    fingerprint data to the node attributes.
    """
    for node_id, data in G.nodes(data=True):
        if data['type'] == 'metabolite':
            try:
                met = model.metabolites.get_by_id(node_id)
                smiles = get_smiles_from_annotation(met)
                fp = get_maccs_fingerprint(smiles)
                G.nodes[node_id]['fingerprint'] = fp
            except:
                pass # Metabolite might be missing or generic
    return G
