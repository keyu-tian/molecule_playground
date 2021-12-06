from typing import Tuple

import torch
import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem


def smiles2graph(smiles_string) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    
    mol = Chem.MolFromSmiles(smiles_string)
    
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    node_feat = np.array(atom_features_list, dtype=np.int64)
    
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_feature = bond_to_feature_vector(bond)
            
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        
        # data.edge_index: Graph connectivity in COO format with shape [num_edges, 2]
        edge_index = np.array(edges_list, dtype=np.int64)
        
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    
    else:  # mol has no bonds
        edge_index = np.empty((0, 2), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
    
    # graph = dict()
    # graph['edge_index'] = edge_index
    # graph['edge_feat'] = edge_attr
    # graph['node_feat'] = node_feat
    # graph['num_nodes'] = len(node_feat)
    
    return (
        torch.tensor(edge_index, dtype=torch.int32),
        torch.tensor(edge_attr, dtype=torch.int32),
        torch.tensor(node_feat, dtype=torch.int32),
    )


if __name__ == '__main__':
    graph = smiles2graph('[C:1]([C:5]1[CH:10]=[CH:9][CH:8]=[CH:7][C:6]=1[N+:11]([O-])=O)([CH3:4])([CH3:3])[CH3:2]')
    print(graph)
    graph = smiles2graph('C(C)(C)(C)C1=C(C=CC=C1)[N+](=O)[O-]')
    print(graph)
