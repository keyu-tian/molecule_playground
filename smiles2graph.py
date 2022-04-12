from typing import Tuple

import torch
import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import rdChemReactions


def smarts2graph(reaction_smarts: str, mole_offset: int, edge_offset: int, atom_offset: int):
    reaction_smarts = reaction_smarts.split(' ')[0].strip()

    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    rxn.ClearComputedProps()
    
    R_mol, SC_mol, O_mol = list(rxn.GetReactants()), list(rxn.GetAgents()), list(rxn.GetProducts())
    for mo in R_mol + SC_mol + O_mol:
        for a in mo.GetAtoms():
            a.SetAtomMapNum(0)

    R_str = '.'.join(map(Chem.MolToSmiles, R_mol))
    SC_str = '.'.join(map(Chem.MolToSmiles, SC_mol))
    O_str = '.'.join(map(Chem.MolToSmiles, O_mol))

    S_ls, C_ls = [], []
    if len(SC_str) > 0:
        for catalyst_or_solvent in SC_str.split('.'):
            if any(x in catalyst_or_solvent for x in {
                'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                'W', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                'Pb', 'Bi',
            }):
                C_ls.append(catalyst_or_solvent)
            else:
                S_ls.append(catalyst_or_solvent)
    S_str, C_str = '.'.join(S_ls), '.'.join(C_ls)

    R_canon, S_canon, C_canon, O_canon = map(Chem.CanonSmiles, (R_str, S_str, C_str, O_str))

    # rm duplicated molecules
    R_set = set(R_canon.split('.'))
    new_O = []
    for o in O_canon.split('.'):
        if o not in R_set:
            new_O.append(o)
    O_canon = Chem.CanonSmiles('.'.join(new_O))
    
    react_mole_roles, react_mole_counts = [], []
    react_edge_index, react_edge_feat, react_atom_feat = [], [], []
    react_edge_offset, react_atom_offset = [], []

    for count, (role, smiles) in (
        [(0, r) for r in R_canon.split('.')]
        + [(1, s) for s in S_canon.split('.')]
        + [(2, c) for c in C_canon.split('.')]
        + [(3, o) for o in O_canon.split('.')]
    ):
        react_mole_roles.append(role)
        react_mole_counts.append(count)
        mole_offset += 1
    
        # shape:
        #  (E, 2)     (E, 3)     (V, 9)
        edge_index, edge_feat, atom_feat = smiles2graph(smiles)
        react_edge_index.append(edge_index), react_edge_feat.append(edge_feat), react_atom_feat.append(atom_feat)
    
        E, V = edge_index.shape[0], atom_feat.shape[0]
        assert V > 0
        edge_offset += E
        react_edge_offset.append(edge_offset)
        atom_offset += V
        react_atom_offset.append(atom_offset)
    
    return (
        mole_offset, edge_offset, atom_offset,
        react_mole_roles, react_mole_counts,
        react_edge_index, react_edge_feat, react_atom_feat,
        react_edge_offset, react_atom_offset,
    )
    

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
    # graph = smiles2graph('[C:1]([C:5]1[CH:10]=[CH:9][CH:8]=[CH:7][C:6]=1[N+:11]([O-])=O)([CH3:4])([CH3:3])[CH3:2]')
    # print(graph)
    # graph = smiles2graph('C(C)(C)(C)C1=C(C=CC=C1)[N+](=O)[O-]')
    # print(graph)

    mole_offset, edge_offset, atom_offset = 0, 0, 0
    smarts = '[C:1]=[C:2].[O:3]>[O].[Pd]>[C:1]([O:3])[C:2]'

    (
        mole_offset, edge_offset, atom_offset,
        react_mole_roles, react_mole_counts,
        react_edge_index, react_edge_feat, react_atom_feat,
        react_edge_offset, react_atom_offset,
    ) = smarts2graph(smarts, mole_offset, edge_offset, atom_offset)
