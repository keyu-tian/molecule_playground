import datetime
import json
import os
import pathlib
import time
from collections import OrderedDict
from pprint import pformat
from typing import List, Dict

import pandas as pd
import torch
import tqdm

from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2

import smiles2graph


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def jsonfy_ord():
    dataset_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-zip' / 'data'
    all_gz_paths = []
    for dir_name in sorted(os.listdir(dataset_root)):
        data_root = dataset_root / dir_name
        data = sorted(os.listdir(data_root))
        for i, gz_name in enumerate(data):
            all_gz_paths.append((f'{dir_name}--{i + 1}of{len(data)}', str(data_root / gz_name)))
    
    output_dir = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
    os.makedirs(output_dir, exist_ok=True)
    gz_len = len(all_gz_paths)
    for i, (ds_name, gz_path) in enumerate(all_gz_paths):
        ds_desc = f'{ds_name} ({i+1:2d}/{gz_len})'
        output_file = str(output_dir / f'{ds_name}.json')
        if os.path.exists(output_file):
            print(f'{time_str()} {ds_desc} already dumped @ {output_file}, skipped !')
            continue
        
        start_t = time.time()
        print(f'{time_str()} loading ...', end='     ')
        data = message_helpers.load_message(gz_path, dataset_pb2.Dataset)
        load_t = time.time()
        
        print(f'{time_str()} casting ...', end='     ')
        rows = [message_helpers.message_to_row(message) for message in tqdm.tqdm(data.reactions, desc=f'[read {ds_desc}]', dynamic_ncols=True)]
        cast_t = time.time()
        
        print(f'{time_str()} dumping ...', end='     ')
        with open(output_file, 'w') as fout:
            json.dump(rows, fout)
        dump_t = time.time()
        
        print(f'{time_str()} {ds_desc} dumped @ {output_file}, load={load_t-start_t:.2f} cast={cast_t-start_t:.2f}, dump={dump_t-cast_t:.2f}')


def main():
    jsons_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
    torch_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-torch'
    os.makedirs(torch_root, exist_ok=True)
    
    for json_name in sorted(os.listdir(jsons_root)):
        torch_file = torch_root / json_name.replace('.json', '.pth')
        if os.path.exists(torch_file):
            print(f'{time_str()} [{json_name}]: already preprocessed !')
            continue
        # reactants, solvents, catalysts, outcomes = [], [], [], []
        # inputs = {
        #     'REACTANT': reactants,
        #     'SOLVENT': solvents,
        #     'CATALYST': catalysts
        # }
        # with open(str(jsons_root / json_name), 'r') as fin:
        #     dataset: List[Dict[str, str]] = json.load(fin)
        # for i, one_reaction in enumerate(dataset):
        #     reactants.clear(), solvents.clear(), catalysts.clear(), outcomes.clear()
        #     for k, v in one_reaction.items():
        #         if k.startswith('inputs') and k.endswith('.type') and v == 'SMILES':
        #             role = one_reaction.get(k.split('identifiers[')[0] + 'reaction_role', 'REACTANT')
        #             inputs[role].append(one_reaction[k.replace('.type', '.value')])
        #         elif k.startswith('outcomes') and k.endswith('.type') and v == 'SMILES':
        #             outcomes.append(one_reaction[k.replace('.type', '.value')])

        role2idx = {
            'REACTANT': 0, 'REATANT': 0, 'REAGENT': 0,
            'SOLVENT': 1,
            'CATALYST': 2,
            'OUTCOME': 3, 'PRODUCT': 3,
        }
        with open(str(jsons_root / json_name), 'r') as fin:
            dataset: List[Dict[str, str]] = json.load(fin)
        bad_reactions = []
        all_mole_roles, all_edge_index, all_edge_feat, all_atom_feat = [], [], [], []
        mole_offset, all_mole_offset = 0, [0]
        edge_offset, all_edge_offset = 0, [0]
        atom_offset, all_atom_offset = 0, [0]
        bar = tqdm.tqdm(dataset, desc=f'[{json_name}]', mininterval=2., dynamic_ncols=True)
        stt = time.time()
        for i, one_reaction in enumerate(bar):
            try:
                rets = parse_reaction(bar, role2idx, one_reaction, mole_offset, edge_offset, atom_offset)
            except:
                bad_reactions.append(i)
            else:
                (
                    mole_offset, edge_offset, atom_offset,
                    react_mole_roles, react_edge_index, react_edge_feat, react_atom_feat,
                    react_edge_offset, react_atom_offset,
                ) = rets
                all_mole_roles.extend(react_mole_roles)
                all_edge_index.extend(react_edge_index), all_edge_feat.extend(react_edge_feat)
                all_atom_feat.extend(react_atom_feat)
                all_mole_offset.append(mole_offset)
                all_edge_offset.extend(react_edge_offset)
                all_atom_offset.extend(react_atom_offset)
                
        bar.close()
        num_reactions = len(dataset)
        avg_mole_cnt, avg_edge_cnt, avg_atom_cnt = mole_offset / num_reactions, edge_offset / num_reactions, atom_offset / num_reactions
        print(f'{time_str()} [{json_name}]: #bad={len(bad_reactions)}, #Mol={avg_mole_cnt:.2f}, #E={avg_edge_cnt:.2f}, #A={avg_atom_cnt:.2f}, cost={time.time()-stt:.2f}s')
        if len(bad_reactions):
            print(f'   ***** [{json_name}]: bad_reactions={bad_reactions}')
        
        tensors = {
            'json_name': json_name,
            'bad_reactions': torch.tensor(bad_reactions, dtype=torch.int32),
            'mole_offset': torch.tensor(all_mole_offset, dtype=torch.int32),
            'mole_roles': torch.tensor(all_mole_roles, dtype=torch.int32),
            'edge_offset': torch.tensor(all_edge_offset, dtype=torch.int32),
            'edge_indices': torch.cat(all_edge_index, dim=0),
            'edge_features': torch.cat(all_edge_feat, dim=0),
            'atom_offset': torch.tensor(all_atom_offset, dtype=torch.int32),
            'atom_features': torch.cat(all_atom_feat, dim=0),
        }
        tensors['num_moles_per_reaction'] = tensors['mole_offset'][1:] - tensors['mole_offset'][:-1]
        tensors['num_edges_per_mole'] = tensors['edge_offset'][1:] - tensors['edge_offset'][:-1]
        tensors['num_atoms_per_mole'] = tensors['atom_offset'][1:] - tensors['atom_offset'][:-1]
        torch.save(tensors, torch_file)


def parse_reaction(bar, role2idx, one_reaction, mole_offset, edge_offset, atom_offset):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    react_mole_roles, react_edge_index, react_edge_feat, react_atom_feat = [], [], [], []
    react_edge_offset, react_atom_offset = [], []
    for k, v in one_reaction.items():
        if k.startswith('inputs') and k.endswith('.type') and v.upper() == 'SMILES':
            role = one_reaction.get(k.split('identifiers[')[0] + 'reaction_role', 'REACTANT').upper()
        elif k.startswith('outcomes') and k.endswith('.type') and v.upper() == 'SMILES':
            role = 'OUTCOME'
        else:
            role = None
        if role is not None:
            react_mole_roles.append(role2idx[role])
            # (E, 2)     (E, 3)     (V, 9)
            edge_index, edge_feat, atom_feat = smiles2graph.smiles2graph(one_reaction[k.replace('.type', '.value')])
            react_edge_index.append(edge_index), react_edge_feat.append(edge_feat), react_atom_feat.append(atom_feat)
            
            E, V = edge_index.shape[0], atom_feat.shape[0]
            bar.set_postfix(OrderedDict(role=f'{role:10s}', E=f'{E:3d}', V=f'{V:3d}'))
            edge_offset += E
            react_edge_offset.append(edge_offset)
            atom_offset += V
            react_atom_offset.append(atom_offset)
            
            mole_offset += 1
    return mole_offset, edge_offset, atom_offset, react_mole_roles, react_edge_index, react_edge_feat, react_atom_feat, react_edge_offset, react_atom_offset


if __name__ == '__main__':
    main()


"""

03--1of1.json
conditions.conditions_are_dynamic
conditions.details
identifiers[0].is_mapped
identifiers[0].type
identifiers[0].value
inputs["m1"].components[0].amount.moles.precision
inputs["m1"].components[0].amount.moles.units
inputs["m1"].components[0].amount.moles.value
inputs["m1"].components[0].identifiers[0].type
inputs["m1"].components[0].identifiers[0].value
inputs["m1"].components[0].identifiers[1].type
inputs["m1"].components[0].identifiers[1].value
inputs["m1"].components[0].identifiers[2].type
inputs["m1"].components[0].identifiers[2].value
inputs["m1"].components[0].reaction_role
inputs["m2"].components[0].amount.moles.precision
inputs["m2"].components[0].amount.moles.units
inputs["m2"].components[0].amount.moles.value
inputs["m2"].components[0].identifiers[0].type
inputs["m2"].components[0].identifiers[0].value
inputs["m2"].components[0].identifiers[1].type
inputs["m2"].components[0].identifiers[1].value
inputs["m2"].components[0].identifiers[2].type
inputs["m2"].components[0].identifiers[2].value
inputs["m2"].components[0].reaction_role
inputs["m3"].components[0].amount.moles.precision
inputs["m3"].components[0].amount.moles.units
inputs["m3"].components[0].amount.moles.value
inputs["m3"].components[0].identifiers[0].type
inputs["m3"].components[0].identifiers[0].value
inputs["m3"].components[0].identifiers[1].type
inputs["m3"].components[0].identifiers[1].value
inputs["m3"].components[0].identifiers[2].type
inputs["m3"].components[0].identifiers[2].value
inputs["m3"].components[0].reaction_role
notes.procedure_details
outcomes[0].products[0].identifiers[0].type
outcomes[0].products[0].identifiers[0].value
outcomes[0].products[0].identifiers[1].type
outcomes[0].products[0].identifiers[1].value
outcomes[0].products[0].identifiers[2].type
outcomes[0].products[0].identifiers[2].value
provenance.doi
provenance.patent
provenance.record_created.person.email
provenance.record_created.person.name
provenance.record_created.person.orcid
provenance.record_created.person.organization
provenance.record_created.person.username
provenance.record_created.time.value
provenance.record_modified[0].details
provenance.record_modified[0].person.email
provenance.record_modified[0].person.username
provenance.record_modified[0].time.value
reaction_id
"""