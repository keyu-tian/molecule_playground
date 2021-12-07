import datetime
import json
import os
import pathlib
import time
from typing import List, Dict

import torch
import tqdm
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

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

    role2idx = {
        'REACTANT': 0, 'REATANT': 0, 'REAGENT': 0,
        'SOLVENT': 1,
        'CATALYST': 2,
        'OUTCOME': 3, 'PRODUCT': 3,
    }
    
    json_names = sorted(os.listdir(jsons_root))
    for json_i, json_name in enumerate(json_names):
        ds_desc = f'{json_name} ({json_i+1:2d}/{len(json_names)})'
        
        torch_file = torch_root / json_name.replace('.json', '.pth')
        if os.path.exists(torch_file):  # skip this dataset
            print(f'{time_str()} [{json_name}]: already preprocessed !')
            continue

        with open(str(jsons_root / json_name), 'r') as fin:
            dataset: List[Dict[str, str]] = json.load(fin)
        
        bad_reaction_ids = []
        all_mole_roles = []
        all_edge_index, all_edge_feat, all_atom_feat = [], [], []
        reaction_ids = []
        mole_offset, all_mole_offset = 0, [0]
        edge_offset, all_edge_offset = 0, [0]
        atom_offset, all_atom_offset = 0, [0]
        
        bar = tqdm.tqdm(dataset, desc=f'[{ds_desc}]', mininterval=1., dynamic_ncols=True)
        stt = time.time()
        for i, one_reaction in enumerate(bar):
            #             one_reaction['reaction_id'] example: ord-56b1f4bfeebc4b8ab990b9804e798aa7
            reaction_id = one_reaction['reaction_id'].replace('ord-', '')
            reaction_id = [int(ch, base=16) for ch in reaction_id]
            try:
                rets = parse_reaction(bar, role2idx, one_reaction, mole_offset, edge_offset, atom_offset)
            except:
                bad_reaction_ids.append(reaction_id)
            else:
                reaction_ids.append(reaction_id)
                (
                    mole_offset, edge_offset, atom_offset,
                    react_mole_roles,
                    react_edge_index, react_edge_feat, react_atom_feat,
                    react_edge_offset, react_atom_offset,
                ) = rets
                all_mole_roles.extend(react_mole_roles)
                all_edge_index.extend(react_edge_index), all_edge_feat.extend(react_edge_feat), all_atom_feat.extend(react_atom_feat)
                all_mole_offset.append(mole_offset)
                all_edge_offset.extend(react_edge_offset)
                all_atom_offset.extend(react_atom_offset)
        bar.close()
        
        num_reactions = len(dataset)
        # per-reaction stats
        avg_mole_cnt, avg_edge_cnt, avg_atom_cnt = mole_offset / num_reactions, edge_offset / num_reactions, atom_offset / num_reactions
        if len(all_edge_index) == 0:    # buggy dataset!
            print(f'{time_str()} [{json_name}]: bad_dataset !')
            torch.save({'json_name': json_name}, str(torch_file) + '.bug')
            continue

        # show per-reaction stats
        print(f'{time_str()} [{json_name}]: #bad={len(bad_reaction_ids)}, #Mol={avg_mole_cnt:.2f}, #E={avg_edge_cnt:.2f}, #A={avg_atom_cnt:.2f}, cost={time.time()-stt:.2f}s')
        if len(bad_reaction_ids):
            str_rids = []
            for rid_as_32_ints in bad_reaction_ids:
                rid_as_32_chars = [f'{i:x}' for i in rid_as_32_ints]
                str_rid = ''.join(rid_as_32_chars)
                str_rids.append(str_rid)
            print(f'   ***** [{json_name}]: bad_reaction_ids={str_rids}')
        
        # save as tensors
        tensors = {
            'json_name': json_name,
            'bad_reaction_ids': torch.tensor(bad_reaction_ids, dtype=torch.uint8),
            'reaction_ids': torch.tensor(reaction_ids, dtype=torch.uint8),
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

        max_rid = tensors['mole_offset'].size(0)
        assert max_rid == tensors['reaction_ids'].size(0) + 1
        
        max_mid = tensors['atom_offset'].size(0)
        assert max_mid == tensors['mole_offset'][-1] + 1
        assert max_mid == tensors['edge_offset'].size(0)
        assert max_mid == tensors['mole_roles'].size(0) + 1

        max_aid = tensors['atom_features'].size(0)
        assert max_aid == tensors['atom_offset'][-1]

        max_eid = tensors['edge_features'].size(0)
        assert max_eid == tensors['edge_offset'][-1]
        assert max_eid == tensors['edge_indices'].size(0)
        
        torch.save(tensors, torch_file)


def parse_reaction(bar, role2idx, one_reaction, mole_offset, edge_offset, atom_offset):
    react_mole_roles = []
    react_edge_index, react_edge_feat, react_atom_feat = [], [], []
    react_edge_offset, react_atom_offset = [], []
    for k, v in one_reaction.items():
        if k.startswith('inputs') and k.endswith('.type') and v.upper() == 'SMILES':
            role = one_reaction.get(k.split('identifiers[')[0] + 'reaction_role', 'REACTANT').upper()
        elif k.startswith('outcomes') and k.endswith('.type') and v.upper() == 'SMILES':
            role = 'OUTCOME'
        else:
            role = None
        if role is not None:
            for smiles in one_reaction[k.replace('.type', '.value')].split('.'):
                #  (E, 2)     (E, 3)     (V, 9)
                edge_index, edge_feat, atom_feat = smiles2graph.smiles2graph(smiles)
                react_edge_index.append(edge_index), react_edge_feat.append(edge_feat), react_atom_feat.append(atom_feat)
                
                E, V = edge_index.shape[0], atom_feat.shape[0]
                bar.set_postfix_str(f'role={role:9s}, E={E:3d}, V={V:3d}', refresh=False)
                edge_offset += E
                react_edge_offset.append(edge_offset)
                atom_offset += V
                react_atom_offset.append(atom_offset)

                react_mole_roles.append(role2idx[role])
                mole_offset += 1
                
    return mole_offset, edge_offset, atom_offset, react_mole_roles, react_edge_index, react_edge_feat, react_atom_feat, react_edge_offset, react_atom_offset


if __name__ == '__main__':
    main()
