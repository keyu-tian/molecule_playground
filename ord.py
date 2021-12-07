import datetime
import json
import os
import pathlib
import random
import time
import zipfile
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from multiprocessing import Pool
from multiprocessing import cpu_count

import pandas
import torch
import tqdm
import wget

# install ord_schema following https://github.com/open-reaction-database/ord-schema
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2

# conda install -c rdkit rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import smiles2graph


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def download_and_jsonfy_data():
    home_path = pathlib.Path(os.path.expanduser('~'))
    dataset_root = home_path / 'datasets' / 'ord-data-zip' / 'data'
    if not os.path.exists(dataset_root):
        datasets_dir = str(home_path / 'datasets')
        zip_file = wget.download(url='https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip', out=datasets_dir)
        fz = zipfile.ZipFile(zip_file, 'r')
        for file in fz.namelist():
            zip_dirname = os.path.split(file)[0]
            fz.extract(file, datasets_dir)
        fz.close()
        os.rename(os.path.join(datasets_dir, zip_dirname), os.path.join(datasets_dir, 'ord-data-zip'))
        assert os.path.exists(dataset_root)
    
    all_gz_paths = []
    for dir_name in sorted(os.listdir(dataset_root)):
        data_root = dataset_root / dir_name
        data = sorted(os.listdir(data_root))
        for i, gz_name in enumerate(data):
            all_gz_paths.append((f'{dir_name}--{i + 1}of{len(data)}', str(data_root / gz_name)))
    
    output_dir = home_path / 'datasets' / 'ord-data-json'
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


def tensorfy_json_data(args):
    rank, json_names = args
    jsons_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
    torch_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-torch'
    os.makedirs(torch_root, exist_ok=True)
    
    role2idx = {
        'REACTANT': 0, 'REATANT': 0, 'REAGENT': 0,
        'SOLVENT': 1,
        'CATALYST': 2, 'INTERNAL_STANDARD': 2,
        'OUTCOME': 3, 'PRODUCT': 3,
    }
    
    meta = defaultdict(list)
    for json_i, json_name in enumerate(json_names):
        torch_file = torch_root / json_name.replace('.json', '.pth')
        if os.path.exists(torch_file):  # skip this dataset
            print(f'{time_str()} [{json_name}]: already preprocessed !')
            continue
        
        with open(str(jsons_root / json_name), 'r') as fin:
            reactions: List[Dict[str, str]] = json.load(fin)

        bad_reaction_idx, bad_reaction_ids, reaction_ids = [], [], []
        all_mole_roles = []
        all_edge_index, all_edge_feat, all_atom_feat = [], [], []
        mole_offset, all_mole_offset = 0, [0]
        edge_offset, all_edge_offset = 0, [0]
        atom_offset, all_atom_offset = 0, [0]
        
        bar = tqdm.tqdm(reactions, desc=f'[{json_name} ({json_i+1:2d}/{len(json_names)})]', mininterval=1., dynamic_ncols=True)
        stt = time.time()
        for i, one_reaction in enumerate(bar):
            one_reaction: Dict[str, str]
            #              one_reaction['reaction_id'] example: ord-56b1f4bfeebc4b8ab990b9804e798aa7
            rid_as_a_str = one_reaction['reaction_id'].replace('ord-', '')
            rid_as_32_ints = [int(ch, base=16) for ch in rid_as_a_str]
            try:
                role_smiles_pairs = parse_one_reaction(one_reaction)
                rets = reaction2graphs(bar, role2idx, role_smiles_pairs, mole_offset, edge_offset, atom_offset)
            except:
                # no need to ROLLBACK xxx_offset, all_xxx, etc. (NOT UPDATED)
                bad_reaction_idx.append(i)
                bad_reaction_ids.append(rid_as_32_ints)
            else:
                # UPDATE xxx_offset, all_xxx, etc.
                reaction_ids.append(rid_as_32_ints)
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
        
        num_reactions = len(reaction_ids)
        buggy_dataset = num_reactions == 0
        if buggy_dataset:
            num_reactions = -1
        # per-reaction stats
        avg_mole_cnt, avg_edge_cnt, avg_atom_cnt = mole_offset / num_reactions, edge_offset / num_reactions, atom_offset / num_reactions
        meta['json_name'].append(json_name)
        meta['#R'].append(num_reactions)
        meta['#M_per_R'].append(round(avg_mole_cnt, 2))
        meta['#E_per_R'].append(round(avg_edge_cnt, 2))
        meta['#A_per_R'].append(round(avg_atom_cnt, 2))
        time_cost = time.time()-stt
        meta['cost'].append(round(time_cost, 2))
        meta['#bad_reac'].append(len(bad_reaction_ids))
        meta['buggy'].append(buggy_dataset)
        
        if buggy_dataset:    # buggy dataset!
            print(f'{time_str()} [{json_name}]: bad_dataset !')
            torch.save({'json_name': json_name}, str(torch_file) + '.bug')
            continue
        
        # show per-reaction stats
        print(f'{time_str()} [{json_name}]: #bad={len(bad_reaction_ids)}, #Mol={avg_mole_cnt:.2f}, #E={avg_edge_cnt:.2f}, #A={avg_atom_cnt:.2f}, cost={time_cost:.2f}s')
        if len(bad_reaction_ids):
            print(f'   ***** [{json_name}]: bad_reaction_idx={bad_reaction_idx}')
            rid_as_strs = []
            for rid_as_32_ints in bad_reaction_ids:
                rid_as_32_chars = [f'{i:x}' for i in rid_as_32_ints]
                rid_as_a_str = 'ord-' + ''.join(rid_as_32_chars)
                rid_as_strs.append(rid_as_a_str)
            print(f'   ***** [{json_name}]: bad_reaction_ids={rid_as_strs}')
        
        check_and_save(torch_file, json_name, bad_reaction_idx, bad_reaction_ids, reaction_ids, all_mole_offset, all_mole_roles, all_edge_offset, all_edge_index, all_edge_feat, all_atom_offset, all_atom_feat)
    
    return meta


def check_and_save(torch_file, json_name, bad_reaction_idx, bad_reaction_ids, reaction_ids, all_mole_offset, all_mole_roles, all_edge_offset, all_edge_index, all_edge_feat, all_atom_offset, all_atom_feat):
    tensors = {
        'json_name': json_name,
        'bad_reaction_idx': torch.tensor(bad_reaction_idx, dtype=torch.int32),
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
    
    # check
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
    
    # save
    torch.save(tensors, torch_file)


def reaction2graphs(bar, role2idx, role_smiles_pairs, mole_offset, edge_offset, atom_offset):
    react_mole_roles = []
    react_edge_index, react_edge_feat, react_atom_feat = [], [], []
    react_edge_offset, react_atom_offset = [], []
    for role, smiles in role_smiles_pairs:
        react_mole_roles.append(role2idx[role])
        mole_offset += 1
        
        #  (E, 2)     (E, 3)     (V, 9)
        edge_index, edge_feat, atom_feat = smiles2graph.smiles2graph(smiles)
        react_edge_index.append(edge_index), react_edge_feat.append(edge_feat), react_atom_feat.append(atom_feat)
        
        E, V = edge_index.shape[0], atom_feat.shape[0]
        bar.set_postfix_str(f'role={role:9s}, E={E:3d}, V={V:3d}', refresh=False)
        edge_offset += E
        react_edge_offset.append(edge_offset)
        atom_offset += V
        react_atom_offset.append(atom_offset)
    
    return mole_offset, edge_offset, atom_offset, react_mole_roles, react_edge_index, react_edge_feat, react_atom_feat, react_edge_offset, react_atom_offset


def parse_one_reaction(one_reaction: Dict[str, str]):
    role_smiles_pairs: Set[Tuple[str, str]] = set()
    for k, v in one_reaction.items():
        if k.startswith('inputs') and k.endswith('.type') and v.upper() == 'SMILES':
            role = one_reaction.get(k.split('identifiers[')[0] + 'reaction_role', 'REACTANT').upper()
        elif k.startswith('outcomes') and k.endswith('.type') and v.upper() == 'SMILES':
            role = 'OUTCOME'
        else:
            role = None
        if role is not None:
            for smiles in one_reaction[k.replace('.type', '.value')].split('.'):
                role_smiles_pairs.add((role, smiles))
    
    if len(role_smiles_pairs) == 0:
        for k, v in one_reaction.items():
            if v.upper() == 'REACTION_SMILES':
                left, mid, right = one_reaction[k.replace('.type', '.value')].split('>')
                for reactant in left.split('.'):
                    role_smiles_pairs.add(('REACTANT', reactant))
                for solvent_or_catalyst in mid.split('.'):
                    if any(x in solvent_or_catalyst for x in {'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Ag', 'Au', 'Pt', 'Pd'}):
                        role_smiles_pairs.add(('CATALYST', solvent_or_catalyst))
                    else:
                        role_smiles_pairs.add(('SOLVENT', solvent_or_catalyst))
                for outcome in right.split('.'):
                    role_smiles_pairs.add(('OUTCOME', outcome))
                break
        assert len(role_smiles_pairs) != 0
    
    return role_smiles_pairs


def main():
    # download_and_jsonfy_data()
    
    jsons_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
    global_json_names = os.listdir(jsons_root)
    random.shuffle(global_json_names)

    world_size = cpu_count()
    json_names = {rk: [] for rk in range(world_size)}
    for i, json_name in enumerate(global_json_names):
        json_names[i % world_size].append(json_name)
        
    with Pool(world_size) as pool:
        metas: List[Dict[str, List]] = list(pool.imap(tensorfy_json_data, [(rk, json_names[rk]) for rk in range(world_size)], chunksize=1))
    meta = {k: [] for k in metas[0].keys()}
    for m in metas:
        for k, v in m.items():
            meta[k].extend(v)
    
    meta = pandas.DataFrame(meta)
    meta = meta.sort_values(by=['json_name'])
    meta.to_csv('meta-' + datetime.datetime.now().strftime('%m%d_%H-%M-%S') + '.csv')


if __name__ == '__main__':
    main()