import datetime
import glob
import json
import os
import pathlib
import random
import re
import sys
import time
import zipfile
from collections import Counter, OrderedDict
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing import cpu_count
from typing import List, Dict, Tuple, Set, Union

import pandas
import torch
import tqdm
import wget
# install ord_schema following https://github.com/open-reaction-database/ord-schema
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2
# conda install -c rdkit rdkit
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')

import smiles2graph


atom_index_p = re.compile(':\d+\]')
rm_duplicated_molecule = True

role2idx = {
    'REACTANT': 0, 'REATANT': 0,
    'REAGENT': 1, 'SOLVENT': 1,
    'CATALYST': 2, 'INTERNAL_STANDARD': 2,
    'OUTCOME': 3, 'PRODUCT': 3,
}
idx2role = {}
for k, v in role2idx.items():
    if v not in idx2role:
        idx2role[v] = k


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


def canonicalize_smiles(smiles: str):
    return Chem.CanonSmiles(smiles) if len(smiles) else ''


def tensorfy_json_data(args):
    json_name, blacklists, with_set = args
    based_on_canonicalized_reactions = isinstance(json_name, list)
    print(f'{time_str()} based_on_canonicalized_reactions={based_on_canonicalized_reactions}')
    if based_on_canonicalized_reactions: # based_on_canonicalized_reactions
        reactions, uspto_root, with_set = args
        json_name = uspto_root
        reactions: List[str]
        uspto_root: str
        torch_file = os.path.join(uspto_root, 'tensor.pth')
        stats_file = os.path.join(uspto_root, 'stats.json')
        blacklist = set()
        
    else:
        blacklist = set()
        for b in blacklists:
            with open(b, 'r') as fp:
                blacklist |= set(json.load(fp))
        print(f'{time_str()} blacklists={blacklists} ({len(blacklist)}, e.g. {sorted(list(blacklist))[0]})')
        
        jsons_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
        torch_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-torch'
        stats_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-stats'
        os.makedirs(torch_root, exist_ok=True)
        os.makedirs(stats_root, exist_ok=True)
        
        torch_file = torch_root / json_name.replace('.json', '.pth')
        stats_file = stats_root / json_name
        
        with open(str(jsons_root / json_name), 'r') as fin:
            reactions: List[Dict[str, str]] = json.load(fin)

    if os.path.exists(torch_file):  # skip this dataset
        print(f'{time_str()} [{json_name}]: already preprocessed !', flush=True)
        return {}
    else:
        print(f'{time_str()} torch_file={torch_file}, stats_file={stats_file}')

    meta = defaultdict(list)
    bad_reaction_idx, bad_reaction_ids = [], []
    blk_reaction_idx, blk_reaction_ids = [], []
    reaction_ids = []
    all_mole_roles, all_mole_counts = [], []
    all_edge_index, all_edge_feat, all_atom_feat = [], [], []
    mole_offset, all_mole_offset = 0, [0]
    edge_offset, all_edge_offset = 0, [0]
    atom_offset, all_atom_offset = 0, [0]
    
    bar = tqdm.tqdm(reactions, desc=f'[{json_name}]', mininterval=10., maxinterval=20, dynamic_ncols=True)
    stt = time.time()
    num_atoms_diff_stats = defaultdict(int)
    for i, one_reaction in enumerate(bar):
        one_reaction: Union[Dict[str, str], str]
        
        if based_on_canonicalized_reactions:
            rid_as_32_ints = [0] * 16
        else:
            #              one_reaction['reaction_id'] example: ord-56b1f4bfeebc4b8ab990b9804e798aa7
            rid_as_a_str = one_reaction['reaction_id'].replace('ord-', '')
            rid_as_32_ints = [int(ch, base=16) for ch in rid_as_a_str]
        
        try:
            if based_on_canonicalized_reactions:
                roles_smiles = reaction_smiles_to_roles_smiles(one_reaction)
                R, C, S, O, canonicalized_reaction, num_atoms_diff = roles_smiles_to_reaction_smiles(roles_smiles)
                role_smiles_pairs = role_smiles_to_role_smiles_pairs(R, C, S, O)
                rets = reaction2graphs(bar, with_set, role_smiles_pairs, mole_offset, edge_offset, atom_offset)
            else:
                role_smiles_pairs, num_atoms_diff = parse_one_reaction_dict(one_reaction, blacklist)
                if role_smiles_pairs is None:  # in blacklist
                    rets = None
                else:
                    rets = reaction2graphs(bar, with_set, role_smiles_pairs, mole_offset, edge_offset, atom_offset)
        except:
            # no need to ROLLBACK xxx_offset, all_xxx, etc. (NOT UPDATED)
            bad_reaction_idx.append(i)
            bad_reaction_ids.append(rid_as_32_ints)
        else:
            if rets is None:  # in blacklist
                blk_reaction_idx.append(i)
                blk_reaction_ids.append(rid_as_32_ints)
                continue
            # UPDATE xxx_offset, all_xxx, etc.
            reaction_ids.append(rid_as_32_ints)
            (
                mole_offset, edge_offset, atom_offset,
                react_mole_roles, react_mole_counts,
                react_edge_index, react_edge_feat, react_atom_feat,
                react_edge_offset, react_atom_offset,
            ) = rets
            num_atoms_diff_stats[num_atoms_diff] += 1
            all_mole_roles.extend(react_mole_roles)
            all_mole_counts.extend(react_mole_counts)
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
    meta['#R_blk'].append(len(blk_reaction_ids))
    meta['#M_per_R'].append(round(avg_mole_cnt, 2))
    meta['#E_per_R'].append(round(avg_edge_cnt, 2))
    meta['#A_per_R'].append(round(avg_atom_cnt, 2))
    time_cost = time.time()-stt
    meta['cost'].append(round(time_cost, 2))
    meta['#bad_reac'].append(len(bad_reaction_ids))
    meta['buggy'].append(buggy_dataset)
    
    if buggy_dataset:    # buggy dataset!
        print(f'{time_str()} [{json_name}]: bad_dataset !', flush=True)
        torch.save({'json_name': json_name}, str(torch_file) + '.bug')
        return meta
    
    # show per-reaction stats
    print(f'{time_str()} [{json_name}]: #bad={len(bad_reaction_ids)}, #blk={len(blk_reaction_ids)} #Mol={avg_mole_cnt:.2f}, #E={avg_edge_cnt:.2f}, #A={avg_atom_cnt:.2f}, cost={time_cost:.2f}s', flush=True)
    if len(bad_reaction_ids):
        print(f'   ***** [{json_name}]: bad_reaction_idx={bad_reaction_idx}', flush=True)
        rid_as_strs = []
        for rid_as_32_ints in bad_reaction_ids:
            rid_as_32_chars = [f'{i:x}' for i in rid_as_32_ints]
            rid_as_a_str = 'ord-' + ''.join(rid_as_32_chars)
            rid_as_strs.append(rid_as_a_str)
        print(f'   ***** [{json_name}]: bad_reaction_ids={rid_as_strs}', flush=True)
    
    check_and_save(torch_file, json_name, bad_reaction_idx, bad_reaction_ids, blk_reaction_idx, blk_reaction_ids, reaction_ids, all_mole_offset, all_mole_roles, all_mole_counts, all_edge_offset, all_edge_index, all_edge_feat, all_atom_offset, all_atom_feat)
    print(f'{time_str()} torch_file saved @ {torch_file}')
    
    # save stats
    num_atoms_diff_stats = {
        diff: f'{freq} ({100.*freq/num_reactions:.3f}%)'
        for diff, freq in sorted(num_atoms_diff_stats.items())
    }
    with open(stats_file, 'w') as fp:
        json.dump({
            'num_atoms_diff_stats': num_atoms_diff_stats,
        }, fp, indent=2)
    print(f'{time_str()} stats_file saved @ {stats_file}')
    
    return meta


def check_and_save(torch_file, json_name, bad_reaction_idx, bad_reaction_ids, blk_reaction_idx, blk_reaction_ids, reaction_ids, all_mole_offset, all_mole_roles, all_mole_counts, all_edge_offset, all_edge_index, all_edge_feat, all_atom_offset, all_atom_feat):
    tensors = {
        'json_name': json_name,
        'bad_reaction_idx': torch.tensor(bad_reaction_idx, dtype=torch.int32),
        'bad_reaction_ids': torch.tensor(bad_reaction_ids, dtype=torch.uint8),
        'blk_reaction_idx': torch.tensor(blk_reaction_idx, dtype=torch.int32),
        'blk_reaction_ids': torch.tensor(blk_reaction_ids, dtype=torch.uint8),
        'reaction_ids': torch.tensor(reaction_ids, dtype=torch.uint8),
        'mole_offset': torch.tensor(all_mole_offset, dtype=torch.int32),
        'mole_roles': torch.tensor(all_mole_roles, dtype=torch.int32),
        'mole_counts': torch.tensor(all_mole_counts, dtype=torch.int32),
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
    assert max_mid == tensors['mole_counts'].size(0) + 1
    
    max_aid = tensors['atom_features'].size(0)
    assert max_aid == tensors['atom_offset'][-1]
    
    max_eid = tensors['edge_features'].size(0)
    assert max_eid == tensors['edge_offset'][-1]
    assert max_eid == tensors['edge_indices'].size(0)
    
    # save
    torch.save(tensors, torch_file)


def reaction2graphs(bar, with_set, role_smiles_pairs, mole_offset, edge_offset, atom_offset):
    react_mole_roles, react_mole_counts = [], []
    react_edge_index, react_edge_feat, react_atom_feat = [], [], []
    react_edge_offset, react_atom_offset = [], []
    
    if with_set:
        unique = OrderedDict()
        for p in role_smiles_pairs:
            if p not in unique:
                unique[p] = 1
            else:
                unique[p] += 1
        role_smiles_pairs = [(count, p) for p, count in unique.items()]
    else:
        role_smiles_pairs = [(1, p) for p in role_smiles_pairs]
    
    for count, (role, smiles) in role_smiles_pairs:
        react_mole_roles.append(role)
        react_mole_counts.append(count)
        mole_offset += 1
        
        #  (E, 2)     (E, 3)     (V, 9)
        edge_index, edge_feat, atom_feat = smiles2graph.smiles2graph(smiles)
        react_edge_index.append(edge_index), react_edge_feat.append(edge_feat), react_atom_feat.append(atom_feat)
        
        E, V = edge_index.shape[0], atom_feat.shape[0]
        assert V > 0
        bar.set_postfix_str(f'role={idx2role[role]:9s}, E={E:3d}, V={V:3d}', refresh=False)
        edge_offset += E
        react_edge_offset.append(edge_offset)
        atom_offset += V
        react_atom_offset.append(atom_offset)
    
    return mole_offset, edge_offset, atom_offset, react_mole_roles, react_mole_counts, react_edge_index, react_edge_feat, react_atom_feat, react_edge_offset, react_atom_offset


def parse_one_reaction_dict(one_reaction: Dict[str, str], blacklist: Set[str]):
    roles_smiles = {v: [] for v in role2idx.values()}
    for k, v in one_reaction.items():
        if k.startswith('inputs') and k.endswith('.type') and v.upper() == 'SMILES':
            role = one_reaction.get(k.split('identifiers[')[0] + 'reaction_role', 'REACTANT').upper()
        elif k.startswith('outcomes') and k.endswith('.type') and v.upper() == 'SMILES':
            role = 'OUTCOME'
        else:
            role = None
        if role is not None:
            for smiles in one_reaction[k.replace('.type', '.value')].split('.'):
                smiles = smiles.strip()
                if len(smiles):
                    roles_smiles[role2idx[role]].append(smiles)
    
    if len(roles_smiles[0]) == 0 or len(roles_smiles[3]) == 0:
        for k, v in one_reaction.items():
            if v.upper() in {'REACTION_SMILES', 'REACTION_CXSMILES'}:
                roles_smiles = reaction_smiles_to_roles_smiles(one_reaction[k.replace('.type', '.value')].split('|')[0].strip())
                break

    assert len(roles_smiles[0]) != 0 and len(roles_smiles[3]) != 0

    R, C, S, O, s, num_atoms_diff = roles_smiles_to_reaction_smiles(roles_smiles)
    if s in blacklist:
        return None, None
    return role_smiles_to_role_smiles_pairs(R, C, S, O), num_atoms_diff


def role_smiles_to_role_smiles_pairs(R: str, C: str, S: str, O: str):
    role_smiles_pairs: List[Tuple[int, str]] = []
    assert R and O
    for x in filter(len, R.split('.')):
        role_smiles_pairs.append((role2idx['REACTANT'], x))
    for x in filter(len, C.split('.')):
        role_smiles_pairs.append((role2idx['CATALYST'], x))
    for x in filter(len, S.split('.')):
        role_smiles_pairs.append((role2idx['SOLVENT'], x))
    for x in filter(len, O.split('.')):
        role_smiles_pairs.append((role2idx['OUTCOME'], x))
    return role_smiles_pairs


def roles_smiles_to_reaction_smiles(roles_smiles):
    
    R, C, S, O = '.'.join(roles_smiles[0]), '.'.join(roles_smiles[1]), '.'.join(roles_smiles[2]), '.'.join(roles_smiles[3])
    
    # if atom_index_p is not None:
    #     R = atom_index_p.sub(']', R)
    #     C = atom_index_p.sub(']', C)
    #     S = atom_index_p.sub(']', S)
    #     O = atom_index_p.sub(']', O)

    mR, mC, mS, mO = map(Chem.MolFromSmiles, (R, C, S, O))
    for mo in (mR, mC, mS, mO):
        for a in mo.GetAtoms():
            a.SetAtomMapNum(0)
    
    # num_atoms_diff = R.GetNumHeavyAtoms() - O.GetNumHeavyAtoms()    # 统计反应物原子数-生成物原子数
    # num_atoms_diff = len(roles_smiles[3])    # 统计生成物分子数
    R, C, S, O = map(Chem.CanonSmiles, map(Chem.MolToSmiles, (mR, mC, mS, mO)))
    
    num_atoms_diff = 0    # 统计duplicated molecule数
    Rs = set(R.split('.'))
    new_O = []
    for o in O.split('.'):
        if o in Rs:
            num_atoms_diff += 1
        else:
            new_O.append(o)
    if rm_duplicated_molecule:
        O = Chem.CanonSmiles('.'.join(new_O))   # todo: rm_duplicated_molecule

    num_atoms_diff = len(new_O) == 0   # 统计空白O的反应个数
    
    s = R + '>' + '.'.join(filter(len, (C, S))) + '>' + O
    return R, C, S, O, s, num_atoms_diff


def reaction_smiles_to_roles_smiles(reaction_smiles: str):
    roles_smiles = {v: [] for v in role2idx.values()}
    #
    # rxn = Chem.rdChemReactions.ReactionFromSmarts(reaction_smiles)
    # rxn.ClearComputedProps()
    # mols = []
    # for mol in rxn.GetProducts():
    #     for atom in mol.GetAtoms():
    #         atom.SetAtomMapNum(0)
    #     mols.append(mol)
    # smi = '.'.join(Chem.MolToSmiles(mol) for mol in mols)
    # smi = Chem.CanonSmiles(smi)
    
    R_str, mid, O_str = reaction_smiles.strip().split('>')
    roles_smiles[0] = list(filter(len, map(str.strip, R_str.split('.'))))
    mid = '.'.join(filter(len, map(str.strip, mid.split('.'))))
    roles_smiles[3] = list(filter(len, map(str.strip, O_str.split('.'))))
    if len(mid) > 0:
        for catalyst_or_solvent in mid.split('.'):
            if any(x in catalyst_or_solvent for x in {
                'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                'W', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                'Pb', 'Bi',
            }):
                roles_smiles[2].append(catalyst_or_solvent)
            else:
                roles_smiles[1].append(catalyst_or_solvent)

    def clamp(l: List[str], max_n: int):
        nl = []
        for k, v in Counter(l).items():
            if v > max_n:
                v = 1
            nl.extend([k] * v)
        return nl

    roles_smiles[0] = clamp(roles_smiles[0], max_n=6)
    roles_smiles[1] = clamp(roles_smiles[1], max_n=6)
    roles_smiles[2] = clamp(roles_smiles[2], max_n=6)
    roles_smiles[3] = clamp(roles_smiles[3], max_n=6)
    
    return roles_smiles


def main():
    # download_and_jsonfy_data()
    dataset, with_set, files = sys.argv[1], sys.argv[2] in {'1', 'True', 'true', 'set', 'with_set'}, list(map(os.path.expanduser, sys.argv[3:]))
    prepare_uspto = dataset.startswith('uspto')

    if os.path.isdir(files[0]):
        files = glob.glob(os.path.join(files[0], '*.txt')) + glob.glob(os.path.join(files[0], '*.csv'))

    print(f'[dataset  ] {dataset}')
    print(f'[with_set ] {with_set}')
    print(f'[files    ] {files}')
    
    if prepare_uspto:
        uspto_root = os.path.dirname(files[0])
        output = os.path.join(uspto_root, 'blacklist.json')
        if os.path.exists(output):
            print(f'{time_str()} output file {output} already exists, returning!')
            return

        reactions = []
        for f in files:
            with open(f, 'r') as fin:
                reactions.extend([l.split(' ')[0].split(',')[-1] for l in fin.read().splitlines()])
            print(f'[after load {f}] len(reactions)={len(reactions)}')
        canonicalized_reactions = []
        for r in tqdm.tqdm(reactions, desc=f'[read]', dynamic_ncols=True, mininterval=2., maxinterval=10):
            roles_smiles = reaction_smiles_to_roles_smiles(r)
            _, _, _, _, canonicalized_reaction, _ = roles_smiles_to_reaction_smiles(roles_smiles)
            canonicalized_reactions.append(canonicalized_reaction)
        
        with open(output, 'w') as fp:
            json.dump(canonicalized_reactions, fp, indent=2)
        print(f'{time_str()} output file saved @ {output}')

        meta = tensorfy_json_data((canonicalized_reactions, uspto_root, with_set))
    
    else:
        jsons_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
        global_json_names = os.listdir(jsons_root)
        random.shuffle(global_json_names)
        args = [(n, files, with_set) for n in global_json_names]
        
        # todo: dbg
        # metas = [tensorfy_json_data(a) for a in args]
        
        world_size = cpu_count()
        with Pool(world_size) as pool:
            metas: List[Dict[str, List]] = list(pool.imap(tensorfy_json_data, args, chunksize=1))
        
        meta = defaultdict(list)
        for m in metas:
            for k, v in m.items():
                meta[k].extend(v)
    
    meta = pandas.DataFrame(meta)
    meta = meta.sort_values(by=['json_name'])
    meta.to_csv(f'meta-set{int(with_set)}-{dataset}-{datetime.datetime.now().strftime("%m%d_%H-%M-%S")}.csv')


if __name__ == '__main__':
    main()
