import datetime
import json
import os
import pathlib
import time

import pandas as pd
import tqdm

from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2

from smile2graph import smiles2graph


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
    dataset_root = pathlib.Path(os.path.expanduser('~')) / 'datasets' / 'ord-data-json'
    meta = {'json_name': [], 'keys': []}
    for json_name in sorted(os.listdir(dataset_root)):
        json_path = str(dataset_root / json_name)
        with open(json_path, 'r') as fin:
            dataset = json.load(fin)
        keys = sorted(list(set(dataset[0].keys())))
        meta['json_name'].append(json_name)
        meta['keys'].append('| '.join(keys))
    
    meta = pd.DataFrame(meta)
    meta.to_csv('ord_meta.csv')
    
    return meta


def cast(item: dict):
    smiles2graph()


if __name__ == '__main__':
    jsonfy_ord()
