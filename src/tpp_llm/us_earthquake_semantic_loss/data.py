"""
Dataset and Data Loader for the TPP-LLM
"""
import json
import os
import random
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset

#torch.manual_seed(0)

class TPPLLMDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch: dict) -> dict:
    """
    batch: items in a batch
    mag_dist: frozen distribution object for magnitude
    dep_dist: frozen distribution object for depth
    """
    
    return {
        'time_since_start': [torch.FloatTensor(item['time_since_start']) for item in batch],
        'time_since_last_event': [torch.FloatTensor(item['time_since_last_event']) for item in batch],
        'type_event': [torch.LongTensor(item['type_event']) for item in batch],
        'type_text': [item['type_text'] for item in batch],
        'magnitude' : [torch.FloatTensor(item['magnitude']) for item in batch],
        'depth': [torch.FloatTensor(item['depth']) for item in batch]

    }

def create_few_shot_dataset(data_dir, output_dir, few_shot_ratio=0.1, seed=0) -> None:
    random.seed(seed)
    train_file = os.path.join(data_dir, 'train.json')
    dev_file = os.path.join(data_dir, 'dev.json')
    test_file = os.path.join(data_dir, 'test.json')
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    few_shot_size = int(len(train_data) * few_shot_ratio)
    few_shot_train_data = random.sample(train_data, few_shot_size)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as file:
        json.dump(few_shot_train_data, file, indent=4)
    shutil.copy(dev_file, os.path.join(output_dir, 'dev.json'))
    shutil.copy(test_file, os.path.join(output_dir, 'test.json'))
    print(f"Few-shot dataset created in {output_dir}.")


def get_all_data_stats(dataset_dir):
    """
    Reads all Train/Val/Test files and calculates the global 
    'maximum' and 'minimum' values across the entire dataset.
    Calculates 'time difference (Delta)' in addition to magnitude, depth, and time.
    """
    print(f"Calculating global stats (min/max) from all files in {dataset_dir}...")

    stats = {
        'mag': {'min': float('inf'), 'max': float('-inf')},
        'dep': {'min': float('inf'), 'max': float('-inf')},
        'time': {'min': float('inf'), 'max': float('-inf')},
        'delta': {'min': float('inf'), 'max': float('-inf')}
    }

    target_files = ['train.json', 'dev.json', 'test.json'] 

    for fname in target_files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            print(f"  - {fname} not found, skipping.")
            continue
        
        print(f"  - Scanning {fname}...")
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            for seq in data:
                m_list = seq.get('magnitude', [])
                if m_list:
                    stats['mag']['min'] = min(stats['mag']['min'], min(m_list))
                    stats['mag']['max'] = max(stats['mag']['max'], max(m_list))
                
                d_list = seq.get('depth', [])
                if d_list:
                    stats['dep']['min'] = min(stats['dep']['min'], min(d_list))
                    stats['dep']['max'] = max(stats['dep']['max'], max(d_list))
                
                t_list = seq.get('time_since_start', [])
                if t_list:
                    stats['time']['min'] = min(stats['time']['min'], min(t_list)) 
                    stats['time']['max'] = max(stats['time']['max'], max(t_list))

                dt_list = seq.get('time_since_last_event', [])
                if dt_list:
                    stats['delta']['min'] = min(stats['delta']['min'], min(dt_list))
                    stats['delta']['max'] = max(stats['delta']['max'], max(dt_list))
                
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    if stats['mag']['max'] == float('-inf'): stats['mag'] = {'min': -2.0, 'max': 10.0}
    if stats['dep']['max'] == float('-inf'): stats['dep'] = {'min': 0.0, 'max': 700.0}
    if stats['time']['max'] == float('-inf'): stats['time'] = {'min': 0.0, 'max': 100.0}
    if stats['delta']['max'] == float('-inf'): stats['delta'] = {'min': 0.0, 'max': 100.0}

    print(f"Global Stats Result:")
    print(f"  > Magnitude: {stats['mag']['min']} ~ {stats['mag']['max']}")
    print(f"  > Depth:     {stats['dep']['min']} ~ {stats['dep']['max']}")
    print(f"  > Time:      {stats['time']['min']} ~ {stats['time']['max']}")
    print(f"  > Delta:     {stats['delta']['min']} ~ {stats['delta']['max']}")
    
    return stats