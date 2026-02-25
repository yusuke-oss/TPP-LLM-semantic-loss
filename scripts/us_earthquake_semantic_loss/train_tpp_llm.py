"""
Train a TPP-LLM Model
"""
# ... (importsはそのまま) ...
import argparse
import os.path
import sys
import pprint
import json
import time
import random
from functools import partial
import numpy as np
import pandas as pd
import torch
import transformers
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from sklearn.utils.class_weight import compute_class_weight

# Local Imports
from src.tpp_llm.us_earthquake_semantic_loss.data import (
    TPPLLMDataset, collate_fn, create_few_shot_dataset
)
from src.tpp_llm.us_earthquake_semantic_loss.model import TPPLLMModel
from src.tpp_llm.us_earthquake_semantic_loss.runner import TPPLLMRunner
from src.tpp_llm.us_earthquake_semantic_loss.utils import get_prompt
from src.tpp_llm.us_earthquake_semantic_loss.common_utils import seed_everything, seed_worker
# MinMaxEncoderは使わないので削除またはコメントアウト
# from src.tpp_llm.us_earthquake30.encoders import MinMaxEncoder 
from src.tpp_llm.us_earthquake_semantic_loss.analysis import (
    perform_quantitative_analysis
)

pprint.pprint(sys.path)
torch.autograd.set_detect_anomaly(True)

start_time = time.time()
def get_all_data_stats(dataset_dir):
    """
    ★修正: Train/Val/Test すべてのファイルを読み込んで、データセット全体の
    「最大値」だけでなく「最小値」も計算する関数。
    マグニチュード、深さ、時刻に加えて「時間差 (Delta)」も計算。
    """
    print(f"Calculating global stats (min/max) from all files in {dataset_dir}...")

    # 初期値設定 (Minは無限大, Maxは無限小で初期化)
    stats = {
        'mag': {'min': float('inf'), 'max': float('-inf')},
        'dep': {'min': float('inf'), 'max': float('-inf')},
        'time': {'min': float('inf'), 'max': float('-inf')},
        'delta': {'min': float('inf'), 'max': float('-inf')} # ★追加: 時間差用
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
                # Magnitude
                m_list = seq.get('magnitude', [])
                if m_list:
                    stats['mag']['min'] = min(stats['mag']['min'], min(m_list))
                    stats['mag']['max'] = max(stats['mag']['max'], max(m_list))
                
                # Depth
                d_list = seq.get('depth', [])
                if d_list:
                    stats['dep']['min'] = min(stats['dep']['min'], min(d_list))
                    stats['dep']['max'] = max(stats['dep']['max'], max(d_list))
                
                # Time (累積なので0〜最後の値)
                t_list = seq.get('time_since_start', [])
                if t_list:
                    stats['time']['min'] = min(stats['time']['min'], min(t_list)) 
                    stats['time']['max'] = max(stats['time']['max'], max(t_list))

                # ★追加: Delta (Time Interval)
                dt_list = seq.get('time_since_last_event', [])
                if dt_list:
                    stats['delta']['min'] = min(stats['delta']['min'], min(dt_list))
                    stats['delta']['max'] = max(stats['delta']['max'], max(dt_list))
                
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # データが空だった場合の安全策 (デフォルト値)
    if stats['mag']['max'] == float('-inf'): stats['mag'] = {'min': -2.0, 'max': 10.0}
    if stats['dep']['max'] == float('-inf'): stats['dep'] = {'min': 0.0, 'max': 700.0}
    if stats['time']['max'] == float('-inf'): stats['time'] = {'min': 0.0, 'max': 100.0}
    if stats['delta']['max'] == float('-inf'): stats['delta'] = {'min': 0.0, 'max': 100.0} # ★追加

    print(f"Global Stats Result:")
    print(f"  > Magnitude: {stats['mag']['min']} ~ {stats['mag']['max']}")
    print(f"  > Depth:     {stats['dep']['min']} ~ {stats['dep']['max']}")
    print(f"  > Time:      {stats['time']['min']} ~ {stats['time']['max']}")
    print(f"  > Delta:     {stats['delta']['min']} ~ {stats['delta']['max']}") # ★追加
    
    return stats

if __name__ == '__main__':
    # ... (argparse部分は変更なし) ...
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train and test the TPP-LLM model with event sequences.')
    parser.add_argument(
        '--model_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='llm path')
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument(
        '--num_event_types', type=int, required=True, help='number of event types')
    parser.add_argument(
        '--temporal_emb_type', type=str, default='positional', choices=['linear', 'positional', 'shifted'],
        help='temporal embedding type')
    parser.add_argument(
        '--temporal_emb_first', action='store_true', help='temporal embedding first or not')
    parser.add_argument(
        '--no_prompt', action='store_true', help='no prompt')
    parser.add_argument(
        '--num_integral_samples', type=int, default=20, help='number of samples during one integral step')
    parser.add_argument(
        '--quant_type', type=str, default=None, choices=['4bit', '8bit', None], help='quantization type')
    parser.add_argument(
        '--peft_type', type=str, default=None, choices=['lora', None], help='peft type')
    parser.add_argument(
        '--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument(
        '--lora_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        help='lora target modules')
    parser.add_argument(
        '--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument(
        '--eval_batch_size', type=int, default=16, help='batch size for evaluation')
    parser.add_argument(
        '--learning_rate', type=float, default=5e-4, help='larning rate')
    parser.add_argument(
        '--lr_scheduler_type', type=str, default='constant', help='learning rate scheduler type')
    parser.add_argument(
        '--num_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument(
        '--warmup_ratio', type=float, default=0, help='warmup ratio')
    parser.add_argument(
        '--beta_type', type=float, default=1, help='loss coefficient of the event type prediction')
    parser.add_argument(
        '--beta_time', type=float, default=1, help='loss coefficient of the event time prediction')
    parser.add_argument(
        '--device', type=str, default='cpu', help='cpu or cuda device')
    parser.add_argument(
        '--seed', type=int, default=2024, help='seed for reproducibility')
    parser.add_argument(
        '--result_save_path', type=str, default='',help='save result')
    parser.add_argument(
        '--model_weight_path', type=str, default='',help='save model weight')
    parser.add_argument(
        '--weight_path', type=str,default='', help='save weight')
    parser.add_argument(
        '--figure_path', type=str,default='', help='save figure')
    parser.add_argument(
        '--save_flag',  action='store_true', default=False, help='save weight and model')
    parser.add_argument(
        '--load_flag',  action='store_true', default=False, help='load weight and model')
    parser.add_argument(
        '--train_flag',  action='store_true', default=False, help='train weight and model')
    parser.add_argument('--alpha_concept', type=float, default=10000.0, help='weight for concept anchor loss')
    


    # Parse arguments
    args = parser.parse_args()
    print(f'args: {args}')

    f = open(args.result_save_path+'/val.txt', 'a')
    f.write(f'seed {args.seed}\n')
    f.write(f'alpha_concept: {args.alpha_concept}\n')
    
    
    f.close()

    print(f'alpha_concept: {args.alpha_concept}\n')
    
    print(f'lr_scheduler_type: {args.lr_scheduler_type}\n')
    print(f'warmup_ratio: {args.warmup_ratio}\n')

    # --- ▼ ここに移動 ▼ ---
    # argparse で受け取ったシードを使って、全ての乱数を固定する
    seed_everything(args.seed)
    
    #transformers.set_seed(args.seed)
    base_dataset_name = os.path.basename(args.dataset_path).replace('_few_shot', '')
    prompt = get_prompt(dataset_name=base_dataset_name, event_time_first=args.temporal_emb_first)
    if args.no_prompt:
        prompt = ''
    #print(f'prompt: {prompt}')

    f = open(args.result_save_path+'/val.txt', 'a')
    f.write(f'prompt: {prompt}\n')
    f.close()

    # Get the quantization config
    if args.quant_type == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quant_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=False,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Get the PEFT config
    if args.peft_type == 'lora':
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=args.lora_modules,
            #lora_dropout=0.05,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        peft_config = None

    

    

    # 3. DataLoaderの設定
    dataset_train = TPPLLMDataset(f'{args.dataset_path}/train.json')
    dataset_val = TPPLLMDataset(f'{args.dataset_path}/dev.json')
    dataset_test = TPPLLMDataset(f'{args.dataset_path}/test.json')

    # ▼▼▼ collate_fn に分布オブジェクト(CDF計算用)を渡す ▼▼▼
    #collate_with_cdf = partial(collate_fn, mag_dist=mag_frozen_dist, dep_dist=dep_frozen_dist)

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, 
        generator=g, worker_init_fn=seed_worker, collate_fn=collate_fn
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    
    
    stats = get_all_data_stats(args.dataset_path)

    model = TPPLLMModel(
        model_name=args.model_path,
        num_event_types=args.num_event_types,
        num_integral_samples=args.num_integral_samples,
        temporal_emb_type=args.temporal_emb_type,
        temporal_emb_first=args.temporal_emb_first,
        prompt=prompt,
        bnb_config=bnb_config,
        peft_config=peft_config,
        device=args.device,
        model_weight_path=args.model_weight_path,
        save_flag=args.save_flag,
        load_flag=args.load_flag,
        train_flag=args.train_flag,
        # ▼▼▼ 追加: ここで渡す！ ▼▼▼
        
        alpha_concept=args.alpha_concept, # ★追加
        
    )

    # ★★★ 重要: 最小値もモデルに属性として注入する (可視化関数で使うため) ★★★
    # (Modelの__init__引数になくても、ここで属性セットすれば動きます)
    model.max_mag = stats['mag']['max']
    model.max_dep = stats['dep']['max']
    model.max_time = stats['time']['max']
    model.min_mag = stats['mag']['min']
    model.min_dep = stats['dep']['min']
    model.min_time = stats['time']['min']
    model.min_delta = stats['delta']['min']
    model.max_delta = stats['delta']['max']
    

    # 3. DataLoaderの設定
    dataset_train = TPPLLMDataset(f'{args.dataset_path}/train.json')
    dataset_val = TPPLLMDataset(f'{args.dataset_path}/dev.json')
    dataset_test = TPPLLMDataset(f'{args.dataset_path}/test.json')

    # ▼▼▼ collate_fn に分布オブジェクト(CDF計算用)を渡す ▼▼▼
    #collate_with_cdf = partial(collate_fn, mag_dist=mag_frozen_dist, dep_dist=dep_frozen_dist)

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, 
        generator=g, worker_init_fn=seed_worker, collate_fn=collate_fn
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Runner実行
    runner = TPPLLMRunner(
        model=model,
        beta_type=args.beta_type,
        beta_time=args.beta_time,
        device=args.device,
        weight_path=args.weight_path,
        load_flag=args.load_flag,
    )
    
    
    runner.run(
        quantitative_analysis_func=perform_quantitative_analysis,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_test=dataloader_test,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        result_save_path=args.result_save_path,
        weight_path=args.weight_path,
        model_weight_path=args.model_weight_path,
        figure_path=args.figure_path,
        save_flag=args.save_flag,
        train_flag=args.train_flag,# ★★★ ここに「4」を追加してください！ ★★★
        gradient_accumulation_steps=4
    )
    

# 処理
end_time = time.time()
f = open(args.result_save_path+'/val.txt', 'a')
f.write(f'validation metrics: {end_time - start_time}seconds\n')
f.close()

print(f"Execution time: {end_time - start_time} seconds")