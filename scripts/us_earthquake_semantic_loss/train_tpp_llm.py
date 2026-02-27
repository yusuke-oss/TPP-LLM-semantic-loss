"""
Train a TPP-LLM Model
"""
import argparse
import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType

# Local Imports
# Note: get_all_data_stats should be moved to data.py
from src.tpp_llm.us_earthquake_semantic_loss.data import (
    TPPLLMDataset, collate_fn, get_all_data_stats
)
from src.tpp_llm.us_earthquake_semantic_loss.model import TPPLLMModel
from src.tpp_llm.us_earthquake_semantic_loss.runner import TPPLLMRunner
from src.tpp_llm.us_earthquake_semantic_loss.utils import get_prompt
from src.tpp_llm.us_earthquake_semantic_loss.common_utils import seed_everything, seed_worker
from src.tpp_llm.us_earthquake_semantic_loss.analysis import perform_quantitative_analysis

# Enable anomaly detection for debugging (Can be set to False for production)
torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    # ... (argparse部分は変更なし) ...
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train and test the TPP-LLM model with event sequences.')
    # Model & Data Configuration
    parser.add_argument(
        '--model_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='llm path')
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument(
        '--num_event_types', type=int, required=True, help='number of event types')
    parser.add_argument(
        '--temporal_emb_type', type=str, default='positional', choices=['positional','MLP'],
        help='temporal embedding type')
    parser.add_argument(
        '--temporal_emb_first', action='store_true', help='temporal embedding first or not')
    parser.add_argument(
        '--no_prompt', action='store_true', help='no prompt')
    parser.add_argument(
        '--num_integral_samples', type=int, default=20, help='number of samples during one integral step')
    # Quantization & PEFT Configuration
    parser.add_argument(
        '--quant_type', type=str, default=None, choices=['4bit', '8bit', None], help='quantization type')
    parser.add_argument(
        '--peft_type', type=str, default=None, choices=['lora', None], help='peft type')
    parser.add_argument(
        '--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument(
        '--lora_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        help='lora target modules')
    # Training Hyperparameters
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
        '--beta_semantic', type=float, default=0.0, help='weight for concept anchor loss')
    # Environment & Save Paths
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
    # Execution Flags
    parser.add_argument(
        '--save_flag',  action='store_true', default=False, help='save weight and model')
    parser.add_argument(
        '--load_flag',  action='store_true', default=False, help='load weight and model')
    parser.add_argument(
        '--train_flag',  action='store_true', default=False, help='train weight and model')
    
    start_time = time.time()
    # Parse arguments
    args = parser.parse_args()
    
    print(f"=== TPP-LLM Training Configuration ===")
    print(f"Args: {args}")
    print(f"bata_semantic: {args.beta_semantic}")
    print(f"LR Scheduler: {args.lr_scheduler_type}")
    print(f"Warmup Ratio: {args.warmup_ratio}")
    print(f"======================================")

    # Reproducibility & Directory Setup
    seed_everything(args.seed)
    os.makedirs(args.result_save_path, exist_ok=True)

    # Initialize log file
    val_log_path = os.path.join(args.result_save_path, 'val.txt')
    with open(val_log_path, 'a') as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Beta Semantic: {args.beta_semantic}\n")
    
    # Prompt Preparation
    base_dataset_name = os.path.basename(args.dataset_path).replace('_few_shot', '')
    prompt = get_prompt(dataset_name=base_dataset_name, event_time_first=args.temporal_emb_first)
    if args.no_prompt:
        prompt = ''
    

    with open(val_log_path, 'a') as f:
        f.write(f"Prompt: {prompt}\n")

    
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
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        peft_config = None

    
    # Data Preparation & Statistics
    print("Loading dataset statistics...")
    stats = get_all_data_stats(args.dataset_path)

    print("Initializing DataLoaders...")
    dataset_train = TPPLLMDataset(f'{args.dataset_path}/train.json')
    dataset_val = TPPLLMDataset(f'{args.dataset_path}/dev.json')
    dataset_test = TPPLLMDataset(f'{args.dataset_path}/test.json')

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, 
        generator=g, worker_init_fn=seed_worker, collate_fn=collate_fn
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    
    # 5. Model Initialization
    print("Initializing TPP-LLM Model...")
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
        beta_semantic=args.beta_semantic,   
    )

   # Inject dataset bounds (min/max) as model attributes for visualization & normalization
    model.max_mag = stats['mag']['max']
    model.max_dep = stats['dep']['max']
    model.max_time = stats['time']['max']
    model.min_mag = stats['mag']['min']
    model.min_dep = stats['dep']['min']
    model.min_time = stats['time']['min']
    model.min_delta = stats['delta']['min']
    model.max_delta = stats['delta']['max']
    
    # Training/Evaluation Execution
    print("Starting Runner...")
    runner = TPPLLMRunner(
        model=model,
        beta_type=args.beta_type,
        beta_time=args.beta_time,
        beta_semantic=args.beta_semantic,
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
        save_flag=args.save_flag,
        train_flag=args.train_flag,
        gradient_accumulation_steps=4
    )
    
# Finalize Execution
end_time = time.time()
execution_time = end_time - start_time
with open(val_log_path, 'a') as f:
        f.write(f"Total Execution Time: {execution_time:.2f} seconds\n")

print(f"Execution successfully completed in {execution_time:.2f} seconds.")