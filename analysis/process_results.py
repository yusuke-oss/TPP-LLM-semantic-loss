import os
import re
import ast
import numpy as np
import pandas as pd
import collections

def find_best_epoch(val_file_path):
    """
    Parses the val.txt file to find the best epoch.
    If 'new best validation metrics' is found (Training mode), returns the epoch number.
    If only 'validation metrics: {' is found (Loaded mode), returns 'loaded'.
    """
    epoch_pattern = re.compile(r"validation metrics of epoch (\d+):")
    
    try:
        with open(val_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  [Warning] File not found: {val_file_path}")
        return None
    except Exception as e:
        print(f"  [Error] Failed to read {val_file_path} - {e}")
        return None

    new_best_indices = [m.start() for m in re.finditer("new best validation metrics", content)]
    
    # If no "new best" is found, check if it's a loaded model (only initial validation)
    if not new_best_indices:
        if "validation metrics: {" in content:
            return "loaded"
        
        print(f"  [Warning] Neither 'new best validation metrics' nor loaded metrics found in {val_file_path}.")
        return None
        
    last_best_index = new_best_indices[-1]
    relevant_content = content[:last_best_index]
    epoch_matches = list(re.finditer(epoch_pattern, relevant_content))
    
    if not epoch_matches:
        print(f"  [Warning] No epoch line found before the 'new best' in {val_file_path}")
        return None
        
    best_epoch = int(epoch_matches[-1].group(1))
    return best_epoch

def get_test_metrics(test_file_path, epoch):
    """
    Parses the test.txt file to extract metrics.
    Handles both trained models (epoch target) and pre-loaded models.
    """
    if epoch == "loaded":
        # For loaded models, the log is simply "test metrics: {...}"
        metric_pattern = re.compile(r"test metrics: (\{.*\})")
    else:
        # For trained models, the log includes the epoch number
        metric_pattern = re.compile(rf"test metrics of epoch {epoch}: (\{{.*\}})")
    
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = metric_pattern.search(line)
                if match:
                    metric_dict_str = match.group(1)
                    metrics = ast.literal_eval(metric_dict_str)
                    return {
                        'log_likelihood': metrics.get('log_likelihood'),
                        'accuracy': metrics.get('accuracy'),
                        'rmse': metrics.get('rmse')
                    }
    except FileNotFoundError:
        print(f"  [Warning] File not found: {test_file_path}")
        return None
    except Exception as e:
        print(f"  [Error] Failed while processing {test_file_path} - {e}")
        return None
        
    print(f"  [Warning] Metrics for epoch '{epoch}' not found in {test_file_path}.")
    return None

def process_experiment_group(runs_list):
    """
    Processes a list of runs (seeds) and aggregates the results.
    """
    results_list = []
    
    # Sort by seed number
    runs_list_sorted = sorted(runs_list, key=lambda x: x['seed'])
    
    for run in runs_list_sorted:
        base_path = run['path']
        seed = run['seed']
        val_file = os.path.join(base_path, "val.txt")
        test_file = os.path.join(base_path, "test.txt")
        
        print(f"--- Processing Seed {seed} ({base_path}) ---")
        
        best_epoch = find_best_epoch(val_file)
        if best_epoch is None:
            print(f"  Failed to identify the best epoch for seed {seed}.")
            continue
            
        if best_epoch == "loaded":
            print(f"  Mode: Loaded Model (Pre-trained)")
        else:
            print(f"  Best Epoch: {best_epoch}")
        
        test_metrics = get_test_metrics(test_file, best_epoch)
        if test_metrics is None:
            print(f"  Failed to find test metrics for epoch {best_epoch}.")
            continue
        
        print(f"  Test Metrics: {test_metrics}")
        # Include 'seed' in the results to easily identify the source
        test_metrics['seed'] = seed
        results_list.append(test_metrics)
        
    return results_list

def summarize_results(results, output_file=None, exp_name=""):
    """
    Calculates and displays/saves the mean and standard deviation 
    from the collected results list.
    """
    if not results:
        print("No results to aggregate.")
        return

    df = pd.DataFrame(results)
    
    # Move the 'seed' column to the front of the DataFrame
    if 'seed' in df.columns:
        cols = ['seed'] + [col for col in df.columns if col != 'seed']
        df = df[cols]

    # --- Console Output ---
    exp_title = f"--- Aggregated Results for Experiment '{exp_name}' (All Seeds) ---"
    print(f"\n{exp_title}")
    print("Collected Data (Ordered by Seed):")
    print(df.to_string(index=False)) # Display without row indices
    
    # Calculate statistics excluding the 'seed' column
    metrics_df = df.drop(columns=['seed'], errors='ignore')
    means = metrics_df.mean()
    stds = metrics_df.std(ddof=1) 
    
    summary_df = pd.DataFrame({'Mean': means, 'StdDev': stds})
    print("\nStatistical Summary:")
    print(summary_df)
    
    print("\nMetrics (Mean ± StdDev):")
    summary_lines = []
    for metric in metrics_df.columns:
        if metric in means and metric in stds:
            line = f"  {metric}: {means[metric]:.6f} ± {stds[metric]:.6f}"
            print(line)
            summary_lines.append(line)
    
    # --- Save to File ---
    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory: {output_dir}")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"{exp_title}\n")
                f.write("Collected Data (Ordered by Seed):\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                f.write("Statistical Summary:\n")
                f.write(summary_df.to_string())
                f.write("\n\n")
                
                f.write("Metrics (Mean ± StdDev):\n")
                f.write("\n".join(summary_lines))
                
            print(f"\nResults successfully saved to {output_file}")
        except Exception as e:
            print(f"\n[Error] Failed to write to file '{output_file}' - {e}")


def main():
    # =========================================================
    # ★ Experiment Mode Switch
    # Change to "MLP" or "positional" to process specific results
    # =========================================================
    TARGET_EMB_TYPE = "MLP" 
    
    # Base directory for all experiments
    root_dir = "result/us_earthquake_semantic_loss"
    
    # The specific directory to scan based on the selected mode
    base_search_dir = os.path.join(root_dir, TARGET_EMB_TYPE.lower())
    
    # Output directory for the summary text files
    output_summary_dir = base_search_dir
    
    # For regex pattern matching of seed folders
    seed_pattern = re.compile(r"^seed_(\d+)$")
    experiment_groups = collections.defaultdict(list)
    
    print(f"Recursively scanning '{base_search_dir}'...")
    found_any_seeds = False

    try:
        # Recursively traverse directories
        for current_root, dirs, files in os.walk(base_search_dir, topdown=True, followlinks=False):

            # Find folders that contain both 'val.txt' and 'test.txt' (completed seed folders)
            if 'val.txt' in files and 'test.txt' in files:
                
                seed_folder_name = os.path.basename(current_root)
                match = seed_pattern.match(seed_folder_name)
                
                if match:
                    seed = int(match.group(1))
                    
                    # Use the relative path from the root_dir to the seed folder's parent
                    # Example: "mlp/beta_10000.0" or "positional"
                    parent_dir_path = os.path.dirname(current_root)
                    rel_path = os.path.relpath(parent_dir_path, root_dir)
                    
                    # Replace directory separators with underscores for a safe filename
                    exp_name = rel_path.replace(os.sep, '_')
                    
                    experiment_groups[exp_name].append({
                        'seed': seed,
                        'path': current_root
                    })
                    found_any_seeds = True

    except FileNotFoundError:
        print(f"[Error] Target directory '{base_search_dir}' not found. Have you run the training script yet?")
        return
    except Exception as e:
        print(f"[Error] A problem occurred while scanning directories - {e}")
        return

    if not found_any_seeds:
        print(f"Could not find any completed experiments in '{base_search_dir}'.")
        return
        
    print(f"\nDetected {len(experiment_groups)} experiment group(s) in '{base_search_dir}'.")
    print("-" * 30)

    # 5. Aggregate and save results for each experiment group (e.g., mlp_beta_10000.0, positional)
    for exp_name, runs_list in experiment_groups.items():
        print(f"\n=======================================================")
        print(f"  Starting processing for Experiment Group '{exp_name}' ({len(runs_list)} seeds)")
        print(f"=======================================================")
        
        results = process_experiment_group(runs_list)
        
        if results:
            # Output as "mlp_beta_10000.0_summary.txt" or "positional_summary.txt"
            output_filename = os.path.join(output_summary_dir, f"{exp_name}_summary.txt")
            summarize_results(results, output_filename, exp_name)
        else:
            print(f"No results available to aggregate for Experiment Group '{exp_name}'.")

# --- Execute Script ---
if __name__ == "__main__":
    main()