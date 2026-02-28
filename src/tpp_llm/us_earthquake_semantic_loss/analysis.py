"""
TPP-LLM Analysis and Visualization Tools
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ====================================================================
# 1. Qualitative & Quantitative Sequence Evaluation
# ====================================================================

def visualize_qualitative_sequence(raw_data, save_dir, phase_name, epoch, num_examples=5):
    """
    Generate trajectory plots for qualitative evaluation.
    Top: True vs Predicted Event Type (Discrete)
    Bottom: Absolute Cumulative Time (Lower bound = 0)
    """
    print(f"Generating qualitative sequence plots (Type & Abs Time) for epoch {epoch}...")
    
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    selected_seqs = raw_data[:num_examples]
    
    type_to_y = {0: 2, 1: 1, 2: 0} 
    y_labels = {0: "Small", 1: "Medium", 2: "Large"}

    # Organize neatly into the "plots_sequence" directory
    vis_dir = os.path.join(save_dir, f"analysis/plots_sequence/{phase_name}")
    os.makedirs(vis_dir, exist_ok=True)

    for i, seq in enumerate(selected_seqs):
        true_types = np.array(seq['true_types']).flatten()
        pred_types = np.array(seq['pred_types']).flatten()
        true_times = np.array(seq['true_times']).flatten()
        pred_times = np.array(seq['pred_times']).flatten()

        min_len = min(len(true_types), len(pred_types), len(true_times), len(pred_times))
        true_types = true_types[:min_len]
        pred_types = pred_types[:min_len]
        true_times = true_times[:min_len]
        pred_times = pred_times[:min_len]

        try:
            true_types = true_types.astype(int)
            pred_types = pred_types.astype(int)
        except ValueError:
            pass 

        true_y = [type_to_y.get(t, t) for t in true_types]
        pred_y = [type_to_y.get(t, t) for t in pred_types]
        steps = np.arange(1, min_len + 1)

        acc_str = "N/A"
        if len(true_types) > 0:
            type_acc = np.mean(true_types == pred_types)
            acc_str = f"Type Acc:\n{type_acc:.1%}"

        rmse_str = "N/A"
        if len(true_times) > 0:
            time_rmse = np.sqrt(np.mean((true_times - pred_times) ** 2))
            rmse_str = f"Time RMSE:\n{time_rmse:.4f}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        plt.subplots_adjust(right=0.85, hspace=0.15)

        # ==========================================
        # Upper plot: Event Types
        # ==========================================
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax1.grid(True, axis='x', linestyle=':', alpha=0.5)

        ax1.plot(steps, true_y, color='#1f77b4', marker='s', markersize=10, 
                 linestyle='-', linewidth=2, label='True Type', alpha=0.6)
        ax1.plot(steps, pred_y, color='#ff7f0e', marker='o', markersize=8, 
                 linestyle='--', linewidth=2, label='Predicted Type', alpha=0.9)

        yticks_vals = sorted(y_labels.keys())
        ax1.set_yticks(yticks_vals)
        ax1.set_yticklabels([y_labels[v] for v in yticks_vals])
        ax1.set_ylim(-0.5, 2.5)
        
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=12)

        props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='gray')
        ax1.text(1.02, 0.0, acc_str, transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        # ==========================================
        # Lower plot: Absolute Times
        # ==========================================
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(steps, true_times, color='black', marker='s', markersize=6,
                 linestyle='-', label='True Time', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', markersize=6,
                 linestyle='--', label='Predicted Time', alpha=0.8)
        ax2.set_ylim(bottom=0) 
        
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=12)

        ax2.text(1.02, 0.0, rmse_str, transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        save_path = os.path.join(vis_dir, f"seq_{i}_type_time.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Saved qualitative plots to {vis_dir}")

def perform_quantitative_analysis(
    all_types_true, all_types_pred,
    all_times_true, all_time_preds,
    all_time_deltas_true,
    result_save_path, phase_name,
    epoch, # Passed from the runner
    seq_scores=None
):
    """
    Evaluate quantitative metrics (Confusion Matrix, RMSE, Classification Report).
    Results are saved neatly in the detailed analysis directory by epoch.
    """
    print(f"--- Running Quantitative Analysis ({phase_name} - Epoch {epoch}) ---")
    
    # Create an epoch-specific directory within the detailed metrics folder
    quant_dir = os.path.join(result_save_path, "analysis", f"detail/epoch_{epoch}/{phase_name}")
    os.makedirs(quant_dir, exist_ok=True)

    # 1. Confusion Matrix
    try:
        class_names = ['Large', 'Medium', 'Small'] 
        cm = confusion_matrix(all_types_true, all_types_pred, labels=[0, 1, 2], normalize='true')
        sns.reset_orig()  # Reset seaborn settings to prevent layout issues
        import matplotlib as mpl
        mpl.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        
        # --- Clean Version (Untitled for Academic Papers) ---
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0.0, vmax=1.0, square=True, cbar_kws={"shrink": .8})
        ax.set_ylabel('True label', fontweight='bold')
        ax.set_xlabel('Predicted label', fontweight='bold')
        
        # Save with a simple filename
        plt.savefig(os.path.join(quant_dir, f'confusion_matrix.png'), 
                    bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    except Exception as e:
        print(f"Failed to generate Confusion Matrix: {e}")

    # 2. Time Error Analysis
    try:
        bins_hours = [0, 1, 6, 12, np.inf]
        bin_labels = ['(0-1 hour)', '(1-6 hours)', '(6-12 hours)', '(12+ hours)']
        
        all_time_deltas_true_hours = all_time_deltas_true * 24.0
        bin_indices = np.digitize(all_time_deltas_true_hours, bins=bins_hours)
        
        analysis_log_time = f"--- {phase_name} Absolute Time RMSE by Intervals ---\n"
        
        for i in range(1, len(bins_hours)):
            label = bin_labels[i-1]
            mask = (bin_indices == i)
            
            if np.sum(mask) == 0:
                analysis_log_time += f"{label}: N=0\n"
                continue
                
            true_abs_times_in_bin = all_times_true[mask]
            pred_abs_times_in_bin = all_time_preds[mask]
            rmse_in_bin = np.sqrt(np.mean((true_abs_times_in_bin - pred_abs_times_in_bin) ** 2))
            
            analysis_log_time += f"{label}: N={np.sum(mask)}, RMSE={rmse_in_bin:.4f}\n"
    except Exception as e:
        analysis_log_time = f"Time Error Analysis Failed: {e}\n"

    # 3. Classification Report
    try:
        report = classification_report(all_types_true, all_types_pred, target_names=class_names, digits=4)
        bal_acc = balanced_accuracy_score(all_types_true, all_types_pred)
        analysis_log_report = f"--- {phase_name} Classification Report ---\n{report}\nBalanced Accuracy: {bal_acc:.4f}\n"
    except Exception as e:
        analysis_log_report = f"Classification Report Failed: {e}\n"
    
    # 4. Save Logs
    try:
        combined_log = analysis_log_time + "\n" + analysis_log_report
        print(combined_log)
        
        with open(os.path.join(quant_dir, f'analysis_report.txt'), 'w') as f:
            f.write(combined_log)
            
    except Exception as e:
        print(f"Failed to save analysis logs: {e}")

    # 5. Call Sequence Qualitative Analysis
    if seq_scores and 'raw_data' in seq_scores:
        visualize_qualitative_sequence(seq_scores['raw_data'], result_save_path, phase_name, epoch, num_examples=30)