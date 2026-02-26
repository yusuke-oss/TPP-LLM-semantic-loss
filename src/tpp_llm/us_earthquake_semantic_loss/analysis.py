"""
TPP-LLM Analysis and Visualization Tools
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, balanced_accuracy_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr, shapiro, norm

# ====================================================================
# 1. Qualitative & Quantitative Sequence Evaluation
# ====================================================================

def visualize_qualitative_sequence(raw_data, save_dir, epoch, num_examples=5):
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

    vis_dir = os.path.join(save_dir, "qualitative_plots")
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

        # Upper plot: Event Types
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

        props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='gray')
        ax1.text(1.02, 0.0, acc_str, transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        # Lower plot: Absolute Times
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(steps, true_times, color='black', marker='s', markersize=6,
                 linestyle='-', label='True Time', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', markersize=6,
                 linestyle='--', label='Predicted Time', alpha=0.8)
        ax2.set_ylim(bottom=0) 

        ax2.text(1.02, 0.0, rmse_str, transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        save_path = os.path.join(vis_dir, f"seq_{i}_type_time_epoch{epoch}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Saved qualitative plots to {vis_dir}")

def perform_quantitative_analysis(
    all_types_true, all_types_pred,
    all_times_true, all_time_preds,
    all_time_deltas_true,
    result_save_path, phase_name,
    seq_scores=None
):
    """
    Evaluate quantitative metrics (Confusion Matrix, RMSE, Classification Report).
    Results are saved neatly in the 'quantitative_metrics' directory.
    """
    print(f"--- Running Quantitative Analysis ({phase_name}) ---")
    
    quant_dir = os.path.join(result_save_path, "quantitative_metrics")
    os.makedirs(quant_dir, exist_ok=True)

    # 1. Confusion Matrix
    try:
        class_names = ['Large', 'Medium', 'Small'] 
        cm = confusion_matrix(all_types_true, all_types_pred, labels=[0, 1, 2], normalize='true')
        sns.reset_orig()  # Seabornの設定を初期化
        # ---------------------------------------------------------
        # タイトルあり版 (Titled)
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6)) # ★ サイズを 6x5 -> 8x6 に拡大
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)
        
        # ★ pad=20 でタイトルとグラフの間に少し余白を作る
        ax.set_title(f'Normalized Confusion Matrix ({phase_name})', pad=20) 
        
        # ★ bbox_inches='tight' が見切れを完全に防ぐ最強のオプションです
        plt.savefig(os.path.join(quant_dir, f'{phase_name}_confusion_matrix_titled.png'), 
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

        # ---------------------------------------------------------
        # タイトルなし版 (Untitled)
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)
        ax.set_title("") 
        
        plt.savefig(os.path.join(quant_dir, f'{phase_name}_confusion_matrix_untitled.png'), 
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
        
        with open(os.path.join(quant_dir, f'{phase_name}_analysis_report.txt'), 'w') as f:
            f.write(combined_log)
        with open(os.path.join(result_save_path, 'val.txt'), 'a') as f:
            f.write(f"\n[{phase_name} Phase]\n" + combined_log)
            
    except Exception as e:
        print(f"Failed to save analysis logs: {e}")

    # 5. Call Sequence Qualitative Analysis
    if seq_scores and 'raw_data' in seq_scores:
        visualize_qualitative_sequence(seq_scores['raw_data'], quant_dir, phase_name, num_examples=30)


# ====================================================================
# 2. Semantic Space Evaluation (PCA & Distributions)
# ====================================================================

def run_full_evaluation(model, tokenizer, save_dir, epoch, device='cpu'):
    """
    Comprehensive evaluation of learned embeddings including Distribution Analysis, 
    Distance Preservation, and 2D Semantic Space Visualization.
    """
    print(f"\n=======================================================")
    print(f"   STARTING SEMANTIC EVALUATION (Epoch {epoch})")
    print(f"=======================================================\n")

    vis_dir = os.path.join(save_dir, "semantic_visualizations", f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    log_path = os.path.join(vis_dir, "analysis_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        def log(text):
            print(text)
            f.write(text + "\n")

        log(f"Semantic Analysis Results for Epoch {epoch}")
        log("-" * 50)

        targets = [
            {'key': 'mag', 'mlp': getattr(model, 'mag_mlp', None), 'label': "Magnitude", 
             'min': getattr(model, 'min_mag', -2.0), 'max': getattr(model, 'max_mag', 9.0), 'anchor': "Magnitude"},
            {'key': 'dep', 'mlp': getattr(model, 'dep_mlp', None), 'label': "Depth",     
             'min': getattr(model, 'min_dep', 0.0), 'max': getattr(model, 'max_dep', 700.0), 'anchor': "Depth"},
            {'key': 'time', 'mlp': getattr(model, 'time_mlp', None), 'label': "Time",      
             'min': getattr(model, 'min_time', 0.0), 'max': getattr(model, 'max_time', 100.0), 'anchor': "Time"}
        ]
        
        valid_targets = [t for t in targets if t['mlp'] is not None]
        if not valid_targets:
            log("No MLP encoders found (Skipping Semantic Space Analysis).")
            return

        model.eval()

        # --- 1. Distribution Analysis ---
        log("\n[1] Distribution Analysis (Variance & Centroid)")
        fig_dist, axes_dist = plt.subplots(len(valid_targets), 2, figsize=(16, 5 * len(valid_targets)))
        if len(valid_targets) == 1: axes_dist = axes_dist.reshape(1, -1)

        for i, t in enumerate(valid_targets):
            inputs_w = tokenizer(t['anchor'], return_tensors='pt', add_special_tokens=False).to(device)
            with torch.no_grad():
                target_vec = model.llm.get_input_embeddings()(inputs_w['input_ids']).mean(dim=1) 
                mlp_vecs = t['mlp'](torch.tensor(np.linspace(t['min'], t['max'], 500), dtype=torch.float32).unsqueeze(1).to(device))
            
            mean_vec = torch.mean(mlp_vecs, dim=0)
            total_variance = torch.sum(torch.var(mlp_vecs, dim=0)).item()

            dists_target = torch.norm(mlp_vecs - target_vec, dim=1).cpu().numpy()
            sims_target  = F.cosine_similarity(mlp_vecs, target_vec).cpu().numpy()

            log(f" >> {t['label']} Distribution (Anchor: {t['anchor']})")
            log(f"    - Centroid-Target Dist: {torch.norm(mean_vec - target_vec).item():.4f}")
            log(f"    - Total Variance:       {total_variance:.4f}")

            sns.histplot(dists_target, kde=True, ax=axes_dist[i, 0], color='skyblue', bins=30)
            axes_dist[i, 0].set_title(f"{t['label']}: Distance to Target\n(Var: {total_variance:.2f})")
            sns.histplot(sims_target, kde=True, ax=axes_dist[i, 1], color='orange', bins=30)
            axes_dist[i, 1].set_title(f"{t['label']}: Cosine Similarity to Target")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "Distribution_Analysis.png"))
        plt.close()

        # --- 2. Distance Preservation ---
        log("\n[2] Distance Preservation Analysis (Shepard Diagram)")
        fig_shep, axes_shep = plt.subplots(1, len(valid_targets), figsize=(8 * len(valid_targets), 7))
        if len(valid_targets) == 1: axes_shep = [axes_shep]

        for i, t in enumerate(valid_targets):
            raw_vals = np.linspace(t['min'], t['max'], 100)
            with torch.no_grad():
                vecs = t['mlp'](torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()
            
            dist_input = pdist(raw_vals.reshape(-1, 1), metric='euclidean')
            dist_output = pdist(vecs, metric='euclidean')
            corr_spearman, _ = spearmanr(dist_input, dist_output)

            axes_shep[i].scatter(dist_input, dist_output, s=2, alpha=0.3, c=['blue', 'green', 'orange'][i])
            axes_shep[i].set_title(f"{t['label']}\nSpearman: {corr_spearman:.4f}")
            axes_shep[i].set_xlabel("Input Distance")
            axes_shep[i].set_ylabel("Embedding Distance")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "Distance_Preservation.png"))
        plt.close()

        # --- 3. 2D PCA Visualization ---
        log("\n[3] 2D Semantic Space Visualization (PCA)")
        fig_comb, axes_comb = plt.subplots(1, len(valid_targets), figsize=(8 * len(valid_targets), 8))
        if len(valid_targets) == 1: axes_comb = [axes_comb]

        for i, t in enumerate(valid_targets):
            raw_vals = np.linspace(t['min'], t['max'], 50)
            with torch.no_grad():
                mlp_vecs = t['mlp'](torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()

            anchor_vec = getattr(model, f"anchor_{t['key']}", None)
            anchor_vecs = np.array([anchor_vec.detach().cpu().numpy()]) if anchor_vec is not None else None

            vectors_list = [mlp_vecs]
            if anchor_vecs is not None: vectors_list.append(anchor_vecs)
            
            pca = PCA(n_components=2 if len(np.vstack(vectors_list)) >= 2 else 1)
            reduced = pca.fit_transform(np.vstack(vectors_list))
            if pca.n_components_ == 1: reduced = np.hstack([reduced, np.zeros_like(reduced)])

            r_mlp = reduced[:len(mlp_vecs)]
            r_anc = reduced[len(mlp_vecs):] if anchor_vecs is not None else None

            def plot_simple_2d(ax, title):
                sc = ax.scatter(r_mlp[:,0], r_mlp[:,1], c=raw_vals, cmap='plasma', s=80, alpha=0.9, edgecolors='white')
                ax.plot(r_mlp[:,0], r_mlp[:,1], c='gray', alpha=0.5, linewidth=2)
                ax.annotate(f"Min\n{raw_vals[0]:.1f}", (r_mlp[0,0], r_mlp[0,1]), fontsize=10, fontweight='bold')
                ax.annotate(f"Max\n{raw_vals[-1]:.1f}", (r_mlp[-1,0], r_mlp[-1,1]), fontsize=10, fontweight='bold')
                
                if r_anc is not None:
                    ax.scatter(r_anc[:,0], r_anc[:,1], c='gold', marker='*', s=400, edgecolors='black')
                    ax.annotate(f"Anc({t['label']})", (r_anc[0,0], r_anc[0,1]), xytext=(0, 10), textcoords='offset points', 
                                ha='center', fontsize=11, fontweight='bold', color='darkgoldenrod')
                
                ax.set_title(title)
                ax.grid(True, linestyle='--', alpha=0.5)

            plot_simple_2d(axes_comb[i], f"{t['label']} Space")

            fig_single, ax_single = plt.subplots(figsize=(8, 8))
            plot_simple_2d(ax_single, f"{t['label']} Space (Numeric + Anchor)")
            fig_single.savefig(os.path.join(vis_dir, f"2D_{t['label']}_Titled.png"), bbox_inches='tight')
            ax_single.set_title("") 
            fig_single.savefig(os.path.join(vis_dir, f"2D_{t['label']}_Untitled.png"), bbox_inches='tight')
            plt.close(fig_single)

        plt.tight_layout()
        fig_comb.savefig(os.path.join(vis_dir, "2D_Semantic_Space_Combined.png"))
        plt.close()
        log(f"Completed Visualizations. Check directory: {vis_dir}")