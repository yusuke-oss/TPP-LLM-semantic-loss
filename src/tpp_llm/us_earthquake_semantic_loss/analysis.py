import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, balanced_accuracy_score
from scipy import stats
from scipy.stats import (
    lognorm, gamma, beta, t, norm, expon, powerlaw, uniform, 
    weibull_min, genextreme, pareto, f, chi2, cauchy 
)

def get_best_fit_distribution_details(data_values, attribute_name):
    """
    AIC最小の分布名、パラメータ辞書、全結果、および固定された分布オブジェクトを返す
    """
    if not data_values.size: 
        return "N/A", {}, [], None

    # ▼▼▼ 変更: Rawデータをそのまま使用 (MinMaxなし+MLPの方針に合わせる) ▼▼▼
    # 0以下が含まれると分布によってはエラーになるため、微小値を足す
    # 地震データ(M, depth)は通常正の値ですが、念のため
    data_for_fit = data_values 
    
    DISTRIBUTIONS = {
        'lognorm': lognorm, 'gamma': gamma, 'beta': beta, 't': t, 'norm': norm, 
        'expon': expon, 'powerlaw': powerlaw, 'uniform': uniform, 'weibull_min': weibull_min, 
        'genextreme': genextreme, 'pareto': pareto, 'f': f, 'chi2': chi2, 'cauchy': cauchy
    }
    
    best_aic, best_dist_name, best_params_tuple = np.inf, "unknown", []
    all_results = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(data_for_fit)
            log_likelihood = np.sum(dist.logpdf(data_for_fit, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood

            all_results.append({
                'Distribution': name, 'AIC': aic, 'Log_Likelihood': log_likelihood, 'Parameters_Tuple': params
            })
            
            if aic < best_aic:
                best_aic = aic
                best_dist_name = name
                best_params_tuple = params
        except Exception:
            all_results.append({
                'Distribution': name, 'AIC': np.nan, 'Log_Likelihood': np.nan, 'Parameters_Tuple': None
            })
            continue
            
    dist_name = best_dist_name
    
    # パラメータキーの割り当て (表示用)
    if dist_name == 'norm': keys = ['loc', 'scale']
    elif dist_name == 'gamma': keys = ['a', 'loc', 'scale']
    elif dist_name == 'lognorm': keys = ['s', 'loc', 'scale']
    elif dist_name in ['t', 'chi2']: keys = ['df', 'loc', 'scale']
    elif dist_name == 'cauchy': keys = ['loc', 'scale']
    elif dist_name == 'expon': keys = ['loc', 'scale']
    elif dist_name == 'uniform': keys = ['loc', 'scale']
    elif dist_name == 'beta': keys = ['a', 'b', 'loc', 'scale']
    elif dist_name == 'f': keys = ['dfn', 'dfd', 'loc', 'scale']
    elif dist_name in ['weibull_min', 'genextreme', 'powerlaw', 'pareto']:
        if len(best_params_tuple) == 3: keys = ['shape', 'loc', 'scale']
        else: keys = [f'p{i+1}' for i in range(len(best_params_tuple))]
    else:
        keys = [f'p{i+1}' for i in range(len(best_params_tuple))]

    params_dict = dict(zip(keys, best_params_tuple))

    # ▼▼▼ 追加: Frozenな分布オブジェクトを作成 ▼▼▼
    # これを使って後でCDFを計算します
    dist_class = DISTRIBUTIONS[best_dist_name]
    frozen_dist = dist_class(*best_params_tuple)

    return best_dist_name, params_dict, all_results, frozen_dist

def plot_distribution_fit(data_values, dist_name, params_dict, attribute_name, save_path):
    """データヒストグラムとベストフィットPDFを比較する画像を生成"""
    if not data_values.size or not params_dict:
        return

    # Rawデータを使用
    data_for_plot = data_values
    

    # (分布クラスマップは get_best... と同じにする必要があるため、ここでは簡易的に scipy.stats から動的取得)
    dist_class = getattr(stats, dist_name, None)
    if not dist_class: return

    plt.figure(figsize=(8, 5))
    x_min, x_max = np.min(data_for_plot), np.max(data_for_plot)
    x = np.linspace(x_min, x_max, 200)
    
    plt.hist(data_for_plot, bins=40, density=True, alpha=0.6, color='skyblue', label='Observed Data (Raw)')
    
    params_args = list(params_dict.values())
    pdf_best = dist_class.pdf(x, *params_args)
    plt.plot(x, pdf_best, 'r-', linewidth=2, label=f'Best Fit: {dist_name}')

    plt.title(f'{attribute_name} Distribution Fit (AIC Best Fit)')
    plt.xlabel('Value (Raw Scale)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()

# ... (format_all_results... と perform_quantitative... は元のままでOK) ...
def format_all_results_for_output(results, attribute_name):
    # (省略: 元のコードと同じ)
    sorted_results = sorted(results, key=lambda x: x['AIC'] if not np.isnan(x['AIC']) else np.inf)
    output_lines = []
    output_lines.append(f"\n==================================================")
    output_lines.append(f"=== {attribute_name} - AIC Ranking ===")
    output_lines.append(f"==================================================")
    output_lines.append(f"{'Rank':<5}| {'Distribution':<15}| {'AIC':<10}| {'Log-L':<18}| {'Params'}")
    output_lines.append("-" * 70)
    
    for i, res in enumerate(sorted_results): 
        aic_str = f"{res['AIC']:.2f}" if not np.isnan(res['AIC']) else "FAILED"
        logl_str = f"{res['Log_Likelihood']:.2f}" if not np.isnan(res['Log_Likelihood']) else "N/A"
        params_str = ", ".join([f"{p:.3f}" for p in res['Parameters_Tuple']]) if res['Parameters_Tuple'] is not None else "N/A"
        output_lines.append(f"{i+1:<5}| {res['Distribution']:<15}| {aic_str:<10}| {logl_str:<18}| {params_str}")
    return "\n".join(output_lines)



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================================
# ★ 定性評価 可視化関数 (True Type vs Pred Type & Time)
# ====================================================================
def visualize_qualitative_sequence(raw_data, save_dir, epoch, num_examples=5):
    """
    定性評価のための時系列プロット（Trajectory Plot）を生成する。
    上段: 真のイベントタイプ（離散） vs 予測イベントタイプ（離散）
          縦軸は [Small, Medium, Large] のラベルを表示
    下段: 絶対時刻（Cumulative Time）の予測 (Y軸下限=0固定)
    """
    print(f"Generating qualitative sequence plots (Type & Abs Time) for epoch {epoch}...")
    
    # スタイル設定：論文・スライド向けに見やすく
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    
    selected_seqs = raw_data[:num_examples]
    
    # 可視化用のY座標定義
    # グラフ上で重ならないように少し高さを変える、または単に等間隔に配置する
    # ここでは単純に 0, 1, 2 としてマッピングし、ラベルを貼り替える
    # 0: Small, 1: Medium, 2: Large と仮定（モデルの定義に合わせて変更してください）
    # もしモデル出力が 0=Large, 1=Medium, 2=Small なら辞書を逆にしてください
    
    # 【重要】あなたのモデルのクラスID定義に合わせてください
    # 例: 0=Small, 1=Medium, 2=Large の場合
    type_to_y = {0: 2, 1: 1, 2: 0} 
    y_labels = {0: "Small", 1: "Medium", 2: "Large"}
    
    # もしクラスIDが逆（0=Large...）ならここを修正
    # type_to_y = {0: 2, 1: 1, 2: 0} 

    vis_dir = os.path.join(save_dir, "qualitative_plots")
    os.makedirs(vis_dir, exist_ok=True)

    for i, seq in enumerate(selected_seqs):
        # ---------------------------------------------------------
        # 1. データの読み込みと整形
        # ---------------------------------------------------------
        true_types = np.array(seq['true_types']).flatten()
        pred_types = np.array(seq['pred_types']).flatten()
        true_times = np.array(seq['true_times']).flatten()
        pred_times = np.array(seq['pred_times']).flatten()

        # 長さ合わせ
        min_len = min(len(true_types), len(pred_types), len(true_times), len(pred_times))
        true_types = true_types[:min_len]
        pred_types = pred_types[:min_len]
        true_times = true_times[:min_len]
        pred_times = pred_times[:min_len]

        # 型変換
        try:
            true_types = true_types.astype(int)
            pred_types = pred_types.astype(int)
        except:
            pass 

        # ---------------------------------------------------------
        # 2. プロット用データ変換 (Y座標へマッピング)
        # ---------------------------------------------------------
        true_y = [type_to_y.get(t, t) for t in true_types]
        pred_y = [type_to_y.get(t, t) for t in pred_types]
        steps = np.arange(1, min_len + 1)

        # ---------------------------------------------------------
        # 3. スコア文字列作成
        # ---------------------------------------------------------
        acc_str = "N/A"
        if len(true_types) > 0:
            type_acc = np.mean(true_types == pred_types)
            acc_str = f"Type Acc:\n{type_acc:.1%}"

        rmse_str = "N/A"
        if len(true_times) > 0:
            mse = np.mean((true_times - pred_times) ** 2)
            time_rmse = np.sqrt(mse)
            rmse_str = f"Time RMSE:\n{time_rmse:.4f}"

        # ---------------------------------------------------------
        # 4. プロット作成 (2段構成)
        # ---------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        plt.subplots_adjust(right=0.85, hspace=0.15) # 隙間調整

        # === 上段: イベントタイプ ===
        # グリッド線をY軸のラベル位置に合わせて表示
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax1.grid(True, axis='x', linestyle=':', alpha=0.5)

        # 真のタイプ（少しずらして見やすくするなら alpha を下げるか、linewidthを変える）
        ax1.plot(steps, true_y, color='#1f77b4', marker='s', markersize=10, 
                 linestyle='-', linewidth=2, label='True Type', alpha=0.6)

        # 予測タイプ
        ax1.plot(steps, pred_y, color='#ff7f0e', marker='o', markersize=8, 
                 linestyle='--', linewidth=2, label='Predicted Type', alpha=0.9)

        # Y軸の設定（数値ではなくラベルにする）
        yticks_vals = sorted(y_labels.keys())
        ax1.set_yticks(yticks_vals)
        ax1.set_yticklabels([y_labels[v] for v in yticks_vals])
        ax1.set_ylim(-0.5, 2.5) # 上下に少し余白を持たせる
        #ax1.set_ylabel("Event Type")

        # ★追加: X軸のラベルと目盛りを設定
        #ax1.set_xlabel("Event Step")
        # 整数ステップだけを表示する場合
        #ax1.set_xticks(steps)

        # 凡例
        #ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

        # 正解率を表示
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='gray')
        ax1.text(1.02, 0.0, acc_str, transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        # === 下段: 絶対時刻 ===
        ax2.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(steps, true_times, color='black', marker='s', markersize=6,
                 linestyle='-', label='True Time', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', markersize=6,
                 linestyle='--', label='Predicted Time', alpha=0.8)

        #ax2.set_ylabel("Time")
        #ax2.set_xlabel("Event Step")
        ax2.set_ylim(bottom=0) # Y軸下限を0に固定

        # 凡例
        #ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

        # RMSEを表示
        ax2.text(1.02, 0.0, rmse_str, transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        # 保存
        save_path = os.path.join(vis_dir, f"seq_{i}_type_time_epoch{epoch}.png")
        # PDFも保存したい場合は以下を追加
        # plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Saved qualitative plots to {vis_dir}")

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_global_distribution_comparison(raw_data, save_dir, epoch):
    """
    データセット全体の「真の分布」と「予測分布」を比較する。
    
    - 左図 (Type): 真 vs 予測を「隣り合わせ」の棒グラフで比較 (Count)
    - 右図 (Time): 真 vs 予測をヒストグラムで比較 (Count)
    """
    print(f"Generating global distribution comparison (Type & Time) for epoch {epoch}...")
    # 全てのグラフでグリッド線を消す場合
    sns.set_theme(style="white", context="talk", font_scale=1.1)
    # データ収集用リスト (DataFrame作成用)
    type_data_list = []
    time_data_list = []

    # タイプをマグニチュード代表値に変換するマップ
    # Large(0)=3.6, Medium(1)=1.5, Small(2)=0.0
    type_map_val = {0: 3.6, 1: 1.5, 2: 0.0}

    for seq in raw_data:
        # --- 1. タイプ (Magnitude) データ収集 ---
        # 真値 (Ground Truth)
        true_vals = [type_map_val.get(t, 0) for t in seq['true_types']]
        for v in true_vals:
            type_data_list.append({'Value': v, 'Source': 'Ground Truth'})
            
        # 予測値 (Prediction)
        pred_vals = [type_map_val.get(p, 0) for p in seq['pred_types']]
        for v in pred_vals:
            type_data_list.append({'Value': v, 'Source': 'Prediction'})

        # --- 2. 時刻 (Time) データ収集 ---
        # 真値
        for t in seq['true_times']:
            time_data_list.append({'Time': t, 'Source': 'Ground Truth'})
        # 予測値
        for t in seq['pred_times']:
            time_data_list.append({'Time': t, 'Source': 'Prediction'})

    # DataFrame化 (Seabornのhue/dodge機能を有効にするため)
    df_type = pd.DataFrame(type_data_list)
    df_time = pd.DataFrame(time_data_list)

    # 保存ディレクトリ
    vis_dir = os.path.join(save_dir, "quantitative_plots")
    os.makedirs(vis_dir, exist_ok=True)

    # --- プロット作成 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # スタイル設定
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)

    # === (1) Type (Magnitude) Comparison [Side-by-Side Bar Plot] ===
    # multiple="dodge" で棒グラフを隣り合わせにする
    # shrink=0.8 でグループ間の隙間を作る
    sns.histplot(
        data=df_type,
        x='Value',
        hue='Source',      # 色分けの基準
        multiple="dodge",  # ★重要: これで隣り合わせになる
        discrete=True,     # 離散値モード
        stat="count",      # 縦軸は個数
        shrink=0.8,        # 棒の太さ調整
        palette={'Ground Truth': 'blue', 'Prediction': 'orange'},
        alpha=0.8,
        ax=axes[0]
    )
    
    #axes[0].set_title(f"Event Type Distribution Comparison\n(Side-by-Side)", fontweight='bold')
    axes[0].set_xlabel("")
    #axes[0].set_ylabel("Count (Number of Events)")
    
    # X軸のラベルを見やすく設定
    axes[0].set_xticks([0.0, 1.5, 3.6])
    axes[0].set_xticklabels(["Small", "Medium", "Large"])

    # === (2) Time Distribution Comparison [Count Histogram] ===
    sns.histplot(
        data=df_time,
        x='Time',
        hue='Source',
        element="step",    # 階段状のヒストグラム
        stat="count",      # ★重要: 縦軸を個数に変更 (density -> count)
        common_norm=False, # True/Predそれぞれの合計数に対して計算するのではなく、単純なカウント
        bins=40,
        palette={'Ground Truth': 'blue', 'Prediction': 'orange'},
        alpha=0.3,         # 重なりが見えるように少し透明に
        ax=axes[1]
    )
    
    #axes[1].set_title(f"Global Time Distribution\n(Epoch {epoch})", fontweight='bold')
    axes[1].set_xlabel("")
    #axes[1].set_ylabel("Count (Number of Events)")

    plt.tight_layout()
    save_path = os.path.join(vis_dir, f"epoch_global_distribution_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution comparison to {save_path}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_performance_distribution(raw_data, save_dir, epoch):
    """
    データセット全体の「シーケンスごとの性能分布」をヒストグラムにする。
    - 左図: タイプ予測の正解率 (Accuracy) の分布
    - 右図: 時刻予測のRMSE (Root Mean Squared Error) の分布
    """
    print(f"Generating performance distribution histograms for epoch {epoch}...")
# 全てのグラフでグリッド線を消す場合
    sns.set_theme(style="white", context="talk", font_scale=1.1)
    metrics_list = []

    for seq in raw_data:
        # --- 1. タイプ予測の正解率 (Accuracy) ---
        true_types = np.array(seq['true_types'])
        pred_types = np.array(seq['pred_types'])
        
        if len(true_types) > 0:
            # 一致した数 / 全イベント数
            acc = np.mean(true_types == pred_types)
        else:
            acc = 0.0

        # --- 2. 時刻予測のRMSE ---
        true_times = np.array(seq['true_times'])
        pred_times = np.array(seq['pred_times'])
        
        if len(true_times) > 0:
            # 二乗誤差の平均のルート
            mse = np.mean((true_times - pred_times) ** 2)
            rmse = np.sqrt(mse)
        else:
            rmse = 0.0
            
        metrics_list.append({'accuracy': acc, 'rmse': rmse})

    # DataFrame化
    df = pd.DataFrame(metrics_list)
    
    # 保存ディレクトリ作成
    vis_dir = os.path.join(save_dir, "quantitative_plots")
    os.makedirs(vis_dir, exist_ok=True)

    # --- プロット作成 (横2枚) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # スタイル設定
    sns.set_theme(style="whitegrid", context="talk")

    # (1) Accuracy Distribution
    sns.histplot(data=df, x='accuracy', bins=20, kde=False, ax=axes[0], color='teal', element="step")
    #axes[0].set_title(f"Type Prediction Accuracy Distribution\n(Mean: {df['accuracy'].mean():.3f})", fontweight='bold')
    axes[0].set_xlabel("")
    #axes[0].set_ylabel("Count (Number of Sequences)")
    #axes[0].set_xlim(0, 1.05) # 0~100%の範囲に固定

    # (2) RMSE Distribution
    sns.histplot(data=df, x='rmse', bins=30, kde=False, ax=axes[1], color='coral', element="step")
    #axes[1].set_title(f"Time Prediction RMSE Distribution\n(Mean: {df['rmse'].mean():.3f})", fontweight='bold')
    axes[1].set_xlabel("")
    #axes[1].set_ylabel("Count (Number of Sequences)")
    
    # RMSEは外れ値があるとグラフが見にくくなるので、上位99%に絞って表示するのもアリ
    # q99 = df['rmse'].quantile(0.99)
    # axes[1].set_xlim(0, q99)

    plt.tight_layout()
    save_path = os.path.join(vis_dir, f"epoch_performance_histograms.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance histograms to {save_path}")

    # 統計情報をコンソールにも出す
    print("\n=== Performance Statistics ===")
    print(df.describe().round(4))


import os
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# ★ 定性評価 可視化関数 (Best / Median / Worst の比較)
# ====================================================================
def visualize_qualitative_sequence_comparison(raw_data, save_dir, epoch):
    """
    データセット全体から予測精度が良い順にソートし、
    Best (Top 1%), Median (Typical), Worst (Bottom 1%) の3つのシーケンスを
    横に並べて可視化する。
    """
    print(f"Generating qualitative comparison plots (Best/Median/Worst) for epoch {epoch}...")
    sns.set_theme(style="white", context="talk", font_scale=1.1)
    # --- 1. 評価スコア（誤差）の計算とソート ---
    # ここでは簡易的に「時刻のMAE + タイプの不一致率」を誤差スコアとします
    scored_sequences = []
    
    for seq in raw_data:
        true_times = np.array(seq['true_times'])
        pred_times = np.array(seq['pred_times'])
        true_types = np.array(seq['true_types'])
        pred_types = np.array(seq['pred_types'])
        
        # 時刻誤差 (Mean Absolute Error)
        time_mae = np.mean(np.abs(true_times - pred_times))
        
        # タイプ不一致率 (0.0 ~ 1.0)
        type_mismatch = np.mean(true_types != pred_types)
        
        # 総合スコア (小さいほど良い)
        # ※必要に応じて重み付けを変更してください
        score = time_mae + (type_mismatch * 1.0)
        
        scored_sequences.append({
            'score': score,
            'data': seq
        })
    
    # スコアが良い順（誤差が小さい順）にソート
    scored_sequences.sort(key=lambda x: x['score'])
    
    # --- 2. Best, Median, Worst の選出 ---
    n = len(scored_sequences)
    if n < 3:
        print("Data size is too small for comparison.")
        return

    targets = [
        ("Best Case (Top 1%)", scored_sequences[0]['data']),           # 最良
        ("Typical Case (Median)", scored_sequences[n // 2]['data']),   # 中央値
        ("Worst Case (Bottom 1%)", scored_sequences[-1]['data'])       # 最悪
    ]

    # --- 3. プロット設定 (変更なし) ---
    type_map_val = {0: 3.6, 1: 1.5, 2: 0.0}
    vis_dir = os.path.join(save_dir, "qualitative_plots")
    os.makedirs(vis_dir, exist_ok=True)

    # 図の作成: 2行 x 3列 (左から Best, Median, Worst)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex='col')
    
    # ループで3つのケースを描画
    for col_idx, (label, seq) in enumerate(targets):
        # 該当する列のサブプロットを取得
        ax1 = axes[0, col_idx] # 上段: マグニチュード
        ax2 = axes[1, col_idx] # 下段: 時刻

        # ---------------------------------------------------------
        # 1. データの読み込みと「型・形状」の強力な補正
        # ---------------------------------------------------------
        # 配列化して、余計な次元があれば潰す (flatten)
        true_types = np.array(seq['true_types']).flatten()
        pred_types = np.array(seq['pred_types']).flatten()
        true_times = np.array(seq['true_times']).flatten()
        pred_times = np.array(seq['pred_times']).flatten()

        # 長さが違う場合は短い方に合わせる (スライス)
        min_len_type = min(len(true_types), len(pred_types))
        true_types = true_types[:min_len_type]
        pred_types = pred_types[:min_len_type]
        
        min_len_time = min(len(true_times), len(pred_times))
        true_times = true_times[:min_len_time]
        pred_times = pred_times[:min_len_time]

        # ★重要: 型を強制的に整数(int)に揃える (これが0%の原因になりやすい)
        # floatの 1.0 と intの 1 を比較できるようにする
        try:
            true_types = true_types.astype(int)
            pred_types = pred_types.astype(int)
        except:
            pass # 文字列などが混ざっている場合はスキップ

        # --- ★ここで個別のスコアを計算 ---
        # 1. Type Accuracy (正解率)
        if len(true_types) > 0:
            type_acc = np.mean(true_types == pred_types)
            acc_str = f"{type_acc:.1%}" # 例: 80.0%
        else:
            acc_str = "N/A"

        # 2. Time RMSE (時間の誤差)
        if len(true_times) > 0:
            mse = np.mean((true_times - pred_times) ** 2)
            time_rmse = np.sqrt(mse)
            rmse_str = f"{time_rmse:.4f}"
        else:
            rmse_str = "N/A"
            
        # 表示用の辞書を作成
        seq_metrics = {
            "Type Acc": acc_str,
            "Time RMSE": rmse_str
        }
        # --------------------------------

        if 'true_mags' in seq:
            true_mags = np.array(seq['true_mags'])
            has_mag = True
        else:
            true_mags = np.array([type_map_val.get(t, 0) for t in true_types])
            has_mag = False
        
        true_type_vals = np.array([type_map_val.get(t, 0) for t in true_types])
        pred_type_vals = np.array([type_map_val.get(p, 0) for p in pred_types])
        steps = np.arange(1, len(true_types) + 1)

        # === 上段: マグニチュード予測 (元のロジックそのまま) ===
        # タイトル設定
        #ax1.set_title(label, fontsize=16, fontweight='bold', pad=15)

        # 背景色（ゾーン）
        ax1.axhspan(2.0, 5.2, color='red', alpha=0.1, label='Large Zone')
        ax1.axhspan(1.0, 2.0, color='orange', alpha=0.1, label='Medium Zone')
        ax1.axhspan(-1.02, 1.0, color='green', alpha=0.1, label='Small Zone')

        # 1. 真のマグニチュード
        if has_mag:
            ax1.plot(steps, true_mags, color='black', marker='*', markersize=10, 
                     linestyle=':', linewidth=1.5, label='True Magnitude', alpha=0.7, zorder=5)
        else:
            ax1.plot(steps, true_type_vals, color='gray', marker='s', markersize=8, 
                     linestyle='--', label='True Type (Simulated)', alpha=0.5, zorder=5)

        # 2. 真のイベントタイプ
        ax1.plot(steps, true_type_vals, color='#1f77b4', marker='s', markersize=12, 
                 linestyle='--', label='True Type (Class)', alpha=0.4, zorder=4)

        # 3. 予測イベントタイプ
        ax1.plot(steps, pred_type_vals, color='orange', marker='o', markersize=8, 
                 linestyle='-', label='Predicted Type', alpha=0.9, zorder=6)

        #ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_ylim(-1.02, 5.2)

        # 凡例は「一番右のグラフ」にだけ表示（見やすくするため）
        if col_idx == 2:
            handles, labels_legend = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels_legend, handles))
            ax1.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1))


        # ---------------------------------------------------------
        # ★計算したメトリクスを右下に表示
        # ---------------------------------------------------------
        text_str = "Seq Performance:\n" + "\n".join([f"{k}: {v}" for k, v in seq_metrics.items()])
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        
        ax1.text(0.98, 0.05, text_str, transform=ax1.transAxes, fontsize=11,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props,
                 fontfamily='monospace', zorder=100)
        # ---------------------------------------------------------

        # === 下段: 絶対時刻 (Absolute Time) ===
        ax2.plot(steps, true_times, color='black', marker='s', label='True Time', linestyle='-', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', label='Predicted Time', linestyle='--', alpha=0.8)
        
        #ax2.grid(True, linestyle='--', alpha=0.5)
        
        max_time_val = max(np.max(true_times), np.max(pred_times))
        upper_limit = max(1.0, max_time_val * 1.1)
        ax2.set_ylim(0, upper_limit)
        
        # 凡例は「一番右のグラフ」にだけ表示
        #if col_idx == 2:
            #ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        
        # 軸ラベル (一番下の行と一番左の列だけにつけるのが一般的)
        #if col_idx == 0:
            #ax1.set_ylabel("Magnitude / Type Value", fontsize=12)
            #ax2.set_ylabel("Absolute Time", fontsize=12)
        
        #ax2.set_xlabel("Event Step", fontsize=12)

    # 全体のレイアウト調整
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(vis_dir, f"epoch_comparison_best_median_worst.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150) # DPIを上げて綺麗に
    plt.close()

    print(f"Saved comparison plot to {save_path}")


import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, balanced_accuracy_score

def generate_qualitative_examples(raw_data, save_dir, epoch, num_examples=10, mode='fixed'):
    """
    [定性評価] テストデータからシーケンスを抽出し、真値と予測値を比較したテキストファイルを生成する。
    ★改良点: シーケンスごとのAccuracyとRMSEを計算してヘッダーに表示
    """
    example_path = os.path.join(save_dir, f'qualitative_examples.txt')
    sns.set_theme(style="white", context="talk", font_scale=1.1)
    # クラス定義
    class_map = {0: "Large", 1: "Medium", 2: "Small"}
    
    # --- 1. サンプル抽出 ---
    n = min(len(raw_data), num_examples)
    
    if mode == 'fixed':
        sample_indices = range(n)
        selection_method = "Fixed selection (First N sequences)"
    else:
        sample_indices = random.sample(range(len(raw_data)), n)
        selection_method = "Random selection"

    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(f"====================================================================================\n")
        f.write(f"=== Qualitative Analysis (Sequence Examples) - Epoch {epoch} ===\n")
        f.write(f"Selection Method: {selection_method}\n")
        f.write(f"Total Sequences Displayed: {n}\n")
        f.write(f"====================================================================================\n\n")
        
        for sample_idx in sample_indices:
            seq = raw_data[sample_idx]
            t_typ = seq['true_types']
            p_typ = seq['pred_types']
            t_tim = seq['true_times']
            p_tim = seq['pred_times']
            
            seq_len = len(t_typ)
            
            # --- ここでシーケンスごとのスコアを計算 ---
            
            # 1. Type Accuracy (正解率)
            correct_count = sum([1 for t, p in zip(t_typ, p_typ) if t == p])
            acc = (correct_count / seq_len) * 100 if seq_len > 0 else 0
            
            # 2. Time RMSE (二乗平均平方根誤差)
            # リストをnumpy配列に変換して計算
            t_tim_arr = np.array(t_tim)
            p_tim_arr = np.array(p_tim)
            if seq_len > 0:
                rmse = np.sqrt(np.mean((t_tim_arr - p_tim_arr) ** 2))
            else:
                rmse = 0.0

            # --- ヘッダーに表示 ---
            # AccuracyとRMSEを併記します
            f.write(f"Sequence ID: {sample_idx} (Length: {seq_len}, Type Acc: {acc:.1f}%, Time RMSE: {rmse:.4f})\n")
            f.write(f"{'Step':<5} | {'True Type':<12} | {'Pred Type':<12} | {'True Time':<15} | {'Pred Time':<15} | {'Diff':<10}\n")
            f.write("-" * 95 + "\n")
            
            for i in range(seq_len):
                true_name = class_map.get(t_typ[i], f"Type {t_typ[i]}")
                pred_name = class_map.get(p_typ[i], f"Type {p_typ[i]}")
                t_val = t_tim[i]
                p_val = p_tim[i]
                time_err = p_val - t_val
                
                type_mark = "✅" if t_typ[i] == p_typ[i] else "🔺"
                err_str = f"{time_err:+.4f}"
                
                f.write(f"{i+1:<5} | {true_name:<12} | {pred_name:<12} {type_mark} | {t_val:<15.6f} | {p_val:<15.6f} | {err_str:<10}\n")
            
            f.write("\n" + "="*95 + "\n\n")

    print(f"定性評価サンプルを保存しました: {example_path}")

def perform_quantitative_analysis(
    all_types_true, all_types_pred,
    all_times_true, all_time_preds,
    all_time_deltas_true,
    result_save_path, epoch,
    seq_scores=None  # ★追加
):
    """
    [定量評価] 詳細な分析を行い、ファイルに保存する
    """
    print(f"--- Epoch {epoch} 詳細な定量分析 を実行中 ---")
    
    analysis_dir = os.path.join(result_save_path, "analysis_results")
    os.makedirs(analysis_dir, exist_ok=True)

    # --- 1. 混合行列 (Confusion Matrix) (★修正: 2パターン保存) ---
    try:
        class_names = ['Large', 'Medium', 'Small'] 
        cm = confusion_matrix(all_types_true, all_types_pred, normalize='true')
        
        # 描画オブジェクト作成
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f')
        
        # ---------------------------------------------------------
        # パターンA: タイトルなし (論文用)
        # ---------------------------------------------------------
        plt.title("")  # タイトルを空にする
        cm_path_untitled = os.path.join(analysis_dir, f'{epoch}_confusion_matrix_untitled.png')
        plt.savefig(cm_path_untitled)
        
        # ---------------------------------------------------------
        # パターンB: タイトルあり (スライド用)
        # ---------------------------------------------------------
        plt.title(f'Epoch {epoch} - Normalized Confusion Matrix')  # タイトルを設定
        cm_path_titled = os.path.join(analysis_dir, f'{epoch}_confusion_matrix_titled.png')
        plt.savefig(cm_path_titled)
        
        plt.close()
        print(f"混合行列 を保存しました:\n  - {cm_path_untitled}\n  - {cm_path_titled}")
        
    except Exception as e:
        print(f"混合行列 の作成に失敗: {e}")

    # --- 2. 時間誤差分析 (★バグ修正あり) ---
    try:
        bins_hours = [0, 1, 6, 12, np.inf]
        bin_labels = ['(0-1 hour)', '(1-6 hours)', '(6-12 hours)', '(12+ hours)']
        
        all_time_deltas_true_hours = all_time_deltas_true * 24.0
        bin_indices = np.digitize(all_time_deltas_true_hours, bins=bins_hours)
        
        analysis_log_time = f"--- Epoch {epoch} 時間誤差分析 (RMSE of Absolute Time by Hours) ---\n"
        
        for i in range(1, len(bins_hours)):
            label = bin_labels[i-1]
            mask = (bin_indices == i)
            
            if np.sum(mask) == 0:
                analysis_log_time += f"{label}: N=0, (データなし)\n"
                continue
                
            true_abs_times_in_bin = all_times_true[mask]
            pred_abs_times_in_bin = all_time_preds[mask]
            
            rmse_in_bin = np.sqrt(np.mean((true_abs_times_in_bin - pred_abs_times_in_bin) ** 2))
            
            analysis_log_time += f"{label}: N={np.sum(mask)}, RMSE={rmse_in_bin:.4f}\n"

        print(analysis_log_time)
        
    except Exception as e:
        print(f"時間誤差分析 に失敗: {e}")
        # ▼▼▼【バグ修正】▼▼▼
        # (エラー時も変数を定義する)
        analysis_log_time = f"時間誤差分析 に失敗しました: {e}\n"
        # ▲▲▲【バグ修正】▲▲▲

    # --- 3. クラス別詳細レポート (★バグ修正あり) ---
    try:
        report = classification_report(
            all_types_true, 
            all_types_pred, 
            target_names=class_names, 
            digits=4
        )
        analysis_log_report = f"--- Epoch {epoch} クラス別分類レポート (Precision, Recall, F1-Score) ---\n"
        analysis_log_report += report + "\n"

        bal_acc = balanced_accuracy_score(all_types_true, all_types_pred)
        analysis_log_report += f"Balanced Accuracy (全クラスのRecall平均): {bal_acc:.4f}\n"

        print(analysis_log_report)
        
    except Exception as e:
        print(f"分類レポートの作成に失敗: {e}")
        # ▼▼▼【バグ修正】▼▼▼
        # (エラー時も変数を定義する)
        analysis_log_report = f"分類レポートの作成に失敗しました: {e}\n"
        # ▲▲▲【バグ修正】▲▲▲
    
    # --- 4. ログの統合 (変更なし) ---
    try:
        analysis_log_combined = analysis_log_time + "\n" + analysis_log_report
        
        txt_path = os.path.join(analysis_dir, f'{epoch}_analysis_report.txt')
        with open(txt_path, 'w') as f:
            f.write(analysis_log_combined)
        with open(os.path.join(result_save_path, 'val.txt'), 'a') as f:
            f.write(analysis_log_combined)
            
    except Exception as e:
        print(f"分析ログの保存に失敗: {e}")

    # ★ 4. シーケンス位置別分析の呼び出し
    if seq_scores and 'raw_data' in seq_scores:
        perform_detailed_sequence_analysis(seq_scores['raw_data'], analysis_dir, epoch)
        visualize_position_wise_difficulty(seq_scores, result_save_path, max_position=15)
        generate_qualitative_examples(seq_scores['raw_data'], analysis_dir, epoch, num_examples=30, mode='fixed')
        visualize_qualitative_sequence(seq_scores['raw_data'], analysis_dir, epoch, num_examples=30)
        visualize_qualitative_sequence_comparison(seq_scores['raw_data'], analysis_dir, epoch)
        visualize_performance_distribution(seq_scores['raw_data'], analysis_dir, epoch)
        visualize_global_distribution_comparison(seq_scores['raw_data'], analysis_dir, epoch)
        visualize_qualitative_sequence_comparison_separate(seq_scores['raw_data'], analysis_dir, epoch)
        visualize_large_event_predictions(seq_scores['raw_data'], analysis_dir, epoch)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_qualitative_sequence_comparison_separate(raw_data, save_dir, epoch, min_seq_len=10):
    """
    「意味のある（大きな地震を含む）シーケンス」の中から、
    Best, Median, Worst の3つを選出して保存する。
    """
    print(f"Generating separate qualitative plots (Best/Median/Worst) for epoch {epoch}...")
    
    # ターゲット: Large(0) か Medium(1) を含むシーケンスに限定する
    target_types = [0, 1] 
    print(f"Filter condition: Sequence length >= {min_seq_len} AND contains types {target_types}")

    sns.set_theme(style="white", context="talk", font_scale=1.0)

    # --- 1. スコア計算とフィルタリング ---
    scored_sequences = []
    
    skipped_len = 0
    skipped_trivial = 0
    
    for seq in raw_data:
        true_types = np.array(seq['true_types'])
        
        # (1) 長さフィルタ
        if len(true_types) < min_seq_len:
            skipped_len += 1
            continue
            
        # (2) 内容フィルタ (★ここが重要)
        # 指定したタイプ(Large/Medium)が1つも含まれていない場合はスキップ
        if not np.any(np.isin(true_types, target_types)):
            skipped_trivial += 1
            continue

        true_times = np.array(seq['true_times'])
        pred_times = np.array(seq['pred_times'])
        pred_types = np.array(seq['pred_types'])
        
        # 誤差計算
        time_mae = np.mean(np.abs(true_times - pred_times))
        type_mismatch = np.mean(true_types != pred_types)
        score = time_mae + (type_mismatch * 1.0)
        
        scored_sequences.append({'score': score, 'data': seq})
    
    print(f" -> Skipped (Short): {skipped_len}, Skipped (Trivial/Quiet): {skipped_trivial}")
    print(f" -> Candidates (Active sequences): {len(scored_sequences)}")

    # 候補が少なすぎる場合の救済措置 (Large/Mediumが一つもない場合など)
    if len(scored_sequences) < 3:
        print("Warning: Not enough 'active' sequences found. Falling back to all sequences.")
        scored_sequences = []
        for seq in raw_data:
             if len(seq['true_types']) >= min_seq_len:
                # 再計算...
                true_times = np.array(seq['true_times'])
                pred_times = np.array(seq['pred_times'])
                true_types = np.array(seq['true_types'])
                pred_types = np.array(seq['pred_types'])
                time_mae = np.mean(np.abs(true_times - pred_times))
                type_mismatch = np.mean(true_types != pred_types)
                score = time_mae + (type_mismatch * 1.0)
                scored_sequences.append({'score': score, 'data': seq})

    # スコアが良い順にソート
    scored_sequences.sort(key=lambda x: x['score'])
    
    n = len(scored_sequences)
    targets = [
        ("best",   scored_sequences[0]['data']),           # 活発な中での成功例
        ("median", scored_sequences[n // 2]['data']),      # 活発な中での典型例
        ("worst",  scored_sequences[-1]['data'])           # 活発な中での失敗例
    ]

    type_map_val = {0: 3.6, 1: 1.5, 2: 0.0}
    vis_dir = os.path.join(save_dir, "qualitative_plots_separate")
    os.makedirs(vis_dir, exist_ok=True)

    # --- 2. 保存処理 (前回と同じ) ---
    for suffix, seq in targets:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

        true_types = seq['true_types']
        pred_types = seq['pred_types']
        true_times = np.array(seq['true_times'])
        pred_times = np.array(seq['pred_times'])
        
        true_type_vals = np.array([type_map_val.get(t, 0) for t in true_types])
        pred_type_vals = np.array([type_map_val.get(p, 0) for p in pred_types])
        steps = np.arange(1, len(true_types) + 1)

        # === 上段: マグニチュード/タイプ ===
        ax1.axhspan(2.0, 5.2, color='red', alpha=0.1, label='Large Zone')
        ax1.axhspan(1.0, 2.0, color='orange', alpha=0.1, label='Medium Zone')
        ax1.axhspan(-1.02, 1.0, color='green', alpha=0.1, label='Small Zone')

        if 'true_mags' in seq:
            true_mags = np.array(seq['true_mags'])
            ax1.plot(steps, true_mags, color='black', marker='*', markersize=10, 
                     linestyle=':', linewidth=1.5, label='True Magnitude', alpha=0.7, zorder=5)
        else:
            ax1.plot(steps, true_type_vals, color='gray', marker='s', markersize=8, 
                     linestyle='--', label='True Type (Simulated)', alpha=0.5, zorder=5)

        ax1.plot(steps, true_type_vals, color='#1f77b4', marker='s', markersize=12, 
                 linestyle='--', label='True Type (Class)', alpha=0.4, zorder=4)
        ax1.plot(steps, pred_type_vals, color='orange', marker='o', markersize=8, 
                 linestyle='-', label='Predicted Type', alpha=0.9, zorder=6)
        
        ax1.set_ylim(-1.02, 5.2)
        #ax1.set_ylabel("Magnitude / Type")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10, framealpha=0.8)

        # === 下段: 絶対時刻 ===
        ax2.plot(steps, true_times, color='black', marker='s', label='True Time', linestyle='-', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', label='Predicted Time', linestyle='--', alpha=0.8)
        
        max_time_val = max(np.max(true_times), np.max(pred_times))
        ax2.set_ylim(0, max(1.0, max_time_val * 1.1))
        ax2.set_ylabel("Absolute Time")
        ax2.set_xlabel("Event Step")
        ax2.legend(loc='upper left', fontsize=10, framealpha=0.8)

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"epoch_comparison_{suffix}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        
        print(f"Saved: {save_path}")


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_large_event_predictions(raw_data, save_dir, epoch, min_seq_len=10):
    """
    「Large地震 (Type 0)」に注目し、以下の3つのケースを抽出して可視化する。
    1. Hit (成功): Largeがあり、予測もLargeだった
    2. Miss (見逃し): Largeがあったが、予測は外れた
    3. False Alarm (空振り): Largeはないが、予測でLargeが出た (オプション)
    """
    print(f"Generating Large Event Prediction plots for epoch {epoch}...")

    sns.set_theme(style="white", context="talk", font_scale=1.0)
    
    # カテゴリごとのリスト
    hit_sequences = []
    miss_sequences = []
    
    # Large (Type 0) の定義
    TARGET_TYPE = 0 
    
    for seq in raw_data:
        true_types = np.array(seq['true_types'])
        pred_types = np.array(seq['pred_types'])
        
        # 長さが足りないものはスキップ
        if len(true_types) < min_seq_len:
            continue
            
        # --- 1. Large地震が「実際に起きた」シーケンスを判定 ---
        large_indices = np.where(true_types == TARGET_TYPE)[0]
        
        if len(large_indices) > 0:
            # Large地震が含まれている場合 -> Hit か Miss か判定
            
            # 少なくとも1つのLarge地震を当てていれば「Hit」とみなす
            is_hit = False
            for idx in large_indices:
                if pred_types[idx] == TARGET_TYPE:
                    is_hit = True
                    break
            
            # スコア計算 (誤差が小さい順に並べるため)
            true_times = np.array(seq['true_times'])
            pred_times = np.array(seq['pred_times'])
            time_mae = np.mean(np.abs(true_times - pred_times))
            type_mismatch = np.mean(true_types != pred_types)
            score = time_mae + (type_mismatch * 1.0) # 小さいほど良い
            
            item = {'score': score, 'data': seq}
            
            if is_hit:
                hit_sequences.append(item)
            else:
                miss_sequences.append(item)

    # ソートしてベストな代表例を選ぶ
    hit_sequences.sort(key=lambda x: x['score'])
    miss_sequences.sort(key=lambda x: x['score'])
    
    # 出力対象の選定
    targets = []
    
    # (1) Hit Case (成功例)
    if len(hit_sequences) > 0:
        targets.append(("large_hit", hit_sequences[0]['data'], "Success: Large Event Predicted"))
    else:
        print("No 'Hit' sequences found (Large event correctly predicted).")

    # (2) Miss Case (見逃し例)
    if len(miss_sequences) > 0:
        # 典型的な失敗 (Median)
        mid_idx = len(miss_sequences) // 2
        targets.append(("large_miss", miss_sequences[mid_idx]['data'], "Failure: Large Event Missed"))
    else:
        print("No 'Miss' sequences found.")

    # --- 保存処理 ---
    type_map_val = {0: 3.6, 1: 1.5, 2: 0.0}
    vis_dir = os.path.join(save_dir, "large_event_plots")
    os.makedirs(vis_dir, exist_ok=True)

    for suffix, seq, title_text in targets:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

        true_types = seq['true_types']
        pred_types = seq['pred_types']
        true_times = np.array(seq['true_times'])
        pred_times = np.array(seq['pred_times'])
        
        true_type_vals = np.array([type_map_val.get(t, 0) for t in true_types])
        pred_type_vals = np.array([type_map_val.get(p, 0) for p in pred_types])
        steps = np.arange(1, len(true_types) + 1)

        # === 上段: マグニチュード/タイプ ===
        # ゾーン背景
        ax1.axhspan(2.0, 5.2, color='red', alpha=0.1, label='Large Zone')
        ax1.axhspan(1.0, 2.0, color='orange', alpha=0.1, label='Medium Zone')
        ax1.axhspan(-1.02, 1.0, color='green', alpha=0.1, label='Small Zone')

        # 1. 真のマグニチュード (星印)
        if 'true_mags' in seq:
            true_mags = np.array(seq['true_mags'])
            ax1.plot(steps, true_mags, color='black', marker='*', markersize=14, 
                     linestyle=':', linewidth=1.5, label='True Magnitude', alpha=0.7, zorder=5)
        else:
            ax1.plot(steps, true_type_vals, color='gray', marker='s', markersize=10, 
                     linestyle='--', label='True Type (Simulated)', alpha=0.5, zorder=5)

        # 2. ★追加: 真のイベントタイプ (青い四角)
        # これがないと「正解のクラス」が見えません
        ax1.plot(steps, true_type_vals, color='#1f77b4', marker='s', markersize=12, 
                 linestyle='--', label='True Type (Class)', alpha=0.4, zorder=4)

        # 3. 予測されたイベントタイプ (オレンジの丸)
        ax1.plot(steps, pred_type_vals, color='orange', marker='o', markersize=10, linewidth=2.5,
                 linestyle='-', label='Predicted Type', alpha=0.9, zorder=6)
        
        ax1.set_ylim(-1.02, 5.2)
        #ax1.set_ylabel("Magnitude / Type")
        #ax1.set_title(title_text, fontsize=14, fontweight='bold', color='#333333')
        
        # 凡例
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        #ax1.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10, framealpha=0.9)

        # === 下段: 絶対時刻 ===
        ax2.plot(steps, true_times, color='black', marker='s', label='True Time', linestyle='-', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', label='Predicted Time', linestyle='--', alpha=0.8)
        
        max_time_val = max(np.max(true_times), np.max(pred_times))
        ax2.set_ylim(0, max(1.0, max_time_val * 1.1))
        ax2.set_ylabel("Absolute Time")
        #ax2.set_xlabel("Event Step")
        #ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"epoch_{suffix}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")


def perform_detailed_sequence_analysis(raw_data, save_dir, epoch):
    """
    raw_data: リスト。各要素は { 'true_types': [], 'pred_types': [], 'true_times': [], 'pred_times': [] }
    """
    max_pos = 30  # 分析する最大イベント位置
    pos_rmses = [[] for _ in range(max_pos)]
    pos_accs = [[] for _ in range(max_pos)]
    
    seq_len_rmses = []
    seq_len_accs = []
    seq_lengths = []

    for seq in raw_data:
        t_typ = np.array(seq['true_types'])
        p_typ = np.array(seq['pred_types'])
        t_tim = np.array(seq['true_times'])
        p_tim = np.array(seq['pred_times'])
        length = len(t_typ)

        # 1. 位置別 (シーケンスの何番目のイベントか)
        for i in range(min(length, max_pos)):
            err = (t_tim[i] - p_tim[i])**2
            acc = 1.0 if t_typ[i] == p_typ[i] else 0.0
            pos_rmses[i].append(err)
            pos_accs[i].append(acc)

        # 2. シーケンス長さ別
        seq_rmse = np.sqrt(np.mean((t_tim - p_tim)**2))
        seq_acc = np.mean(t_typ == p_typ)
        seq_lengths.append(length)
        seq_len_rmses.append(seq_rmse)
        seq_len_accs.append(seq_acc)

    # 関数の上の方でこれらを計算しておけば、グラフとテキスト両方に対応できます
    avg_pos_rmse = [np.sqrt(np.mean(pos_rmses[i])) if len(pos_rmses[i]) > 0 else 0 for i in range(max_pos)]
    avg_pos_acc = [np.mean(pos_accs[i]) if len(pos_accs[i]) > 0 else 0 for i in range(max_pos)]
    # --- 位置別グラフ作成 ---
    # 【ここを差し替え】以前の avg_pos_rmse などの計算の代わりに以下を入れます
    valid_indices = [i for i, x in enumerate(pos_rmses) if len(x) > 0]
    display_rmse = [np.sqrt(np.mean(pos_rmses[i])) for i in valid_indices]
    display_acc = [np.mean(pos_accs[i]) for i in valid_indices]
    display_x = [i + 1 for i in valid_indices]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(range(1, max_pos + 1), avg_pos_rmse, ...) の代わりにこれを使います
    plt.plot(display_x, display_rmse, marker='o', color='red') 
    plt.title('RMSE by Event Position in Sequence')
    plt.xlabel('Position (1st, 2nd, ...)')
    plt.ylabel('RMSE (Seconds)')
    
    plt.subplot(1, 2, 2)
    # Accuracy側も同様に display_x, display_acc を使います
    plt.plot(display_x, display_acc, marker='o', color='blue')
    plt.title('Accuracy by Event Position')
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(save_dir, f'{epoch}_positional_analysis.png'))
    plt.close()

    # --- 長さ別グラフ作成 (散布図) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(seq_lengths, seq_len_rmses, alpha=0.5, color='orange')
    plt.title('Sequence Length vs RMSE')
    plt.xlabel('Total Events in Sequence')
    plt.ylabel('RMSE')
    
    plt.subplot(1, 2, 2)
    plt.scatter(seq_lengths, seq_len_accs, alpha=0.5, color='green')
    plt.title('Sequence Length vs Accuracy')
    plt.xlabel('Total Events in Sequence')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(save_dir, f'{epoch}_length_analysis.png'))
    plt.close()

    # テキストレポート出力
    with open(os.path.join(save_dir, f'{epoch}_detailed_stats.txt'), 'w') as f:
        f.write("Position, RMSE, Accuracy, N_Samples\n")
        for i in range(max_pos):
            f.write(f"{i+1}, {avg_pos_rmse[i]:.4f}, {avg_pos_acc[i]:.4f}, {len(pos_rmses[i])}\n")
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
def visualize_position_wise_difficulty(seq_scores, save_path, max_position=20):
    """
    シーケンス内の「位置（何個目のイベントか）」ごとの予測精度を可視化する。
    初期位置（Position 0, 1...）のエラーが高いことを証明するためのグラフ。
    
    :param seq_scores: Runnerから出力された seq_scores 辞書 (raw_dataを含む)
    :param save_path: 保存先のディレクトリパス
    :param max_position: グラフに表示する最大ポジション（長すぎると後半ノイズが増えるため20程度推奨）
    """
    raw_data = seq_scores['raw_data']
    
    # 1. 位置ごとのエラーを集計する辞書
    # pos_sq_errors[0] = [ (pred-true)^2, ... ]
    pos_time_sq_errors = defaultdict(list)
    pos_type_hits = defaultdict(list)
    pos_counts = defaultdict(int)

    print("集計中...")
    for seq in raw_data:
        # データの取り出し (リスト型を想定)
        true_times = seq['true_times']
        pred_times = seq['pred_times']
        true_types = seq['true_types']
        pred_types = seq['pred_types']
        
        # シーケンスの長さ分ループ
        seq_len = len(true_times)
        for i in range(seq_len):
            if i >= max_position: break # 指定位置以降はカット
            
            # 時間予測誤差 (二乗誤差)
            time_err = (true_times[i] - pred_times[i]) ** 2
            pos_time_sq_errors[i].append(time_err)
            
            # タイプ予測正誤 (1 or 0)
            type_hit = 1 if true_types[i] == pred_types[i] else 0
            pos_type_hits[i].append(type_hit)
            
            pos_counts[i] += 1

    # 2. 平均値（RMSE, Accuracy）を計算
    positions = sorted(pos_time_sq_errors.keys())
    rmses = []
    accuracies = []
    counts = []
    
    for p in positions:
        # RMSE = sqrt(mean(squared_errors))
        rmse = np.sqrt(np.mean(pos_time_sq_errors[p]))
        acc = np.mean(pos_type_hits[p])
        
        rmses.append(rmse)
        accuracies.append(acc)
        counts.append(pos_counts[p])

    # 3. プロット作成（2軸グラフ）
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # X軸の設定
    ax1.set_xlabel('Position in Sequence (History Length)')
    ax1.set_xticks(positions)
    ax1.set_xlim(-0.5, max_position - 0.5)

    # --- 左軸: RMSE (時間誤差) ---
    color = 'tab:red'
    ax1.set_ylabel('Time Prediction RMSE (Lower is Better)', color=color, fontweight='bold')
    ax1.plot(positions, rmses, marker='o', color=color, linewidth=2, label='Time RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 右軸: Accuracy (タイプ正解率) ---
    ax2 = ax1.twinx()  # 2軸目を作成
    color = 'tab:blue'
    ax2.set_ylabel('Type Accuracy (Higher is Better)', color=color, fontweight='bold')
    ax2.plot(positions, accuracies, marker='s', color=color, linewidth=2, linestyle='--', label='Type Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0) # Accuracyは0~1

    # --- サンプル数の表示（棒グラフで薄く背景に）---
    # サンプル数が十分かどうかの確認用
    # グラフが見にくくなる場合はコメントアウトしてもOK
    """
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15)) # 軸をさらに外側に
    ax3.bar(positions, counts, alpha=0.1, color='gray', label='Sample Count')
    ax3.set_ylabel('Sample Count', color='gray')
    ax3.set_ylim(0, max(counts)*1.5)
    """

    # タイトルと保存
    plt.title('Prediction Difficulty by Sequence Position (The "Cold Start" Problem)', fontsize=14)
    fig.tight_layout()
    
    save_file = os.path.join(save_path, "position_wise_difficulty.png")
    plt.savefig(save_file)
    print(f"グラフを保存しました: {save_file}")
    plt.close()

def visualize_semantic_space(model, tokenizer, save_dir, epoch, device='cpu'):
    """
    MLPが出力する数値埋め込みと、LLMが持つ単語埋め込みを可視化する。
    【最終修正版】
    ★ アンカーが存在しない場合（損失なしのベースライン等）でもエラー落ちせず、
      星印なしでプロットできるように修正。
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F

    # -------------------------------------------------
    # 保存先ディレクトリ
    # -------------------------------------------------
    vis_root = os.path.join(save_dir, 'visualizations')
    epoch_dir = os.path.join(vis_root, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    log_path = os.path.join(epoch_dir, "analysis_log.txt")

    # -------------------------------------------------
    # 定義
    # -------------------------------------------------
    word_anchors = {
        "Magnitude": ["Magnitude", "Large Magnitude", "Small Magnitude", "Medium Magnitude"],
        "Depth": ["Depth", "Deep Depth", "Shallow Depth", "Intermediate Depth"],
        "Time": ["Time", "Early Time", "Late Time", "Middle Time"]
    }
    control_words = ["Apple", "Banana", "Music", "Piano", "Love", "Happy", "Dog", "Cat", "Computer", "Internet"]
    
    # -------------------------------------------------
    # 範囲(Ranges)自動設定 (属性がない場合はデフォルト値)
    # -------------------------------------------------
    max_m = model.max_mag if hasattr(model, 'max_mag') else 9.0
    max_d = model.max_dep if hasattr(model, 'max_dep') else 700.0
    max_t = model.max_time if hasattr(model, 'max_time') else 100.0

    ranges = {
        'mag':  np.linspace(0, max_m, 50),
        'dep':  np.linspace(0, max_d, 50),
        'time': np.linspace(0, max_t, 50)
    }
    
    targets = [
        ('mag',  model.mag_mlp,  "Magnitude"),
        ('dep',  model.dep_mlp,  "Depth"),
        ('time', model.time_mlp, "Time")
    ]
    
    model.eval()
    
    # -------------------------------------------------
    # Helper functions
    # -------------------------------------------------
    def get_nearest_neighbors(query_vec, k=3):
        emb_weight = model.llm.get_input_embeddings().weight 
        sim = F.cosine_similarity(query_vec.unsqueeze(0), emb_weight, dim=1)
        top_k_val, top_k_idx = torch.topk(sim, k)
        
        neighbor_ids = top_k_idx.tolist()
        neighbor_words = tokenizer.convert_ids_to_tokens(neighbor_ids)
        neighbor_scores = top_k_val.tolist()
        
        with torch.no_grad():
            neighbor_vecs = emb_weight[top_k_idx].cpu().numpy()
            
        return neighbor_words, neighbor_scores, neighbor_vecs

    control_vecs = []
    for word in control_words:
        inputs = tokenizer(word, return_tensors='pt', add_special_tokens=False).to(device)
        with torch.no_grad():
            emb_layer = model.llm.get_input_embeddings()
            vec = emb_layer(inputs['input_ids']).mean(dim=1).squeeze().cpu().numpy()
        control_vecs.append(vec)
    control_vecs = np.array(control_vecs)

    # -------------------------------------------------
    # Main Loop
    # -------------------------------------------------
    with open(log_path, 'w', encoding='utf-8') as log_file:
        def log(text):
            print(text)
            log_file.write(text + "\n")

        fig_comb, axes_comb = plt.subplots(1, 3, figsize=(24, 7))
        
        log("-" * 60)
        log(f"Epoch {epoch}: Semantic Decoding Analysis")
        log(f"Visualization Ranges: Mag=0~{max_m:.1f}, Dep=0~{max_d:.1f}, Time=0~{max_t:.1f}")
        log("-" * 60)

        for i, (key, mlp, label) in enumerate(targets):
            log(f">> Analyzing {label} Space:")
            
            # --- A. Target Words ---
            target_words = word_anchors.get(label, [])
            word_vecs = []
            word_labels = []
            for word in target_words:
                inputs = tokenizer(word, return_tensors='pt', add_special_tokens=False).to(device)
                with torch.no_grad():
                    emb_layer = model.llm.get_input_embeddings()
                    vec = emb_layer(inputs['input_ids']).mean(dim=1).squeeze().cpu().numpy()
                word_vecs.append(vec)
                word_labels.append(word)
            word_vecs = np.array(word_vecs)

            # --- B. Numeric MLP ---
            raw_vals = ranges[key]
            inputs_raw = torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(1).to(device)
            
            with torch.no_grad():
                mlp_vecs_tensor = mlp(inputs_raw)
                mlp_vecs = mlp_vecs_tensor.cpu().numpy()
                
                # Nearest Neighbors Log
                min_words, min_scores, min_n_vecs = get_nearest_neighbors(mlp_vecs_tensor[0], k=3)
                log(f"  [Min(0.0)]: {', '.join([f'{w}({s:.2f})' for w,s in zip(min_words, min_scores)])}")
                
                mid_idx = len(mlp_vecs_tensor) // 2
                mid_val = raw_vals[mid_idx]
                mid_words, mid_scores, mid_n_vecs = get_nearest_neighbors(mlp_vecs_tensor[mid_idx], k=3)
                log(f"  [Mid({mid_val:.1f})]: {', '.join([f'{w}({s:.2f})' for w,s in zip(mid_words, mid_scores)])}")

                max_words, max_scores, max_n_vecs = get_nearest_neighbors(mlp_vecs_tensor[-1], k=3)
                log(f"  [Max({raw_vals[-1]:.1f})]: {', '.join([f'{w}({s:.2f})' for w,s in zip(max_words, max_scores)])}")
                log("-" * 30)
                
                neighbor_vecs = np.vstack([min_n_vecs, mid_n_vecs, max_n_vecs])
                
                labels_min = [f"{w}\n({s:.2f})" for w, s in zip(min_words, min_scores)]
                labels_mid = [f"{w}\n({s:.2f})" for w, s in zip(mid_words, mid_scores)]
                labels_max = [f"{w}\n({s:.2f})" for w, s in zip(max_words, max_scores)]

            # --- C. Anchors (存在確認: 1点 or 3点 or なし) ---
            anchor_vecs = []
            anchor_names = [] # ラベル用
            
            base_attr = f"anchor_{key}" # anchor_mag, anchor_dep, anchor_time
            
            # 3点アンカーがあるかチェック (min と max があれば3点とみなす)
            if hasattr(model, f"{base_attr}_min") and hasattr(model, f"{base_attr}_max"):
                # 3点アンカー取得
                for suffix, a_name in [("_min", "Anchor(Min)"), ("_mid", "Anchor(Mid)"), ("_max", "Anchor(Max)")]:
                    attr_name = f"{base_attr}{suffix}"
                    if hasattr(model, attr_name):
                        vec = getattr(model, attr_name).detach().cpu().numpy()
                        anchor_vecs.append(vec)
                        #anchor_names.append(a_name)
            
            # 3点がない場合、1点アンカーがあるかチェック
            elif hasattr(model, base_attr):
                vec = getattr(model, base_attr).detach().cpu().numpy()
                anchor_vecs.append(vec)
                #anchor_names.append("Anchor")
            
            # アンカーが見つかった場合のみ numpy化、なければ None のまま
            anchor_vecs = np.array(anchor_vecs) if len(anchor_vecs) > 0 else None

            # --- D. PCA ---
            vectors_list = [word_vecs, mlp_vecs]
            
            # ★ 修正: アンカーがある場合のみリストに追加
            if anchor_vecs is not None: 
                vectors_list.append(anchor_vecs)
                
            vectors_list.append(control_vecs)
            vectors_list.append(neighbor_vecs)
            
            combined_vecs = np.vstack(vectors_list)
            pca = PCA(n_components=2)
            reduced_vecs = pca.fit_transform(combined_vecs)
            
            # Split
            idx = 0
            n_words = len(word_vecs)
            r_words = reduced_vecs[idx : idx + n_words]; idx += n_words
            
            n_mlp = len(mlp_vecs)
            r_mlp = reduced_vecs[idx : idx + n_mlp]; idx += n_mlp
            
            # ★ 修正: アンカーの取り出しも条件分岐
            r_anchors = None
            if anchor_vecs is not None:
                n_anchors = len(anchor_vecs)
                r_anchors = reduced_vecs[idx : idx + n_anchors]; idx += n_anchors
            
            n_ctrl = len(control_vecs)
            r_ctrl = reduced_vecs[idx : idx + n_ctrl]; idx += n_ctrl
            
            n_neigh = len(neighbor_vecs)
            r_neigh = reduced_vecs[idx : idx + n_neigh]
            
            n_min = len(min_n_vecs)
            n_mid = len(mid_n_vecs)
            r_neigh_min = r_neigh[:n_min]
            r_neigh_mid = r_neigh[n_min : n_min + n_mid]
            r_neigh_max = r_neigh[n_min + n_mid:]

            # --- Plot ---
            def plot_on_ax(ax, title_text, point_size_scale=1.0):
                # Control Words
                ax.scatter(r_ctrl[:, 0], r_ctrl[:, 1], c='black', marker='.', s=30*point_size_scale, alpha=0.2, label='Unrelated')
                
                # Numeric
                sc = ax.scatter(r_mlp[:, 0], r_mlp[:, 1], c=raw_vals, cmap='plasma', s=50*point_size_scale, alpha=0.7, label='Numeric')
                ax.plot(r_mlp[:, 0], r_mlp[:, 1], c='gray', alpha=0.3, linewidth=1)
                ax.annotate("0", (r_mlp[0, 0], r_mlp[0, 1]), fontsize=12*point_size_scale, fontweight='bold', color='black')
                ax.annotate(f"{raw_vals[-1]:.1f}", (r_mlp[-1, 0], r_mlp[-1, 1]), fontsize=12*point_size_scale, fontweight='bold', color='black')

                # Target Words
                ax.scatter(r_words[:, 0], r_words[:, 1], c='red', marker='x', s=100*point_size_scale, label='Target Words')
                for j, txt in enumerate(word_labels):
                    ax.annotate(txt, (r_words[j, 0], r_words[j, 1]), fontsize=11*point_size_scale, fontweight='bold', color='darkred')
                
                # Neighbors
                ax.scatter(r_neigh_min[:, 0], r_neigh_min[:, 1], c='cyan', marker='v', s=80*point_size_scale, edgecolors='blue', label='Pred (Min)')
                for j, txt in enumerate(labels_min):
                    ax.annotate(txt, (r_neigh_min[j, 0], r_neigh_min[j, 1]), fontsize=8*point_size_scale, color='blue', alpha=0.9, xytext=(-5, -15), textcoords='offset points', ha='center')

                ax.scatter(r_neigh_mid[:, 0], r_neigh_mid[:, 1], c='lime', marker='D', s=80*point_size_scale, edgecolors='green', label='Pred (Mid)')
                for j, txt in enumerate(labels_mid):
                    ax.annotate(txt, (r_neigh_mid[j, 0], r_neigh_mid[j, 1]), fontsize=8*point_size_scale, color='green', alpha=0.9, xytext=(10, 0), textcoords='offset points', ha='left')

                ax.scatter(r_neigh_max[:, 0], r_neigh_max[:, 1], c='orange', marker='^', s=80*point_size_scale, edgecolors='darkred', label='Pred (Max)')
                for j, txt in enumerate(labels_max):
                    ax.annotate(txt, (r_neigh_max[j, 0], r_neigh_max[j, 1]), fontsize=8*point_size_scale, color='darkred', alpha=0.9, xytext=(5, 5), textcoords='offset points', ha='center')

                # ★ Anchors (存在する場合のみプロット) ★
                if r_anchors is not None:
                    ax.scatter(r_anchors[:, 0], r_anchors[:, 1], c='gold', marker='*', s=300*point_size_scale, edgecolors='black', label='Anchor')
                    for j, txt in enumerate(anchor_names):
                        ax.annotate(txt, (r_anchors[j, 0], r_anchors[j, 1]), 
                                    fontsize=9*point_size_scale, fontweight='bold', color='goldenrod', 
                                    xytext=(5, 5), textcoords='offset points')
                
                ax.set_title(title_text, fontsize=14*point_size_scale)
                ax.grid(True, linestyle='--', alpha=0.3)
                return sc

            sc_comb = plot_on_ax(axes_comb[i], title_text=f"{label} Space")
            plt.colorbar(sc_comb, ax=axes_comb[i], label=f'{label}')

            fig_single, ax_single = plt.subplots(figsize=(10, 8))
            sc_single = plot_on_ax(ax_single, title_text=f"{label} Semantic Space (Epoch {epoch})", point_size_scale=1.2)
            plt.colorbar(sc_single, ax=ax_single, label=f'{label} Value')
            
            ax_single.set_xlabel('PCA Dim 1')
            ax_single.set_ylabel('PCA Dim 2')
            ax_single.legend(loc='best')
            
            plt.savefig(os.path.join(epoch_dir, f"{label}.png"))
            plt.close(fig_single)

        axes_comb[0].legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, "Combined.png"))
        plt.close(fig_comb)
        
        log(f"Visualizations and log saved to {epoch_dir}")

def evaluate_distance_preservation(model, save_dir, epoch, device='cpu'):
    """
    t-SNE Lossの効果を可視化する。
    1. Shepard Diagram: 入力距離 vs 出力距離の全ペア相関
    2. Linearity Check: 入力値の増加に対して、出力ベクトルがどれだけ直線的に離れていくか
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    import os

    epoch_dir = os.path.join(save_dir, 'visualizations', f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # 解析対象の取得
    targets = [
        ('Magnitude', model.mag_mlp, model.max_mag if hasattr(model, 'max_mag') else 9.0, 'blue'),
        ('Depth',     model.dep_mlp, model.max_dep if hasattr(model, 'max_dep') else 700.0, 'green'),
        ('Time',      model.time_mlp, model.max_time if hasattr(model, 'max_time') else 100.0, 'orange')
    ]

    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for i, (label, mlp, max_val, color) in enumerate(targets):
        # 1. サンプリング (100点)
        raw_vals = np.linspace(0, max_val, 100)
        inputs = torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            vecs = mlp(inputs).cpu().numpy()
        
        # 2. 距離行列の計算 (pdist)
        # 入力スカラー空間の距離
        dist_input = pdist(raw_vals.reshape(-1, 1), metric='euclidean')
        # 出力ベクトル空間の距離
        dist_output = pdist(vecs, metric='euclidean')
        
        # 3. 相関係数 (スピアマン)
        corr, _ = spearmanr(dist_input, dist_output)
        
        # 4. プロット
        ax = axes[i]
        # 散布図 (密になりすぎるのを防ぐため alpha を調整)
        ax.scatter(dist_input, dist_output, s=2, alpha=0.3, c=color)
        
        # 理論的な理想線 (原点を通る直線) を描画
        ideal_line = np.linspace(0, dist_input.max(), 100)
        scale_factor = dist_output.max() / dist_input.max()
        ax.plot(ideal_line, ideal_line * scale_factor, 'r--', alpha=0.5, label='Ideal Linear')
        
        ax.set_title(f"{label} Distance Preservation\nSpearman Corr: {corr:.4f}", fontsize=14)
        ax.set_xlabel("Input Scalar Distance (|x_i - x_j|)")
        ax.set_ylabel("Output Vector Distance (||v_i - v_j||)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(epoch_dir, "tsne_distance_analysis.png")
    plt.savefig(save_path)
    plt.close()
    print(f">> t-SNE Analysis saved: {save_path}")

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from scipy.stats import shapiro, norm
# import plotly.graph_objects as go  # 3Dを使用しないためコメントアウトまたは削除可能です

def run_full_evaluation(model, tokenizer, save_dir, epoch, device='cpu'):
    """
    【統合分析関数・3D削除版】
    ・Magnitude, Depth, Time の個別2D plot等は維持。
    ・メモリ消費の大きい3D Plotly可視化処理を削除しました。
    """
    print(f"\n=======================================================")
    print(f"   STARTING FULL EVALUATION (Epoch {epoch}) - 2D Only")
    print(f"=======================================================\n")

    vis_root = os.path.join(save_dir, 'visualizations')
    epoch_dir = os.path.join(vis_root, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    log_path = os.path.join(epoch_dir, "analysis_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        def log(text):
            print(text)
            f.write(text + "\n")

        log(f"Analysis Results for Epoch {epoch}")
        log("-" * 50)

        # -------------------------------------------------
        # 共通設定 & データ準備
        # -------------------------------------------------
        max_m = getattr(model, 'max_mag', 9.0)
        max_d = getattr(model, 'max_dep', 700.0)
        max_t = getattr(model, 'max_time', 100.0)
        min_m = getattr(model, 'min_mag', -2.0) 
        min_d = getattr(model, 'min_dep', 0.0)
        min_t = getattr(model, 'min_time', 0.0)

        targets = [
            {
                'key': 'mag', 'mlp': model.mag_mlp, 'label': "Magnitude", 
                'min': min_m, 'max': max_m, 
                'anchor_target': "Magnitude", 
                'mid_val': (min_m + max_m) / 2
            },
            {
                'key': 'dep', 'mlp': model.dep_mlp, 'label': "Depth",     
                'min': min_d, 'max': max_d, 
                'anchor_target': "Depth",     
                'mid_val': (min_d + max_d) / 2
            },
            {
                'key': 'time', 'mlp': model.time_mlp, 'label': "Time",      
                'min': min_t, 'max': max_t, 
                'anchor_target': "Time",      
                'mid_val': (min_t + max_t) / 2
            }
        ]

        model.eval()

        # -------------------------------------------------
        # 1. Semantic Distribution Analysis (分布形状と重心解析 + 分散)
        # -------------------------------------------------
        log("\n[1] Semantic Distribution Analysis (Histogram, Normality, Centroid & Variance)")
        log("-" * 40)
        
        # 分布プロット用のFigure
        fig_dist, axes_dist = plt.subplots(len(targets), 2, figsize=(16, 5 * len(targets)))
        if len(targets) == 1: axes_dist = axes_dist.reshape(1, -1)

        for i, t in enumerate(targets):
            # --- A. ベクトル取得 ---
            inputs_w = tokenizer(t['anchor_target'], return_tensors='pt', add_special_tokens=False).to(device)
            with torch.no_grad():
                target_vec = model.llm.get_input_embeddings()(inputs_w['input_ids']).mean(dim=1) 
            
            # 数値データ群 (500分割)
            raw_vals = np.linspace(t['min'], t['max'], 500)
            inputs_n = torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(1).to(device)
            
            with torch.no_grad():
                mlp_vecs = t['mlp'](inputs_n) # (500, hidden_size)
            
            # --- B. 埋め込みの平均 (Centroid) ---
            mean_vec = torch.mean(mlp_vecs, dim=0)

            # --- C. ★追加: 分散 (Variance) の計算 ---
            # 各次元ごとの分散を計算し、その総和をとる (Total Variance)
            # これが「分布の体積（広がり）」を表す指標になります
            var_per_dim = torch.var(mlp_vecs, dim=0) 
            total_variance = torch.sum(var_per_dim).item()
            
            # (参考) 平均分散
            mean_variance = torch.mean(var_per_dim).item()

            # --- D. 距離・類似度計算 ---
            # 1. 重心 vs ターゲット
            dist_centroid_target = torch.norm(mean_vec - target_vec).item()
            sim_centroid_target  = F.cosine_similarity(mean_vec.unsqueeze(0), target_vec).item()
            
            # 2. 個々の点 vs ターゲット
            dists_target = torch.norm(mlp_vecs - target_vec, dim=1).cpu().numpy()
            sims_target  = F.cosine_similarity(mlp_vecs, target_vec).cpu().numpy()
            
            # 3. 個々の点 vs 重心 (Spread)
            dists_mean = torch.norm(mlp_vecs - mean_vec, dim=1).cpu().numpy()
            sims_mean  = F.cosine_similarity(mlp_vecs, mean_vec.unsqueeze(0)).cpu().numpy()

            # --- E. 正規性検定 ---
            stat_dist, p_dist = shapiro(dists_target)
            is_normal = "Likely Normal" if p_dist > 0.05 else "Not Normal"

            # --- F. ログ出力 ---
            log(f" >> {t['label']} Distribution Analysis (N=500)")
            log(f"    Target Word: '{t['anchor_target']}'")
            
            log(f"    [Alignment] Centroid vs Target")
            log(f"       - Centroid-Target Dist: {dist_centroid_target:.4f}")
            log(f"       - Centroid-Target Sim:  {sim_centroid_target:.4f}")
            
            log(f"    [Spread / Variance] Distribution Metrics")
            log(f"       - Total Variance:        {total_variance:.4f}  (Higher = Richer Info, ~0 = Collapse)")
            log(f"       - Mean Variance per Dim: {mean_variance:.6f}")
            log(f"       - Dist Mean (to Self):   {np.mean(dists_mean):.4f}")
            log(f"       - Sim Mean (to Self):    {np.mean(sims_mean):.4f}  (If >0.99, check Variance)")

            # --- G. プロット作成 ---
            # 1. 距離 (対ターゲット)
            ax_d = axes_dist[i, 0]
            sns.histplot(dists_target, kde=True, ax=ax_d, color='skyblue', bins=30)
            # 正規分布近似線
            xmin, xmax = ax_d.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            ax_d.plot(x, norm.pdf(x, np.mean(dists_target), np.std(dists_target)) * len(dists_target) * (xmax-xmin)/30, 'r--', alpha=0.5)
            
            ax_d.set_title(f"{t['label']}: Distance to Target\n(Var: {total_variance:.2f})")
            ax_d.set_xlabel("Euclidean Distance")
            
            # 2. 類似度 (対ターゲット)
            ax_s = axes_dist[i, 1]
            sns.histplot(sims_target, kde=True, ax=ax_s, color='orange', bins=30)
            ax_s.set_title(f"{t['label']}: Cosine Similarity to Target")
            ax_s.set_xlabel("Cosine Similarity")

        plt.tight_layout()
        save_path = os.path.join(epoch_dir, "Distribution_Analysis_Full.png")
        plt.savefig(save_path)
        plt.close(fig_dist)
        log(f"    -> Saved full distribution plot: {save_path}")

        # -------------------------------------------------
        # 2. Distance Preservation Analysis (Shepard Diagram)
        # -------------------------------------------------
        log("\n[2] Distance Preservation Analysis (Shepard Diagram)")
        log("-" * 40)
        
        fig_shep, axes_shep = plt.subplots(1, 3, figsize=(24, 7))
        colors = ['blue', 'green', 'orange']

        for i, t in enumerate(targets):
            raw_vals = np.linspace(t['min'], t['max'], 100)
            inputs = torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                vecs = t['mlp'](inputs).cpu().numpy()
            
            dist_input = pdist(raw_vals.reshape(-1, 1), metric='euclidean')
            dist_output = pdist(vecs, metric='euclidean')
            
            corr_spearman, _ = spearmanr(dist_input, dist_output)
            corr_pearson, _ = pearsonr(dist_input, dist_output)

            # Step Consistency
            diff_input = np.diff(raw_vals)
            diff_output = np.linalg.norm(vecs[1:] - vecs[:-1], axis=1)
            valid_idx = diff_input > 1e-6
            ratios = diff_output[valid_idx] / diff_input[valid_idx]
            
            if np.mean(ratios) > 0:
                cv = np.std(ratios) / np.mean(ratios)
            else:
                cv = 0.0

            log(f" >> {t['label']} ({t['min']:.1f}~{t['max']:.1f}):")
            log(f"    - Spearman Correlation = {corr_spearman:.4f}")
            log(f"    - Pearson Correlation  = {corr_pearson:.4f}")
            log(f"    - Step Consistency CV  = {cv:.4f}")

            ax = axes_shep[i]
            ax.scatter(dist_input, dist_output, s=2, alpha=0.3, c=colors[i])
            
            ideal_x = np.linspace(0, dist_input.max(), 100)
            scale_factor = dist_output.max() / dist_input.max() if dist_input.max() > 0 else 1.0
            ax.plot(ideal_x, ideal_x * scale_factor, 'r--', alpha=0.5, label='Ideal Linear')
            
            ax.set_title(f"{t['label']}\nSpearman: {corr_spearman:.4f} / Pearson: {corr_pearson:.4f}", fontsize=12)
            ax.set_xlabel("Input Distance")
            ax.set_ylabel("Embedding Distance")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, "Distance_Preservation.png"))
        plt.close(fig_shep)
        log("    -> Saved: Distance_Preservation.png")

        # -------------------------------------------------
        # 3. 2D Semantic Space Visualization (PCA) - SIMPLE
        # -------------------------------------------------
        log("\n[3] 2D Semantic Space Visualization (Numeric & Anchor Only)")
        log("-" * 40)

        # 描画用ヘルパー関数
        def plot_simple_2d(ax, title, r_m, raw_v, r_a, a_nms):
            # Numeric (MLP Trajectory)
            sc = ax.scatter(r_m[:,0], r_m[:,1], c=raw_v, cmap='plasma', s=80, alpha=0.9, label='Numeric Trajectory', edgecolors='white', linewidth=0.5)
            # 線でつなぐ
            ax.plot(r_m[:,0], r_m[:,1], c='gray', alpha=0.5, linewidth=2, linestyle='-')
            
            # 開始点と終了点のラベル
            ax.annotate(f"Min\n{raw_v[0]:.1f}", (r_m[0,0], r_m[0,1]), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            ax.annotate(f"Max\n{raw_v[-1]:.1f}", (r_m[-1,0], r_m[-1,1]), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            
            # Anchors
            if r_a is not None:
                ax.scatter(r_a[:,0], r_a[:,1], c='gold', marker='*', s=400, edgecolors='black', label='Learned Anchor', zorder=10)
                for j, nm in enumerate(a_nms):
                    if nm:
                        ax.annotate(nm, (r_a[j,0], r_a[j,1]), xytext=(0, 10), textcoords='offset points', 
                                    ha='center', fontsize=11, fontweight='bold', color='darkgoldenrod', 
                                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
            
            ax.set_title(title, fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            return sc

        # 統合プロット用のFigure作成
        fig_comb, axes_comb = plt.subplots(1, 3, figsize=(24, 8))

        for i, t in enumerate(targets):
            # Numeric MLP Path
            raw_vals_50 = np.linspace(t['min'], t['max'], 50)
            inputs_50 = torch.tensor(raw_vals_50, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                mlp_vecs_tensor = t['mlp'](inputs_50)
                mlp_vecs = mlp_vecs_tensor.cpu().numpy()

            # Nearest Neighbors (ログ出力用のみ計算)
            def get_nn(query_vec, k=3):
                emb_weight = model.llm.get_input_embeddings().weight 
                sim = F.cosine_similarity(query_vec.unsqueeze(0), emb_weight, dim=1)
                vals, idxs = torch.topk(sim, k)
                return tokenizer.convert_ids_to_tokens(idxs.tolist()), vals.tolist()

            min_w, min_s = get_nn(mlp_vecs_tensor[0])
            mid_w, mid_s = get_nn(mlp_vecs_tensor[25])
            max_w, max_s = get_nn(mlp_vecs_tensor[-1])
            
            log(f" >> {t['label']} Neighbors (Excluded from Plot):")
            log(f"    - Min({raw_vals_50[0]:.1f}): {', '.join([f'{w}({s:.2f})' for w,s in zip(min_w, min_s)])}")
            log(f"    - Mid({raw_vals_50[25]:.1f}): {', '.join([f'{w}({s:.2f})' for w,s in zip(mid_w, mid_s)])}")
            log(f"    - Max({raw_vals_50[-1]:.1f}): {', '.join([f'{w}({s:.2f})' for w,s in zip(max_w, max_s)])}")

            # Anchors
            anchor_vecs, anchor_names = [], []
            base_attr = f"anchor_{t['key']}"
            if hasattr(model, f"{base_attr}_min") and hasattr(model, f"{base_attr}_max"):
                for sfx, nm in [("_min", "Anc(Min)"), ("_mid", "Anc(Mid)"), ("_max", "Anc(Max)")]:
                    if hasattr(model, f"{base_attr}{sfx}"):
                        anchor_vecs.append(getattr(model, f"{base_attr}{sfx}").detach().cpu().numpy())
                        anchor_names.append(nm)
            elif hasattr(model, base_attr):
                anchor_vecs.append(getattr(model, base_attr).detach().cpu().numpy())
                anchor_names.append(f"Anc({t['label']})") 
            
            anchor_vecs = np.array(anchor_vecs) if len(anchor_vecs) > 0 else None

            # --- PCA Execution (MLP + Anchor ONLY) ---
            vectors_list = [mlp_vecs]
            if anchor_vecs is not None:
                vectors_list.append(anchor_vecs)
            
            combined = np.vstack(vectors_list)
            
            n_components = 2
            if combined.shape[0] < 2:
                n_components = 1
            
            pca = PCA(n_components=n_components)
            if n_components == 1:
                reduced_1d = pca.fit_transform(combined)
                reduced = np.hstack([reduced_1d, np.zeros_like(reduced_1d)])
            else:
                reduced = pca.fit_transform(combined)

            # Split Data
            idx = 0
            r_mlp = reduced[idx : idx+len(mlp_vecs)]; idx+=len(mlp_vecs)
            r_anc = None
            if anchor_vecs is not None:
                r_anc = reduced[idx : idx+len(anchor_vecs)]

            # Plotting on Combined Figure
            plot_simple_2d(axes_comb[i], f"{t['label']} Space (Numeric + Anchor)", 
                           r_mlp, raw_vals_50, r_anc, anchor_names)

            # --- ★変更: 個別のFigureを作成して2パターン保存 ---
            fig_single, ax_single = plt.subplots(figsize=(8, 8))
            
            # 1. タイトルあり版 (スライド用)
            plot_simple_2d(ax_single, f"{t['label']} Space (Numeric + Anchor)", 
                           r_mlp, raw_vals_50, r_anc, anchor_names)
            fig_single.tight_layout()
            fig_single.savefig(os.path.join(epoch_dir, f"2D_{t['label']}_Titled.png"))
            
            # 2. タイトルなし版 (論文用) -> タイトルを空にして再保存
            ax_single.set_title("") 
            fig_single.savefig(os.path.join(epoch_dir, f"2D_{t['label']}_Untitled.png"))
            
            plt.close(fig_single)
            log(f"    -> Saved single plots (Titled & Untitled) for {t['label']}")
            # ----------------------------------------
            # ----------------------------------------

        fig_comb.tight_layout()
        fig_comb.savefig(os.path.join(epoch_dir, "2D_Semantic_Space_Clean_Combined.png"))
        plt.close(fig_comb)
        log("    -> Saved combined plot: 2D_Semantic_Space_Clean_Combined.png")

        # -------------------------------------------------
        # 4. 3D Interactive Visualization (SKIPPED)
        # -------------------------------------------------
        log("\n[4] 3D Interactive Visualization")
        log("-" * 40)
        log("Skipped to save memory.")

    print(f"\n>> All analysis completed. Check: {epoch_dir}")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_value_consistency(values, embeddings, label_name="Magnitude", save_path=None, sample_size=1000,ylim=[-1.1, 1.1]):
    """
    数値の差（例: マグニチュード差）と埋め込みベクトルの類似度の関係を可視化
    
    Args:
        values (torch.Tensor): 数値データ (N, 1) または (N,)
        embeddings (torch.Tensor): モデルが出力した埋め込みベクトル (N, Dim)
        label_name (str): グラフのラベル名 ("Magnitude", "Depth" 等)
    """
    # テンソルの形状調整
    if values.dim() == 1:
        values = values.unsqueeze(1) # (N, 1)
    
    # データ数が多すぎる場合はサンプリング
    N = values.shape[0]
    if N > 1000:
        indices = torch.randperm(N)[:500]
        values = values[indices]
        embeddings = embeddings[indices]
        N = 500
    
    # 1. 数値の差の計算 |v_i - v_j|
    # (N, 1) - (1, N) -> (N, N) -> abs
    diff_matrix = torch.abs(values - values.t())
    
    # 2. コサイン類似度の計算
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(emb_norm, emb_norm.t())
    
    # 3. 上三角成分抽出
    triu_indices = torch.triu_indices(N, N, offset=1)
    differences = diff_matrix[triu_indices[0], triu_indices[1]].detach().cpu().numpy()
    similarities = sim_matrix[triu_indices[0], triu_indices[1]].detach().cpu().numpy()
    
    # 4. 描画用サンプリング
    if len(differences) > sample_size:
        idx = np.random.choice(len(differences), sample_size, replace=False)
        differences = differences[idx]
        similarities = similarities[idx]

    # -----------------------
    # 2. 描画処理 (ここを変更)
    # -----------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(differences, similarities, alpha=0.5, s=15, c='green', edgecolors='none')
    
    if len(differences) > 1:
        m, b = np.polyfit(differences, similarities, 1)
        plt.plot(differences, m*differences + b, color='red', linestyle='--', linewidth=2, label=f'Trend (Slope: {m:.4f})')

    plt.xlabel(f"Difference in {label_name}", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout() # レイアウト調整
    
    # -----------------------
    # 3. 保存処理 (2回保存する)
    # -----------------------
    if save_path:
        root, ext = os.path.splitext(save_path)
        
        # パターンA: タイトルなし (論文用) を先に保存
        # (まだタイトルを設定していない今の状態で保存)
        plt.savefig(f"{root}_untitled{ext}")
        
        # パターンB: タイトルあり (スライド用) を保存
        # (タイトルを追加してから保存)
        plt.title(f"{label_name} Consistency", fontsize=14)
        plt.savefig(f"{root}_titled{ext}", bbox_inches='tight', pad_inches=0.1)
        #plt.savefig(f"{root}_titled{ext}")
        
        plt.close()
        print(f"Saved plots to {root}_untitled{ext} and {root}_titled{ext}")
    else:
        plt.title(f"{label_name} Consistency", fontsize=14)
        plt.show()