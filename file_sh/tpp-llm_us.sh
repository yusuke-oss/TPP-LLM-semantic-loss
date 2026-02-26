#!/bin/bash

# =========================================================
#   TPP-LLM Training Script
# =========================================================

# 実験に使用する乱数シード
SEEDS=(42 123 501 1000 2025)

# =========================================================
#   パス・パラメータ設定 (各配列の要素数は一致させること)
# =========================================================

BASE_DATASET_PATHS=(
  "data/us_earthquake"
)

BASE_RESULT_PATHS=( 
  "result/us_earthquake_semantic_loss"
)

BASE_MODEL_LOAD_PATHS=(
  "save_model/us_earthquake_semantic_loss"
)

BASE_WEIGHT_SAVE_PATHS=(
  "save_weight/us_earthquake_semantic_loss"
)

# 損失関数の係数設定 (Concept/Directional Loss)
BETA_SEMANTICS=(10000.0)

# =========================================================
#   実行ループ
# =========================================================

for seed in "${SEEDS[@]}"; do
  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  # 実験配列のインデックスをループ
  for i in "${!BASE_RESULT_PATHS[@]}"; do
    
    # 変数セットアップ
    DATASET_PATH="${BASE_DATASET_PATHS[$i]}"
    CUR_BETA_S="${BETA_SEMANTICS[$i]}"
    
    # パス生成 (すべてシードごとにディレクトリを分ける)
    CUR_RESULT="${BASE_RESULT_PATHS[$i]}/seed_${seed}"
    CUR_WEIGHT="${BASE_WEIGHT_SAVE_PATHS[$i]}/seed_${seed}"
    CUR_MODEL_LOAD="${BASE_MODEL_LOAD_PATHS[$i]}/seed_${seed}"
    
    # ディレクトリ作成
    mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

    echo "-------------------------------------------------------"
    echo " Running Experiment Index: $i (Seed: $seed)"
    echo " Dataset: $DATASET_PATH"
    echo " Beta Semantic: $CUR_BETA_S"
    echo "-------------------------------------------------------"

    # メイン学習スクリプトの実行
    # temp.configは作成せず、直接引数で上書きして実行する
    python "scripts/us_earthquake_semantic_loss/train_tpp_llm.py" \
      @configs/tpp_llm_ue.config \
      --dataset_path="${DATASET_PATH}" \
      --result_save_path="${CUR_RESULT}" \
      --model_weight_path="${CUR_MODEL_LOAD}" \
      --weight_path="${CUR_WEIGHT}" \
      --seed="${seed}" \
      --beta_semantic="${CUR_BETA_S}"

    echo ">>> Experiment $i (Seed: $seed) Completed."
    echo ""
  done
done