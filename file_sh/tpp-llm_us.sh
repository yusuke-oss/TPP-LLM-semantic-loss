#!/bin/bash

# =========================================================
#   TPP-LLM Training Script
# =========================================================

# Random seeds for experiments
SEEDS=(42 123 501 1000 2025)

# =========================================================
# ★ Experiment Mode Switch
# Simply change to "MLP" or "positional" and all paths will be adjusted automatically
# =========================================================
TEMP_EMB_TYPE="MLP"

# =========================================================
#   Path and Parameter Settings
# =========================================================

BASE_DATASET_PATHS=("data/us_earthquake")
BASE_RESULT_PATHS=("result/us_earthquake_semantic_loss")
BASE_MODEL_LOAD_PATHS=("save_model/us_earthquake_semantic_loss")
BASE_WEIGHT_SAVE_PATHS=("save_weight/us_earthquake_semantic_loss")

# Loss coefficient settings (Semantic Loss)
BETA_SEMANTICS=(10000.0)

# =========================================================
#   Execution Loop
# =========================================================

for seed in "${SEEDS[@]}"; do
  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  for i in "${!BASE_RESULT_PATHS[@]}"; do
    
    # Set up variables
    DATASET_PATH="${BASE_DATASET_PATHS[$i]}"
    CUR_BETA_S="${BETA_SEMANTICS[$i]}"
    
    # ★ Branch directory structure based on the embedding method
    if [ "$TEMP_EMB_TYPE" = "MLP" ]; then
        # Create a beta sub-directory for MLP
        SUB_DIR="mlp/beta_${CUR_BETA_S}/seed_${seed}"
    else
        # Omit the beta sub-directory for positional (TPE)
        SUB_DIR="positional/seed_${seed}"
    fi
    
    # Assemble the final paths
    CUR_RESULT="${BASE_RESULT_PATHS[$i]}/${SUB_DIR}"
    CUR_WEIGHT="${BASE_WEIGHT_SAVE_PATHS[$i]}/${SUB_DIR}"
    CUR_MODEL_LOAD="${BASE_MODEL_LOAD_PATHS[$i]}/${SUB_DIR}"
    
    # Create directories
    mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

    echo "-------------------------------------------------------"
    echo " Running Experiment Index: $i (Seed: $seed)"
    echo " Dataset: $DATASET_PATH"
    echo " Emb Type: $TEMP_EMB_TYPE"
    echo " Output Dir: $CUR_RESULT"
    echo "-------------------------------------------------------"

    # Execute the main training script
    # Arguments are passed directly to override the config file
    python "scripts/us_earthquake_semantic_loss/train_tpp_llm.py" \
      @configs/tpp_llm_ue.config \
      --dataset_path="${DATASET_PATH}" \
      --result_save_path="${CUR_RESULT}" \
      --model_weight_path="${CUR_MODEL_LOAD}" \
      --weight_path="${CUR_WEIGHT}" \
      --beta_semantic="${CUR_BETA_S}" \
      --seed="${seed}" \
      --temporal_emb_type="${TEMP_EMB_TYPE}"
      
  done
done