# A Study on Methods for Embedding Numerical Data into Language Space in LLM-Driven Point Process Analysis

This repository provides an extended implementation of **TPP-LLM**, a framework that integrates Temporal Point Processes (TPPs) with Large Language Models (LLMs) for event sequence prediction. 

Building upon the [original TPP-LLM framework](https://arxiv.org/abs/2410.02062), this project introduces a novel approach to effectively embed continuous numerical data (such as time, earthquake magnitude, and depth) into the LLM's latent space using **Multi-Layer Perceptrons (MLPs)** and **Semantic Loss**.

<div align="center">
  <img src="images/tpp-llm_semantic_loss.png" alt="TPP-LLM with Semantic Loss Architecture" width="1020"/>
  <p><em>Overview of our enhanced TPP-LLM architecture. Continuous variables (Time, Magnitude, Depth) are encoded via MLPs and aligned to frozen LLM target anchors using MSE Loss.</em></p>
</div>

## 🌟 Features & Novel Contributions

- **Continuous Value Embedding via MLPs**: Directly encodes continuous numerical features (Magnitude, Depth, Time) using specialized MLP encoders, avoiding the precision loss typical in standard tokenization methods.
- **Semantic Loss (`beta_semantic`)**: Introduces a custom MSE-based loss function that aligns the output vectors of the MLPs with the pre-trained word embeddings of their respective concepts. This ensures the LLM intuitively "understands" the numerical scales.
- **Dynamic Token & Prompt Management**: Automatically controls the insertion of structural/delimiter tokens (e.g., `<|time_prefix|>`) and dynamically adjusts the LLM's system prompts based on the selected embedding strategy (Proposed MLP vs. Baseline TPE).
- **Parameter-Efficient Fine-Tuning**: Utilizes Low-Rank Adaptation (LoRA) to efficiently fine-tune the LLM for temporal modeling, reducing computational costs while keeping the base LLM frozen.
- **Comprehensive Evaluation & Visualization**: Includes robust tools to automatically generate:
  - Epoch-by-epoch Normalized Confusion Matrices.
  - Qualitative Sequence Trajectory Plots (True vs. Predicted event types/times).
  - Automated metric extraction and aggregation across multiple experimental seeds.

## 🤗 Pre-trained Models & Datasets

For easy reproducibility and immediate evaluation, we provide the fully processed dataset and our trained model weights on Hugging Face:

- **📊 Dataset (U.S. Earthquake)**: [Download from Hugging Face](https://huggingface.co/datasets/yusuke-oss/TPP-LLM-semantic-loss)
- **🧠 Model Weights**: [Download from Hugging Face](https://huggingface.co/yusuke-oss/TPP-LLM-semantic-loss)

*You can download the dataset and place it directly into the `data/us_earthquake/` directory to skip the preprocessing step. Similarly, downloading the model weights allows you to run evaluations immediately without training from scratch.*

## 📂 Directory Structure

```text
.
├── analysis/
│   └── process_results.py       # Script to extract and summarize metrics across seeds
├── configs/
│   └── tpp_llm_ue.config        # Configuration file for US Earthquake experiments
├── data/
│   └── us_earthquake/           # Processed datasets (train.json, dev.json, test.json)
├── file_sh/
│   └── tpp-llm_us.sh            # Execution script (runs multiple seeds & betas automatically)
├── images/
│   └── tpp-llm_semantic_loss.png # Architecture diagrams
├── notebooks/
│   └── tpp_data.ipynb           # Data download and preprocessing pipeline (USGS API)
├── result/                      # ⚠️ Generated locally: Training logs, metrics, and visualization plots
├── save_model/                  # ⚠️ Generated locally: Saved LoRA adapters (Ignored by Git)
├── save_weight/                 # ⚠️ Generated locally: Merged model weights (Ignored by Git)
├── scripts/
│   └── us_earthquake_semantic_loss/
│       └── train_tpp_llm.py     # Main entry point for training
├── src/
│   └── tpp_llm/us_earthquake_semantic_loss/
│       ├── model.py             # Core model with MLP encoders and Semantic Loss
│       ├── runner.py            # Training loop and evaluation logic
│       ├── data.py              # Dataset loader and global statistics calculator
│       ├── layers.py            # Temporal Positional Encoding (Baseline)
│       ├── utils.py             # Dynamic prompt generation for event sequences
│       ├── common_utils.py      # Reproducibility (Seed) utilities
│       └── analysis.py          # Visualization and quantitative metric tools
├── requirements.txt             # Strict version dependencies for reproducibility
└── .gitignore                   # Keeps heavy model weights and caches out of the repository
```

## 🛠️ Installation

To ensure full reproducibility of the results reported in this project, please install the exact versions of the dependencies specified in the `requirements.txt`. 

1. Clone the repository:
   ```bash
   git clone https://github.com/yusuke-oss/TPP-LLM-semantic-loss.git
   cd TPP-LLM-semantic-loss
   ```

2. We highly recommend using a virtual environment (e.g., Conda):
   ```bash
   conda create -n tpp-llm python=3.13.2
   conda activate tpp-llm
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add the source code to your Python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:src
   ```

## 🚀 Quick Start (Automated Download)

To quickly reproduce our results, you can automatically download the pre-processed dataset and pre-trained model weights directly from Hugging Face. *(Note: Ensure `git-lfs` is installed on your system to download model weights properly).*

Run the following command block in your terminal from the root `TPP-LLM-semantic-loss` directory:

```bash
# ==========================================
# 1. Download and setup the Dataset
# ==========================================
# Clone the dataset repository to a temporary folder
git clone https://huggingface.co/datasets/yusuke-oss/TPP-LLM-semantic-loss hf_dataset

# Create the target directory
mkdir -p data

# Copy the dataset folder preserving the structure
cp -r hf_dataset/data/* data/ 2>/dev/null || true

# Remove the temporary folder
rm -rf hf_dataset

# ==========================================
# 2. Download and setup Pre-trained Weights
# ==========================================
# Clone the model repository to a temporary folder
git clone https://huggingface.co/yusuke-oss/TPP-LLM-semantic-loss hf_model

# Create the target directories
mkdir -p save_model save_weight

# Copy the model and weight files
cp -r hf_model/save_model/* save_model/ 2>/dev/null || true
cp -r hf_model/save_weight/* save_weight/ 2>/dev/null || true

# Remove the temporary folder
rm -rf hf_model
```

## 📊 Dataset Preparation (U.S. Earthquake)

**⚠️ Important Note on Reproducibility:**
We highly recommend downloading the pre-processed dataset directly from **[Hugging Face](https://huggingface.co/datasets/yusuke-oss/TPP-LLM-semantic-loss)** and placing it into the `data/us_earthquake/` directory. Because the USGS Earthquake API is continuously updated, running the notebook today will yield a different raw dataset than the one used for our original experiments and pre-trained models.

The notebook is provided primarily for reference, or for users who wish to collect entirely new, up-to-date earthquake data.

If you still wish to run the data pipeline from scratch:
1. Open `notebooks/tpp_data.ipynb` in Jupyter or VS Code.
2. Run all cells. The notebook will:
   - Download raw CSV data via the USGS Earthquake API.
   - Filter and group earthquakes into discrete sequences based on location and time.
   - Extract continuous values (`magnitude`, `depth`, `time_since_start`).
   - Split the data into 80/10/10 and generate `train.json`, `dev.json`, and `test.json` in the `data/us_earthquake/` directory.

## 🚀 Usage

### ⚙️ Configuring the Execution Script (`tpp-llm_us.sh`)
We provide a bash script to automate training and evaluation across multiple configurations. Before running the script, open `tpp-llm_us.sh` to configure the experimental mode and random seeds:

```bash
# Toggle between the proposed method and the baseline
TEMP_EMB_TYPE="MLP"  # Use "MLP" for Semantic Loss, or "positional" for baseline TPE

# Set the random seeds for robust evaluation
SEEDS=(42 123 501 1000 2025)

# Set the Semantic Loss coefficients (beta) to test (used only if TEMP_EMB_TYPE="MLP")
BETA_SEMANTICS=(10000.0) 
```

The script automatically overrides the `.config` file and builds a clean, hierarchical directory structure to prevent overwriting results:
- **MLP Mode**: `.../mlp/beta_10000.0/seed_42/`
- **Baseline Mode**: `.../positional/seed_42/`

### 🎯 Execution Modes (Choose One)
The behavior of the training script is controlled by **mutually exclusive** execution flags. You must select exactly one mode at the bottom of your configuration file (`configs/tpp_llm_ue.config`):

* `--save_flag`: Trains the model from scratch, evaluates it, and saves the best weights locally.
* `--train_flag`: Trains and evaluates the model, but does *not* save the weights (useful for debugging).
* `--load_flag`: Skips training, loads pre-trained weights from your local paths, and runs evaluation directly.

---

### Case 1: Training from Scratch and Saving
To train the model from scratch and save the best weights locally:

1. Open `configs/tpp_llm_ue.config` and ensure **only** `--save_flag` is present at the bottom.
2. Run the provided bash script:
   ```bash
   bash file_sh/tpp-llm_us.sh
   ```

### Case 2: Evaluating a Pre-trained Model
If you downloaded our pre-trained weights from Hugging Face, you can evaluate them directly on the test set without training.

1. **Place the downloaded weights** into their corresponding local directories. For example, for `seed=42`, `TEMP_EMB_TYPE="MLP"`, and `beta_semantic=10000.0`, ensure the files are extracted like this:
   * `save_model/us_earthquake_semantic_loss/mlp/beta_10000.0/seed_42/` *(Contains LoRA & Tokenizer)*
   * `save_weight/us_earthquake_semantic_loss/mlp/beta_10000.0/seed_42/` *(Contains TPP Prediction Heads)*
   *(Ensure your dataset is also placed in `data/us_earthquake/`)*

2. **Update the config file** (`configs/tpp_llm_ue.config`):
   Replace `--save_flag` with `--load_flag` so the model skips the training loop and loads the weights.

   ```text
   # --- Bottom of configs/tpp_llm_ue.config ---
   --seed=42
   --model_weight_path=save_model/us_earthquake_semantic_loss
   --weight_path=save_weight/us_earthquake_semantic_loss
   --load_flag    # <--- USE ONLY THIS FLAG
   ```

3. **Run the script**:
   ```bash
   bash file_sh/tpp-llm_us.sh
   ```

### 📈 Aggregating Results
After running experiments across multiple seeds, you can automatically aggregate the final test metrics (Mean ± StdDev) by running:

```bash
python analysis/process_results.py
```
**Note:** The script dynamically supports both trained models (automatically extracting the best epoch from validation logs) and pre-trained evaluated models (extracting final test metrics from loaded weights).

This will generate a summary text file (e.g., `mlp_beta_10000.0_summary.txt`) in your results directory containing the compiled statistics.

## 📝 Citation

If you find this code or the original TPP-LLM framework useful in your research, please cite the foundational [paper](https://arxiv.org/abs/2410.02062):

```bibtex
@misc{liu2025tppllmmodelingtemporalpoint,
      title={TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models}, 
      author={Zefang Liu and Yinzhu Quan},
      year={2025},
      eprint={2410.02062},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02062}, 
}
```

## ❓ Questions or Issues

If you have any questions or encounter any issues, please feel free to [submit an issue](https://github.com/yusuke-oss/TPP-LLM-semantic-loss/issues) on our GitHub repository.

## 🙏 Acknowledgment

We sincerely thank the authors of the original **[TPP-LLM](https://github.com/zefang-liu/TPP-LLM)** for their excellent contribution to the field of event sequence prediction. 

## 📜 License

This project is licensed under the [Apache-2.0 License](LICENSE).