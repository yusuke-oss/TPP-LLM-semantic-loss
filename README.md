# TPP-LLM: Modeling Temporal Point Processes with Semantic Alignment for Continuous Data

This repository provides an extended implementation of **TPP-LLM**, a framework that integrates Temporal Point Processes (TPPs) with Large Language Models (LLMs) for event sequence prediction. 

Building upon the [original TPP-LLM framework](https://arxiv.org/abs/2410.02062), this project introduces a novel approach to effectively embed continuous numerical data (such as time, earthquake magnitude, and depth) into the LLM's latent space using **Multi-Layer Perceptrons (MLPs)** and **Semantic Alignment Loss** (Concept Anchor Loss).

<div align="center">
  <img src="images/tpp_llm.png" alt="TPP-LLM Framework" width="1020"/>
</div>

## 🌟 Features & Novel Contributions

- **Continuous Value Embedding via MLPs**: Directly encodes continuous numerical features (Magnitude, Depth, Time) using specialized MLP encoders, avoiding the precision loss typical in standard tokenization methods.
- **Semantic Alignment Loss (`beta_semantic`)**: Introduces a custom MSE-based loss function that aligns the output vectors of the MLPs with the pre-trained word embeddings (Anchors) of their respective concepts (e.g., the word "Magnitude"). This ensures the LLM intuitively "understands" the numerical scales.
- **Parameter-Efficient Fine-Tuning**: Utilizes Low-Rank Adaptation (LoRA) to efficiently fine-tune the LLM for temporal modeling, reducing computational costs while maintaining high performance.
- **Comprehensive Evaluation & Visualization**: Includes a robust suite of analysis tools (`analysis.py` & `aggregate_metrics.py`) to automatically generate:
  - Epoch-by-epoch Normalized Confusion Matrices.
  - Qualitative Sequence Trajectory Plots (True vs. Predicted event types/times).
  - 2D Semantic Space Visualizations (PCA) to verify the distance preservation of learned numerical embeddings.

## 📂 Directory Structure

```text
.
├── configs/
│   └── tpp_llm_ue.config        # Configuration file for US Earthquake experiments
├── data/
│   └── us_earthquake/           # Directory for dataset files (train.json, dev.json, test.json)
├── scripts/
│   └── us_earthquake_semantic_loss/
│       └── train_tpp_llm.py     # Main entry point for training
├── src/
│   └── tpp_llm/us_earthquake_semantic_loss/
│       ├── model.py             # Core model with MLP encoders and Semantic Loss
│       ├── runner.py            # Training loop and evaluation logic
│       ├── data.py              # Dataset loader and global statistics calculator
│       ├── layers.py            # Temporal Positional Encoding
│       ├── utils.py             # Prompt generation for event sequences
│       ├── common_utils.py      # Reproducibility (Seed) utilities
│       ├── analysis.py          # Visualization and quantitative metric tools
│       └── aggregate_metrics.py # Script to aggregate metrics across multiple seeds
├── tpp-llm_us.sh                # Execution script (runs multiple seeds & betas automatically)
└── requirements.txt             # Strict version dependencies for reproducibility
```

## 🛠️ Installation

To ensure full reproducibility of the results reported in this project, please install the exact versions of the dependencies specified in the `requirements.txt`. 

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TPP-LLM.git
   cd TPP-LLM
   ```

2. We highly recommend using a virtual environment (e.g., Conda):
   ```bash
   conda create -n tpp-llm python=3.10 -y
   conda activate tpp-llm
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add the source code to your Python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## 🚀 Usage

### Running Experiments
To train and evaluate the model using the US Earthquake dataset across multiple random seeds, simply execute the provided bash script. This script automatically handles directory creation and iterates over the defined seeds.

```bash
bash tpp-llm_us.sh
```

You can adjust the configuration (e.g., learning rate, LoRA rank, batch size) by editing `configs/tpp_llm_ue.config` or modifying the arguments in the shell script.

### Aggregating Results
After running the experiments across multiple seeds, you can easily aggregate the results (Mean ± StdDev) and generate metric plots by running:

```bash
python src/tpp_llm/us_earthquake_semantic_loss/aggregate_metrics.py
```

## 📊 Datasets

This project includes configurations tailored for the **U.S. Earthquake** dataset. The dataloader automatically calculates global statistics (min/max for magnitude, depth, and time deltas) to normalize inputs for the MLP encoders. 

Please ensure your processed `.json` files are placed within the `data/us_earthquake/` directory. For other datasets (Stack Overflow, Chicago Crime, NYC Taxi, Amazon Reviews), please refer to the original TPP-LLM repository.

## 📝 Citation

If you find this code or the original TPP-LLM framework useful in your research, please cite the foundational [paper](https://arxiv.org/abs/2410.02062):

```bibtex
@article{liu2024tppllmm,
  title={TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models},
  author={Liu, Zefang and Quan, Yinzhu},
  journal={arXiv preprint arXiv:2410.02062},
  year={2024}
}
```

## ❓ Questions or Issues

If you have any questions or encounter any issues, please feel free to [submit an issue](https://github.com/your-username/TPP-LLM/issues) on our GitHub repository.

## 🙏 Acknowledgment

We sincerely thank the authors of the original **[TPP-LLM](https://github.com/zefang-liu/TPP-LLM)** for their excellent contribution to the field of event sequence prediction. We would also like to thank the developers of [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess) for their valuable implementation of TPPs.

## 📜 License

This project is licensed under the [Apache-2.0 License](LICENSE).