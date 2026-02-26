# TPP-LLM-semantic-loss
**TPP-LLM: A Study on Embedding Numerical Data into Language Space in LLM-Driven Point Process Analysis**
*(大規模言語モデル駆動型点過程解析における数値データの言語空間への埋め込み手法の検討)*

Official PyTorch implementation of TPP-LLM with MLP embedding method and semantic alignment loss.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Designs

:star2: **LLM-Based Event Prediction**: Leverages the strong reasoning capabilities of base LLMs (e.g., TinyLlama) via LoRA (Low-Rank Adaptation) fine-tuning to predict complex continuous-time event sequences.

:star2: **MLP Value-to-Vector Encoders**: Replaces standard categorical embeddings with MLP-based encoders to seamlessly project continuous numerical values into the LLM's high-dimensional space.

:star2: **Semantic Alignment Loss**: Introduces a novel semantic loss (`beta_semantic`) that forces the numerical embeddings (Magnitude, Depth, Time) to align with the pre-trained semantic word embeddings (Anchor words) of the LLM.

### Model Architectures

The following images provide visual representations of the TPP-LLM architecture and the Semantic Alignment concept.

![TPP-LLM Architecture]([https://github.com/yusuke-oss/TPP-LLM-semantic-loss/blob/main/images/tpp-llm_semantic_loss.png])
*Figure 1: The architecture of the TPP-LLM model.*

## Results

### Performance Comparison

The enhanced TPP-LLM model with Semantic Alignment Loss demonstrates significant improvements in forecasting accuracy (both Time RMSE and Type Accuracy) compared to baseline models without concept alignment.

![Performance Comparison]([ここにConfusion Matrixなどの結果画像のURLを貼ります])
*Figure 3: Normalized Confusion Matrix and Absolute Time RMSE evaluation.*

### Visualizations

The following visualizations illustrate the effectiveness of our semantic loss. The continuous values correctly form a trajectory towards the target semantic anchors (e.g., "Large Magnitude").

![Visualization 1]([ここに分析コードで出力した2D_Magnitude_Untitled.pngなどのURLを貼ります])
![Visualization 2]([ここに分析コードで出力したqualitative_plotsなどのURLを貼ります])

## Running the Experiments

To run the experiments with tuned hyperparameters, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/](https://github.com/)[あなたのユーザー名]/TPP-LLM-semantic-loss.git
   cd TPP-LLM-semantic-loss
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Place your preprocessed datasets (`train.json`, `dev.json`, `test.json`) in the `./data/us_earthquake` directory.

4. **Run the experiments**:
   Execute the training script with the specified parameters:
   ```bash
   python train_tpp_llm.py \
       --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --dataset_path data/us_earthquake \
       --num_event_types 3 \
       --temporal_emb_type MLP \
       --peft_type lora \
       --beta_semantic 1.0 \
       --num_epochs 10 \
       --train_batch_size 16 \
       --result_save_path results/earthquake_experiment \
       --model_weight_path weights/earthquake_model \
       --save_flag \
       --train_flag
   ```
   ```bash
   bash tpp-llm_us.sh
   ```

## File Structure

The repository is organized as follows:

```plaintext
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── train_tpp_llm.py             # Main script to run the model training and evaluation
├── data/
│   └── us_earthquake/           # Directory for dataset files (train.json, etc.)
├── results/                     # Directory where evaluation logs and plots are saved
├── weights/                     # Directory for trained LoRA model checkpoints
└── src/
    └── tpp_llm/
        └── us_earthquake_semantic_loss/
            ├── analysis.py      # Comprehensive evaluation, PCA visualizations, and metric logs
            ├── data.py          # Data loader and preprocessing (TPPLLMDataset)
            ├── layers.py        # Custom embedding layers (e.g., TimePositionalEncoding)
            ├── model.py         # Core TPP-LLM architecture and Semantic Loss calculation
            └── runner.py        # Training loop, evaluation steps, and gradient accumulation
```

## Explanation of Key Files and Directories

- **`README.md`**: This file, providing an overview of the project and instructions for setup and usage.
- **`train_tpp_llm.py`**: Main entry point to execute the model training and evaluation.
- **`src/.../model.py`**: Contains the `TPPLLMModel` class. It manages the LoRA LLM wrapper, MLP encoders, and computes the `semantic_loss`.
- **`src/.../runner.py`**: The `TPPLLMRunner` class that handles the epoch loops, backpropagation, and logging.
- **`src/.../analysis.py`**: A powerful evaluation suite that generates Confusion Matrices, Sequence Trajectory Plots, and 2D PCA Semantic Space Visualizations.
- **`results/`**: The output directory containing `quantitative_metrics/`, `semantic_visualizations/`, and `consistency_plots/`.

## Acknowledgement

We appreciate the open-source community, especially the developers of [Hugging Face Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) for providing the valuable code base.

## Contact

If you have any questions or concerns, please submit an issue on GitHub.
