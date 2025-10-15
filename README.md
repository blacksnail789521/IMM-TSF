# IMM-TSF Benchmark Library

[![arXiv](https://img.shields.io/badge/arXiv-2506.10412-b31b1b.svg)](https://arxiv.org/abs/2506.10412)
[![GitHub Stars](https://img.shields.io/github/stars/blacksnail789521/Time-IMM?style=social)](https://github.com/blacksnail789521/Time-IMM/stargazers)
[![](https://img.shields.io/badge/Project-Website-blue?style=flat)](https://blacksnail789521.github.io/time-imm-project-page/)
[![How to Cite](https://img.shields.io/badge/Cite-bibtex-orange)](#citation)

<p align="center"><sub>
✨ If you find our <em>paper</em> useful, a <strong>star ⭐ on GitHub</strong> helps others discover it and keeps you updated on future releases.
</sub></p>

## Overview

Welcome to the **IMM-TSF** benchmark library, part of the Time-IMM dataset collection for NeurIPS 2025 Datasets & Benchmarks Track. This repository provides tools for loading irregular, multimodal time-series data and running reproducible forecasting benchmarks.

## Repository Structure

```
IMM-TSF/                     
├── data/                   # Place your datasets here (raw and processed)
├── fusions/                # Fusion model implementations
├── layers/                 # Attention, correlation, and transformer layers
├── lib/                    # Utility modules (parsing, evaluation, flows)
├── models/                 # Base forecasting and generative models
├── utils/                  # Helper utilities and scripts
├── go.sh                   # Helper script to launch experiments
├── main.py                 # Entry point for training and evaluation
├── main_all.py             # Run all combinations of datasets and models
├── main_all.sh             # Shell wrapper for batch experiments
├── compute_text_embeddings.py  # Precompute LLM-based embeddings for text
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

> **Note:** All your dataset files should reside under the `data/` folder.

## Data Hierarchy

All datasets must follow the structure below for compatibility:

```
data/
└── {dataset_name}/
    └── processed/
        └── {entity_id}/
            ├── time_series.csv         # Multivariate, irregular time-series data
            ├── text.csv                # Associated unstructured text data
            └── text_embeddings_xxx.pt  # (Optional) Precomputed text embeddings
```

> **Example:** `data/EPA-Air/processed/Los_Angeles/time_series.csv` and `data/EPA-Air/processed/Los_Angeles/text.csv`

Each `{entity_id}` directory represents a unique data unit (e.g., a patient, sensor, or location) and should contain both structured and unstructured views of the data.

## Download Dataset

Please download the datasets from one of the following sources:
* **GitHub Repository:** [https://github.com/blacksnail789521/Time-IMM](https://github.com/blacksnail789521/Time-IMM)
* **Kaggle Dataset:** [https://www.kaggle.com/datasets/blacksnail789521/time-imm](https://www.kaggle.com/datasets/blacksnail789521/time-imm)

After downloading, extract the files and place them under the appropriate folder within the `data/` directory, following the hierarchy above.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/IMM-TSF.git
   cd IMM-TSF
   ```

2. (Recommended) Create and activate a **conda** environment:

   ```bash
   conda create -n immtsf python=3.10
   conda activate immtsf
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> Using `conda` helps manage dependencies and avoid platform-specific conflicts.

## Usage

You can run experiments in multiple ways:

### 1. Direct invocation (`main.py`)

Use command-line arguments for flexible training and evaluation:

```bash
python main.py \
  --data_name GDELT \
  --model_name tPatchGNN \
  --TTF_module TTF_RecAvg \
  --TTF_RecAvg MMF_GR_Add \
  --llm_model_fusion GPT2 \
  --batch_size 32 \
  --use_text_embeddings True \ 
  ...
```

Use `--help` to view all available flags.

### 2. `go.sh` wrapper

Edit default arguments in `main.py` and then launch an experiment:

```bash
. go.sh
```

This script sources predefined parameters and launches your run in one command.


## Precomputing Text Embeddings

To speed up training by avoiding repeated LLM computation, you can precompute embeddings using:

```bash
python compute_text_embeddings.py
```

* This stores `text_embeddings_xxx.pt` under each `processed/{entity_id}/` folder.
* To use the precomputed embeddings, add `--use_text_embeddings True` when running `main.py`.
> **Note:** Please make sure you have at least **24GB total GPU RAM** to run **LLaMA 3.1** and **DeepSeek** models.

## Run All Benchmarks

To benchmark all model–dataset combinations:

```bash
. main_all.sh
```

This automates evaluation across all supported configurations.


## MIMIC Preprocessing

Due to access restrictions, raw MIMIC data must be downloaded manually. Please follow the instruction here:

```
data/MIMIC/mimic_preprocess.ipynb
```

This will generate processed files in:

```
data/MIMIC/processed/{entity_id}/
├── time_series.csv
└── text.csv
```

## Citation

If you find this resource useful, please cite our paper.
```bibtex
@inproceedings{
chang2025timeimm,
title={Time-{IMM}: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series},
author={Ching Chang and Jeehyun Hwang and Yidan Shi and Haixin Wang and Wei Wang and Wen-Chih Peng and Tien-Fu Chen},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=yeqrrn51TL}
}