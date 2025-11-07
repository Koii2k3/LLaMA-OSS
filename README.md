# LLaMA-OSS: Reasoning-Enhanced Language Model Training

A customized LLaMA Factory pipeline for training and evaluating language models with advanced reasoning capabilities. This project focuses on generating, filtering, and fine-tuning models using reasoning-intensive datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Workflows](#workflows)
  - [1. Model Inference & Data Generation](#1-model-inference--data-generation)
  - [2. Data Filtering Pipeline](#2-data-filtering-pipeline)
  - [3. Model Training](#3-model-training)
  - [4. Model Serving](#4-model-serving)
- [Scripts Reference](#scripts-reference)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project extends [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) with specialized tools for:

- **Reasoning-aware inference**: Generate model responses with configurable reasoning effort levels (low/medium/high)
- **Multi-stage data filtering**: Filter generated responses by label accuracy, token count, and duplication
- **Reasoning quality metrics**: Track and analyze reasoning token usage and output quality
- **Production deployment**: Serve models using vLLM with optimized settings

The primary use case is training models on mathematical reasoning (GSM8K, CompMath) and logical reasoning (LogiQA) tasks.

## Project Structure

```
LLaMA-OSS/
├── README.md                          # This file
└── LLaMA-Factory/                     # Main codebase
    ├── src/
    │   ├── train.py                   # Main training entry point
    │   ├── webui.py                   # Gradio web interface (port 7860)
    │   ├── api.py                     # OpenAI-style API server (port 8000)
    │   └── llamafactory/              # Core library
    ├── scripts/                       # Workflow automation scripts
    │   ├── 0_*.sh                     # Inference & serving scripts
    │   ├── 1_filter_by_label.py       # Filter by prediction accuracy
    │   ├── 2_filter_by_token.py       # Filter by reasoning token count
    │   ├── 3_filter_by_duplication.py # Remove duplicates
    │   ├── lowest_reasoning_samples.py # Analyze reasoning quality
    │   ├── stat_utils/                # Statistical analysis tools
    │   ├── convert_ckpt/              # Model checkpoint converters
    │   └── api_example/               # API usage examples
    ├── data/                          # Training datasets
    │   ├── gsm8k_train_alpaca.jsonl
    │   ├── logiqa_train_alpaca.jsonl
    │   ├── competition_math_train_alpaca.jsonl
    │   └── dataset_info.json
    ├── examples/                      # Training configuration examples
    └── requirements.txt               # Python dependencies
```

## Key Features

### Reasoning-Enhanced Generation
- Support for multiple reasoning effort levels (low/medium/high)
- Concurrent async inference for high throughput
- Token usage tracking (input, output, reasoning tokens)
- Configurable temperature, top_p, and max_tokens

### Multi-Stage Filtering Pipeline
1. **Label-based filtering**: Keep only samples where prediction matches ground truth
2. **Token-based filtering**: Filter by reasoning token thresholds
3. **Duplication removal**: Eliminate redundant samples

### Model Support
- GPT-OSS-20B (primary model)
- All LLaMA Factory supported models (100+ models)
- Support for LoRA, QLoRA, and full fine-tuning

### Production-Ready Serving
- vLLM integration with optimized settings
- FlashInfer sampler support
- Configurable GPU memory utilization
- OpenAI-compatible API endpoints

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: A100/H100 for 20B models)
- 40GB+ GPU memory for GPT-OSS-20B inference

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LLaMA-OSS/LLaMA-Factory
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. Install optional dependencies for vLLM serving:
```bash
pip install -e ".[vllm]"
```

4. Install filtering script dependencies:
```bash
pip install tabulate
```

## Workflows

### 1. Model Inference & Data Generation

Generate synthetic reasoning data using the trained model:

#### Start the model server:
```bash
bash LLaMA-Factory/scripts/0_serve_ofa.sh
```

This starts vLLM server with:
- GPU 1 (configure via `CUDA_VISIBLE_DEVICES`)
- 90% GPU memory utilization
- Max model length: 9120 tokens
- Max batch size: 64 sequences

#### Generate responses:
```bash
bash LLaMA-Factory/scripts/0_gptoss20b_ofa.sh
```

This runs inference on GSM8K dataset with three reasoning effort levels:
- **Low effort**: Faster, simpler reasoning (5 generations per sample)
- **Medium effort**: Balanced reasoning depth
- **High effort**: Most thorough reasoning

**Key parameters:**
- `--batch_size 2048`: Batch size for processing
- `--generations_per_sample 5`: Multiple responses per input
- `--max_new_tokens 8096`: Maximum output length
- `--concurrency 64`: Async request concurrency
- `--temperature 1.0`: Sampling temperature
- `--enable_thinking True`: Enable reasoning mode

**Output files:**
```
gsm8k_train_low_raw.jsonl
gsm8k_train_medium_raw.jsonl
gsm8k_train_high_raw.jsonl
```

### 2. Data Filtering Pipeline

#### Step 1: Filter by Label Accuracy
Keep only samples where the model's prediction matches the ground truth:

```bash
cd LLaMA-Factory/scripts
python 1_filter_by_label.py gsm8k \
  --input-root new_dataset/0_raw \
  --output-root new_dataset/1_fbl \
  --rejected-root new_dataset/1_fbl/rejected
```

**Supported datasets:** `gsm8k`, `logiqa`, `compmath`

**What it does:**
- Extracts answers from `\boxed{...}` or `####` markers
- Compares prediction with label
- Saves matched samples to output, rejected samples separately
- Provides detailed statistics per file

#### Step 2: Filter by Reasoning Token Count
Filter samples based on reasoning quality metrics:

```bash
python 2_filter_by_token.py gsm8k medium \
  --input-root new_dataset/1_fbl \
  --output-root new_dataset/2_fbt
```

**Reasoning effort thresholds:**
- **Low**: min 100 reasoning tokens, max 700 output tokens
- **Medium**: min 300 reasoning tokens, max 700 output tokens
- **High**: min 500 reasoning tokens, max 700 output tokens

**Additional validations:**
- Ensures prediction contains only `\boxed{...}` format
- Filters incomplete or malformed predictions

#### Step 3: Remove Duplicates
```bash
python 3_filter_by_duplication.py gsm8k \
  --input-root new_dataset/2_fbt \
  --output-root new_dataset/3_fbd
```

### 3. Model Training

#### Using the Web UI:
```bash
cd LLaMA-Factory
python src/webui.py
```
Visit `http://localhost:7860` for the Gradio interface.

#### Using the CLI:
```bash
cd LLaMA-Factory
llamafactory-cli train \
  --model_name_or_path /path/to/base/model \
  --dataset your_dataset \
  --template gpt \
  --do_train \
  --output_dir ./output
```

See `LLaMA-Factory/examples/` for more training configurations.

### 4. Model Serving

#### Local deployment:
```bash
cd LLaMA-Factory
python src/api.py
```
API docs available at `http://localhost:8000/docs`

#### Production deployment with vLLM:
```bash
bash LLaMA-Factory/scripts/0_serve_ofa.sh
```

Modify the script to customize:
- GPU selection: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Memory usage: `--gpu-memory-utilization 0.9`
- Context length: `--max-model-len 9120`
- Batch size: `--max-num-seqs 64`

## Scripts Reference

### Inference Scripts
| Script | Purpose |
|--------|---------|
| `0_serve_ofa.sh` | Start vLLM server for GPT-OSS-20B |
| `0_serve_ofa_vast.sh` | Serve with VAST.ai specific settings |
| `0_gptoss20b_ofa.sh` | Generate GSM8K responses (all effort levels) |
| `0_gptoss20b_ofa_compmath.sh` | Generate CompMath responses |
| `0_openai_responses_infer.py` | Core inference script (async OpenAI API) |

### Filtering Scripts
| Script | Purpose |
|--------|---------|
| `1_filter_by_label.py` | Filter by prediction accuracy |
| `2_filter_by_token.py` | Filter by reasoning token thresholds |
| `3_filter_by_duplication.py` | Remove duplicate samples |
| `lowest_reasoning_samples.py` | Analyze samples with lowest reasoning tokens |

### Analysis Scripts
| Script | Purpose |
|--------|---------|
| `stat_utils/cal_lr.py` | Calculate learning rate schedules |
| `stat_utils/cal_mfu.py` | Calculate model FLOPs utilization |
| `stat_utils/cal_flops.py` | Calculate training FLOPs |
| `stat_utils/cal_ppl.py` | Calculate perplexity metrics |
| `stat_utils/length_cdf.py` | Analyze token length distributions |

### Utilities
| Script | Purpose |
|--------|---------|
| `convert_ckpt/tiny_llama4.py` | Convert Tiny LLaMA checkpoints |
| `convert_ckpt/llamafy_qwen.py` | Convert Qwen to LLaMA format |
| `convert_ckpt/llamafy_baichuan2.py` | Convert Baichuan2 to LLaMA format |
| `api_example/test_toolcall.py` | Test tool calling functionality |
| `api_example/test_image.py` | Test multimodal image understanding |

## Datasets

### Available Datasets
The project includes preprocessed training data in Alpaca format:

- **GSM8K**: Grade school math problems (train: 5.8MB, test: 1MB)
- **LogiQA**: Logical reasoning questions (train: 8.8MB, validation: 779KB, test: 784KB)
- **CompMath**: Competition mathematics (train: 13.7MB)

### Dataset Format
```jsonl
{
  "instruction": "Problem statement...",
  "input": "",
  "output": "Solution with reasoning... \\boxed{answer}",
  "reasoning": "Detailed reasoning process...",
  "predict": "\\boxed{answer}",
  "label": "#### answer",
  "reasoning_tokens": 450,
  "output_tokens": 520,
  "input_tokens": 125
}
```

### Adding Custom Datasets
Edit `LLaMA-Factory/data/dataset_info.json`:
```json
{
  "your_dataset": {
    "file_name": "your_dataset.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

## Configuration

### vLLM Server Settings
Edit `LLaMA-Factory/scripts/0_serve_ofa.sh`:

```bash
VLLM_USE_V1=1                          # Use vLLM v1 API
VLLM_USE_FLASHINFER_SAMPLER=1          # Enable FlashInfer sampler
VLLM_LOGGING_LEVEL=DEBUG               # Logging verbosity
CUDA_VISIBLE_DEVICES=1                 # GPU selection
--gpu-memory-utilization 0.9           # GPU memory usage (0.0-1.0)
--max-num-batched-tokens 18240         # Max tokens per batch
--max-model-len 9120                   # Context window size
--max-num-seqs 64                      # Max concurrent sequences
```

### Inference Parameters
Edit inference scripts or modify `0_openai_responses_infer.py`:

```bash
--temperature 1.0                      # Sampling temperature (0.0-2.0)
--top_p 1.0                           # Nucleus sampling threshold
--max_new_tokens 8096                  # Max generation length
--reasoning_effort low|medium|high     # Reasoning depth
--generations_per_sample 5             # Responses per input
--concurrency 64                       # Async concurrency
--batch_size 2048                      # Processing batch size
```

### Training Parameters
Use the Web UI or create a YAML config in `LLaMA-Factory/examples/`:

```yaml
model_name_or_path: /path/to/model
template: gpt
dataset: gsm8k_train
learning_rate: 5.0e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
output_dir: ./output
```

## Advanced Usage

### Analyzing Reasoning Quality
View samples with lowest reasoning tokens:
```bash
python LLaMA-Factory/scripts/lowest_reasoning_samples.py \
  new_dataset/2_fbt/gsm8k/gsm8k_train_medium_fbt.jsonl \
  --top 50
```

View samples with highest reasoning tokens:
```bash
python LLaMA-Factory/scripts/lowest_reasoning_samples.py \
  new_dataset/2_fbt/gsm8k/gsm8k_train_medium_fbt.jsonl \
  --top -50
```

### Custom Filtering Thresholds
Modify `2_filter_by_token.py` to add custom thresholds:
```python
THRESHOLDS = {
    "custom": (800, 700),  # (min_reasoning, max_output)
}
```

### Multi-GPU Inference
Edit serving script for tensor parallelism:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  ...
```

### Checkpoint Conversion
Convert non-LLaMA format models:
```bash
python LLaMA-Factory/scripts/convert_ckpt/llamafy_qwen.py \
  --input_dir /path/to/qwen \
  --output_dir /path/to/llama_format
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

Based on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) by hiyouga.

## Acknowledgments

- **LLaMA Factory**: The core training framework
- **vLLM**: High-performance inference engine
- **OpenAI**: API specification for reasoning models
- **Dataset providers**: GSM8K, LogiQA, CompMath datasets

---

For more detailed information about LLaMA Factory features, see the [official documentation](https://github.com/hiyouga/LLaMA-Factory).
