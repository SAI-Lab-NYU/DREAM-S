<div align="center">
  <h1 align="center">DREAM-S</h1>
  <p align="center">
    <strong>S</strong>peculative Decoding with <strong>T</strong>arget-Aware Refinement and <strong>A</strong>daptive <strong>R</strong>efinement for Multimodal Generation
  </p>
  <p align="center">
    An open-source framework to accelerate Vision Language Model (VLM) inference by up to 3.8x with no quality loss.
  </p>
</div>

<p align="center">
  <a href="https://github.com/SAI-Lab-NYU/DREAM-S/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache_2.0-blue.svg"></a>
  <a href="https://pypi.org/project/dream-s-llm/"><img alt="Version" src="https://img.shields.io/badge/version-1.0.0-brightgreen.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
</p>

---

## 🚀 Overview

DREAM-S is a cutting-edge framework designed to significantly accelerate the inference speed of Vision Language Models (VLMs), such as LLaVA. By employing a novel speculative decoding mechanism, DREAM-S achieves up to a **3.8× speedup** over traditional autoregressive methods without compromising the quality of the output.

The core of DREAM-S is its innovative approach: **S**peculative Decoding with **T**arget-Aware Refinement and **A**daptive **R**efinement for Multimodal Generation. DREAM-S leverages a neural architecture search (NAS) framework with target-aware supernet training to automatically identify both the optimal interaction strategy between the draft and target models, and the most suitable draft model architecture for the underlying hardware platform. This allows the model to generate multiple tokens in parallel and validate them efficiently, leading to substantial gains in performance.

## ✨ Key Features

- **High-Performance Inference:** Up to 3.8× faster inference for Vision Language Models (VLMs) compared to standard autoregressive decoding.
- **Zero Quality Loss:** Maintains the same output distribution as the original model.
- **Multimodal Support:** Fully compatible with multimodal models like LLaVA, SmolVLM, and Pixtral.
- **Hardware-Adaptive NAS:** Automatically searches for the optimal draft model configuration tailored to the target hardware platform.
- **Efficient Training:** Includes scripts for training the auto-regression head using DeepSpeed, with a two-phase progressive training (TPPT) strategy.
- **Adaptive Feature Distillation:** Leverages attention-entropy-guided intermediate feature distillation for improved draft model accuracy.

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SAI-Lab-NYU/DREAM-S.git
    cd dream-s
    ```

2.  **Install dependencies:**
    We recommend creating a virtual environment first.
    ```bash
    pip install -e .
    ```
    *Note: `-e` installs the project in editable mode.*

3.  **Download Model Weights:**
    See the [Model Weights](#-model-weights) section below for links to the available models.

## ⚡ Quick Start

### 1. Inference with Web UI

Run our Gradio-based web interface for an interactive experience. The command automatically handles model allocation across multiple GPUs.

```bash
python -m dream_s.application.webui \
    --ea-model-path [PATH_TO_DREAM_S_WEIGHTS] \
    --base-model-path [PATH_TO_BASE_MODEL]
```

-   `[PATH_TO_DREAM_S_WEIGHTS]`: Path to the downloaded DREAM-S weights (e.g., `./DREAM-S-llava-v1.6-vicuna-7b`).
-   `[PATH_TO_BASE_MODEL]`: Path to the original base model weights (e.g., the original `vicuna-7b-v1.3`).
-   `total-token`: Number of draft tokens. Adjust this based on your hardware for optimal performance. Set to `-1` for auto-configuration.

Once the model is loaded, a URL will be displayed in the terminal.

### 2. Training the Auto-regression Head

First, generate the necessary training data (see `./ge_data` for detailed instructions and generation scripts):
```bash
python -m dream_s.ge_data.allocation_mix665
```

Then, use the following DeepSpeed command to start training:
```bash
cd dream_s/train
deepspeed main_deepspeed.py \
    --deepspeed_config ./ds_config.json \
    --tmpdir [PATH_TO_TRAINING_DATA] \
    --cpdir [PATH_TO_SAVE_CHECKPOINTS] \
    --configpath ./vicuna_7B_config.json
```

### 3. Evaluation

Test the inference speed of DREAM-S on benchmarks like MT-Bench.
```bash
python -m dream_s.evaluation.eval_llava \
    --ea-model-path [PATH_TO_DREAM_S_WEIGHTS] \
    --base-model-path [PATH_TO_BASE_MODEL]
```
This will generate a `.jsonl` file containing the generation results and wall time.

## 📦 Model Weights

| Model | Base Model | Download |
|---|---|---|
| `DREAM-S-llava-v1.6-vicuna-7b` | `vicuna-7b-v1.6` | [🤗 HideonBed12138/DREAM-S-llava-v1.6-vicuna-7b](https://huggingface.co/HideonBed12138/DREAM-S-llava-v1.6-vicuna-7b/edit/main/README.md) |
| `DREAM-S-llava-v1.6-vicuna-13b` | `vicuna-13b-v1.6` | [🤗 HideonBed12138/DREAM-S-llava-v1.6-vicuna-13b](https://huggingface.co/HideonBed12138/DREAM-S-llava-v1.6-vicuna-13b/tree/main) |
<!-- ## 📄 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex

``` -->

## 🙏 Acknowledgements

This project is built upon the incredible work of the open-source community. We are especially grateful to the developers of [Medusa](https://github.com/FasterDecoding/Medusa), [EAGLE](https://github.com/SafeAILab/EAGLE), and [FastChat](https://github.com/lm-sys/FastChat).

## 📜 License

DREAM-S is licensed under the [Apache 2.0 License](LICENSE).