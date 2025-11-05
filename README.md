# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
This is the code for the paper "MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation".

### 🏠[Project Page](https://shihao1895.github.io/MemoryVLA/) | 📑[Paper](https://arxiv.org/abs/2508.19236) | 🤗[Models & Data](https://huggingface.co/collections/shihao1895/memoryvla)

## 🌟 News

- 🔥 [2025-11-5] The code of [MemoryVLA](https://arxiv.org/abs/2508.19236) is released! (Both MemoryVLA and MemoryVLA+)
- 🔥 [2025-10-21] Our VLA codebase [Dexbotic](https://github.com/Dexmal/dexbotic) is released, it now fully integrates MemoryVLA !
- 🔥 [2025-8-26] Our paper [MemoryVLA](https://arxiv.org/abs/2508.19236) is now on arxiv!

## Overview

MemoryVLA is a Cognition-Memory-Action framework for robotic manipulation inspired by human memory systems. It builds a hippocampal-like perceptual-cognitive memory to capture the temporal dependencies essential for current decision-making, enabling long-horizon, temporally aware action generation.

![MemoryVLA Overview](images/intro.png)

We release two versions of the code in separate branches:

- **[MemoryVLA](https://github.com/shihao1895/MemoryVLA/tree/openvla-codebase)**:  built upon the OpenVLA codebase.
- **[MemoryVLA+](https://github.com/shihao1895/MemoryVLA/tree/dexbotic-codebase)**:  built upon our self-developed [Dexbotic](https://dexbotic.com) codebase, which offers higher simulation performance.

## TODO

- [x] Code Release (currently in a preview stage and will be further improved)
  - [x] MemoryVLA (OpenVLA codebase)
  - [x] MemoryVLA+ (Dexbotic codebase)

- [ ] Model Weights Release (We are accelerating the upload process)
  - [ ] MemoryVLA (OpenVLA codebase)
  - [ ] MemoryVLA+ (Dexbotic codebase)
- [ ] Dataset Upload to HuggingFace

## Contents

This is MemoryVLA+ based on Dexbotic codebase, **if you need use OpenVLA codebase**, please use [MemoryVLA](https://github.com/shihao1895/MemoryVLA/tree/openvla-codebase).

 * [**Model Zoo & Benchmark Results**](#Model-Zoo-&-Benchmark-Results)
 * [**Install**](#Install)
 * [**Evaluation**](#Evaluation)
 * [**Training**](#Training)
 * [**FAQ**](#FAQ)
 * [**Citation**](#Citation)

## Model Zoo & Benchmark Results

### Bridge

| Model      | Spoon | Carrot | Cube | Eggplant | Avg. | CKPT & Logs                                                  |
| ---------- | ----- | ------ | ---- | -------- | ---- | ------------------------------------------------------------ |
| MemoryVLA  | 75.0  | 75.0   | 37.5 | 100.0    | 71.9 | [🤗 HF](https://huggingface.co/shihao1895/memvla-bridge)      |
| MemoryVLA+ | 100.0 | 66.7   | 70.8 | 100.0    | 84.4 | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-bridge) |

### LIBERO

| Model            | Spatial | Object | Goal | Long-10 | Long-90 | Avg. | CKPT & Logs                                                  |
| ---------------- | ------- | ------ | ---- | ------- | ------- | ---- | ------------------------------------------------------------ |
| MemoryVLA        | 98.4    | 98.4   | 96.4 | 93.4    | 95.6    | 96.5 | [🤗 Spa](https://huggingface.co/shihao1895/memvla-libero-spatial), [🤗 Obj](https://huggingface.co/shihao1895/memvla-libero-object), [🤗 Goal](https://huggingface.co/shihao1895/memvla-libero-goal), [🤗 100](https://huggingface.co/shihao1895/memvla-libero-100) |
| MemoryVLA+       | 98.2    | 97.8   | 96.4 | 93.6    | 96.2    | 96.5 | [🤗 Spa](https://huggingface.co/shihao1895/memvla-plus-libero-spatial), [🤗 Obj](https://huggingface.co/shihao1895/memvla-plus-libero-object), [🤗 Goal](https://huggingface.co/shihao1895/memvla-plus-libero-goal), [🤗 100](https://huggingface.co/shihao1895/memvla-plus-libero-100) |
| MemoryVLA+ (mix) | 97.2    | 99.2   | 98.4 | 93.2    | 97.2    | 97.1 | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-libero-mix) |

### Fractal-VM

| Model      | Coke Can | Move Near | Open/Close Drawer | Put In Drawer | Avg. | CKPT & Logs                                                  |
| ---------- | -------- | --------- | ----------------- | ------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA  | 90.7     | 88.0      | 84.7              | 47.2          | 77.7 | [🤗 HF](https://huggingface.co/shihao1895/memvla-fractal)     |
| MemoryVLA+ | 92.0     | 91.7      | 71.8              | -             | -    | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-fractal) |

### Fractal-VA

| Model      | Coke Can | Move Near | Open/Close Drawer | Put In Drawer | Avg. | CKPT & Logs                                                  |
| ---------- | -------- | --------- | ----------------- | ------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA  | 80.5     | 78.8      | 53.2              | 58.3          | 67.7 | [🤗 HF](https://huggingface.co/shihao1895/memvla-fractal)     |
| MemoryVLA+ | 83.5     | 81.8      | 63.2              | -             | -    | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-fractal) |

### Maniskill2

| Model      | Pick Cube | Stack Cube | Pick Single YCB | Pick Single EGAD | Pick Clutter YCB | Avg. | CKPT & Logs                                                  |
| ---------- | --------- | ---------- | --------------- | ---------------- | ---------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA+ | 85        | 70         | 55              | 80               | 60               | 70   | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-maniskill2) |

## Install

#### 🐳 Docker (Recommended)


We strongly recommend using the docker as a unified, consistent, and reproducible environment for training and deployment. This approach not only ensures reliability across workflows but also minimizes potential issues arising from CUDA version differences and Python dependency conflicts.

> Please see the [`Dockerfile`](Dockerfile) for details about the image contents.

0. Prerequisites

+ Ubuntu 20.04 or 22.04

+ NVIDIA GPU: RTX 4090 / A100 / H100 (8 GPUs recommended for training; 1 GPU for deployment)

+ NVIDIA Docker installed

1. Step 1: Clone the Repository

```bash
git clone https://github.com/Dexmal/dexbotic.git
```

2. Step 2: Start Docker

```bash
docker run -it --rm --gpus all \
  -v /path/to/dexbotic:/dexbotic \
  dexmal/dexbotic \
  bash
```

3. Step 3: Activate Dexbotic Environment

```bash
cd /dexbotic
conda activate dexbotic
pip install -e .
```

#### Conda Installation

0. Prerequisites

+ Ubuntu 20.04 or 22.04

+ NVIDIA GPU: RTX 4090 / A100 / H100 (8 GPUs recommended for training; 1 GPU for deployment)

+ CUDA 11.8 (tested; other versions may also work)

+ Anaconda

1. Step 1: Clone the Repository

```bash
git clone https://github.com/Dexmal/dexbotic.git
```

2. Step 2: Install Dependencies

```bash
conda create -n dexbotic python=3.10 -y
conda activate dexbotic

pip install torch==2.2.2 torchvision==0.17.2 xformers --index-url https://download.pytorch.org/whl/cu118
cd dexbotic
pip install -e .

# Install FlashAttention
pip install ninja packaging
pip install flash-attn --no-build-isolation
```

## Evaluation

We provide pre-trained models for many benchmarks. 
Here we use the Libero pre-trained model as an example.

First, you should download the pre-trained models and put it in the `checkpoints` folder.

```bash
mkdir -p checkpoints/libero
cd checkpoints/libero
git clone https://huggingface.co/shihao1895/memvla-plus-libero-mix memvla-plus-libero-mix
```

We will demonstrate two ways to evaluate the model. The first is to directly infer one sample, which is the quick way to experience the model. The other is to first deploy the model server and then use a client to get the results, which is more practical in real-world deployment.

### Inference One Sample

```bash
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_cogact.py --task inference_single --image_path test_data/libero_test.png --prompt 'What action should the robot take to put both moka pots on the stove?'
```

You will expect the model to output a set of actions.

### Deploy Mode

1. Start Inference Server

```bash
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_goal_memvla.py --task inference
```

2. Test Model Inference Results

```bash
curl -X POST \
  -F "text=What action should the robot take to put both moka pots on the stove?" \
  -F "image=@test_data/libero_test.png" \
  http://localhost:7891/process_frame

```

3. Test Libero Benchmark with Dexbotic-Benchmark

Set up the [dexbotic-benchmark](https://github.com/Dexmal/dexbotic-benchmark.git) following its instructions and test the deployed model in the LIBERO-GOAL environment.

```bash
cd dexbotic-benchmark
docker run --gpus all --network host -v $(pwd):/workspace \
  dexmal/dexbotic_benchmark \
  bash /workspace/scripts/env_sh/libero.sh /workspace/evaluation/configs/libero/example_libero.yaml
```

> **NOTE**: Due to the instability of the benchmark and diffusion process, the performance scores across different iterations can vary significantly. Please evaluate multiple checkpoints and report the best result.

## Training

### Prepare Data

We adopt the **DexData** format. For more detailed guidelines, please refer to the official documentation: [Dexbotic Data Guide](https://github.com/Dexmal/dexbotic/blob/main/docs/Data.md).

| Dataset            | Link                                                         |
| ------------------ | ------------------------------------------------------------ |
| SimplerEnv-Bridge  | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/simpler) |
| LIBERO             | [🤗 Hugging Face](https://huggingface.co/datasets/shihao1895/libero-dex) |
| SimplerEnv-Fractal | coming soon                                                  |
| ManiSkill2         | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/maniskill2) |

> **NOTE**: For LIBERO, we use 5 suite, include Spatial, Object, Goal, 10, 90.

Please organize the data according to the following directory structure:

```bash
[Your Code Path]
├── dexbotic
├── docs
├── data
│   ├── libero
│   │   ├── libero_10
│   │   │   ├── video
│   │   │   └── jsonl
│   │   ├── libero_90
│   │   ├── libero_goal
│   │   ├── libero_object
│   │   └── libero_spatial
│   ├── maniskill2
│   │   └── video
│   │   └── jsonl
│   └── simpler
│       ├── video
│       └── jsonl
└── ...
```

If you want to use your own custom data, you can create a data source file in the `dexbotic/data/data_source/` directory.
 For detailed instructions, please refer to the [Custom Data Usage Guide](https://dexbotic.com/docs/5. Use Custom Data.html#custom-data-usage-guide).

### Prepare Pretrained Models

| Model              | Description                             | Model Size | Link                                                         |
| ------------------ | --------------------------------------- | ---------- | ------------------------------------------------------------ |
| Dexbotic-Base      | Discrete VLA model (similar to OpenVLA) | 7B         | [🤗 Hugging Face](https://huggingface.co/Dexmal/Dexbotic-Base) |
| Dexbotic-Base-Cont | Continue VLA model (similar to CogACT)  | 7B         | [🤗 Hugging Face](https://huggingface.co/shihao1895/Dexbotic-Base-Cont) |


It is recommended to download the pretrained models into the following folders.

```bash
mkdir checkpoints
cd checkpoints
git clone https://huggingface.co/Dexmal/Dexbotic-Base Dexbotic-Base
git clone https://huggingface.co/shihao1895/Dexbotic-Base-Cont
```

> **Note**: For LIBERO and ManiSkill2, please use the Dexbotic-Base pretrained models, for SimplerEnv-Bridge and Fractal, please use Dexbotic-Base-Cont, in which the action expert is also pretrained.

### Training a Model with Provided Data

We use Libero as an example to demonstrate how to train a model with Dexbotic.
The experiment configuration file for this example is located at: [`playground/benchmarks/libero/libero_spatial_memvla.py`](playground/benchmarks/libero/libero_spatial_memvla.py)

1. Experiment Configuration

```python
# LiberoCogActTrainerConfig
output_dir = [Path to save checkpoints]

```

2. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/libero/libero_spatial_memvla.py
```
> We recommend using 8 × NVIDIA A100/H100 GPUs for training.

### Training a Model with Your Own Data

1. Prepare Your Own Data

Refer to  [Custom Data Usage Guide](https://dexbotic.com/docs/5. Use Custom Data.html#custom-data-usage-guide) for detailed instructions on data preparation.
Once created, register your dataset under `dexbotic/data/data_source`.

2. Experiment Configuration

Create a new experiment configuration file (based on [`playground/example_memvla_exp.py`](playground/example_memvla_exp.py)) and set the required keys:

```python
# CogActTrainerConfig
output_dir = [Path to save checkpoints]

# CogActDataConfig
dataset_name = [Name of your registered dataset]

```

3. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/example_memvla_exp.py
```

After training, please refer to the [Evaluation](#evaluation) section above to evaluate your model. Update the `model_name_or_path` in the inference config to your trained checkpoint, and run inference or start the inference server as described.

## FAQ

If you have any questions about the **Dexbotic** framework, please refer to the official documentation:

- [Dexbotic Docs](https://dexbotic.com/docs)
- [Dexbotic GitHub Repository](https://github.com/Dexmal/dexbotic)

SimplerEnv and ManiSkill may involve several dependency issues during installation. Below are some common troubleshooting tips based on our experience.

**(1) Vulkan / SAPIEN issues**  
Example errors:
ImportError: libvulkan.so.1: cannot open shared object file: No such file or directory
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.

Fix:

```bash
sudo apt install -y libegl1-mesa libgl1-mesa-dev libgles2-mesa-dev
```

and reference:
https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting

> **Note**: Check that the .json files correctly link to the .so file corresponding to your current NVIDIA driver version. Use `nvidia-smi` to check your driver version and locate the correct .so under /usr/lib/x86_64-linux-gnu/.

**(2) OpenGL issues**  
Example errors:
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

Fix:

```bash
sudo apt install -y libgl1 libglib2.0-0 libglx-mesa0 libopengl0 libglu1-mesa mesa-utils
```

**(3) Video recording in SimplerEnv**

```bash
sudo apt install -y ffmpeg
```

(4) **Benchmark Score Fluctuations**

Benchmark scores tend to fluctuate, so we recommend evaluating checkpoints at regular iteration intervals and reporting the best result. Moreover, we have observed that even slight differences in Conda package versions may lead to small variations in the scores.

**(5) Failed to install Flash-Attention**

For detailed installation instructions and troubleshooting, please refer to the official documentation at https://github.com/Dao-AILab/flash-attention.

## Citation

If you find our work helpful in your research, please consider citing [our paper](https://arxiv.org/abs/2508.19236). 

```bibtex
@article{shi2025memoryvla,
  title={MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation},
  author={Shi, Hao and Xie, Bin and Liu, Yingfei and Sun, Lin and Liu, Fengrong and Wang, Tiancai and Zhou, Erjin and Fan, Haoqiang and Zhang, Xiangyu and Huang, Gao},
  journal={arXiv preprint arXiv:2508.19236},
  year={2025}
}

@article{dexbotic,
  title={Dexbotic: Open-Source Vision-Language-Action Toolbox},
  author={Dexbotic Contributors},
  journal={arXiv preprint arXiv:2510.23511},
  year={2025}
}
```

