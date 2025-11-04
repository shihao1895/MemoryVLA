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

 * [**Model Zoo & Benchmark Results**](#Model Zoo & Benchmark Results)
 * [**Install**](#Install)
 * [**Training**](#Training)
 * [**Evaluation**](#Evaluation)
 * [**Deployment in The Real World**](#deployment-in-the-real-world)
 * [**SimplerEnv Installation FAQ**](#SimplerEnv Installation FAQ)
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

| Model      | PickCube | StackCube | PickSingleYCB | PickSingleEGAD | PickClutterYCB | Avg. | CKPT & Logs                                                  |
| ---------- | -------- | --------- | ------------- | -------------- | -------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA+ | 85       | 70        | 55            | 80             | 60             | 70   | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-maniskill2) |



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

We provide pre-trained models for both simulation benchmarks and real-robot settings.
Here we use the Libero pre-trained model as an example.

First, you should download the pre-trained models and put it in the `checkpoints` folder.

```bash
mkdir -p checkpoints/libero
cd checkpoints/libero
git clone https://huggingface.co/Dexmal/libero-db-cogact libero_cogact
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
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_cogact.py --task inference
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

> dexbotic-benchmark also works without docker, see its documentation for further support

**NOTE**: Due to the instability of the benchmark and diffusion process, the performance scores across different iterations can vary significantly. Please evaluate multiple checkpoints and report the best result.



## Training

Before starting training, please follow the instructions in [ModelZoo.md](docs/ModelZoo.md) to set up the pre-trained models, and download the Libero dataset as described in [docs/Data.md](docs/Data.md).

### Training a Model with Provided Data

We use Libero as an example to demonstrate how to train a model with Dexbotic.
The experiment configuration file for this example is located at: [`playground/benchmarks/libero/libero_cogact.py`](playground/benchmarks/libero/libero_cogact.py)

1. Experiment Configuration

```python
# LiberoCogActTrainerConfig
output_dir = [Path to save checkpoints]

```

2. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/libero/libero_cogact.py
```
> We recommend using 8 × NVIDIA A100/H100 GPUs for training.
> If you are using 8 × RTX 4090, please use the configuration file
> `scripts/deepspeed/zero3_offload.json` to reduce GPU memory utilization.

### Training a Model with Your Own Data

1. Prepare Your Own Data

Refer to  [docs/Data.md](docs/Data.md) for detailed instructions on data preparation.
Once created, register your dataset under `dexbotic/data/data_source`.

2. Experiment Configuration

Create a new experiment configuration file (based on [`playground/example_exp.py`](playground/example_exp.py)) and set the required keys:

```python
# CogActTrainerConfig
output_dir = [Path to save checkpoints]

# CogActDataConfig
dataset_name = [Name of your registered dataset]

```

3. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/example_exp.py
```

After training, please refer to the [Evaluation](#evaluation) section above to evaluate your model. Update the `model_name_or_path` in the inference config to your trained checkpoint, and run inference or start the inference server as described.



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

