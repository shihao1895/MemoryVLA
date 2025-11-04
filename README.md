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

 * [**Model Zoo & Simulation Results**](#Model Zoo & Simulation Results)
 * [**Install**](#Install)
 * [**Training**](#Training)
 * [**Evaluation in SimplerEnv**](#Evaluation in SimplerEnv)
 * [**Evaluation in LIBERO**](#Evaluation in LIBERO)
 * [**Deployment in The Real World**](#deployment-in-the-real-world)
 * [**SimplerEnv Installation FAQ**](#SimplerEnv Installation FAQ)
 * [**Citation**](#Citation)



## Model Zoo & Simulation Results

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

The code is built using Python 3.10, and we use PyTorch == 2.2.0 and CUDA == 12.1 (It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:
```bash
conda create --name memvla python=3.10
conda activate memvla

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia cuda-nvcc=12.1 cuda-toolkit=12.1 -y
```
If you need to use the traning code, please also install the [Flash Attention](https://github.com/Dao-AILab/flash-attention), we use flash-attn==2.5.5:

```bash
# Install Flash Attention 2.5.5, this is an example for pytorch2.2-cuda12.1
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Next, clone our repo and install the required packages:

```bash
git clone https://github.com/shihao1895/MemoryVLA
cd MemoryVLA
pip install -e .
```
If you are using an NVIDIA Hopper GPU (e.g., H20) and encounter the error  
**“Floating point exception (core dumped)”**, try reinstalling the specific cuBLAS version below:

```bash
# Fix for NVIDIA H20: "Floating point exception (core dumped)"
pip install nvidia-cublas-cu12==12.4.5.8
```



## Training

1. Prepare training dataset with [RLDS](https://github.com/google-research/rlds) format:

   - [LIBERO](https://libero-project.github.io/intro.html) (including Spatial, Object, Goal, Long-10, Long-90 suites)
   - Bridge from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
   - Fractal from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)

   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   # Download the LIBERO dataset (processed, ~22 GB)
   git clone https://huggingface.co/datasets/shihao1895/libero-rlds
   # Download the Bridge dataset (processed, ~157 GB)
   git clone https://huggingface.co/datasets/shihao1895/bridge-rlds
   # Download the Fractal dataset (processed)
   git clone https://huggingface.co/datasets/shihao1895/fractal-rlds
   ```

2. Download pretrained model, we use [CogACT Pretrained Model](https://huggingface.co/CogACT/CogACT-Large)

   ```bash
   # Download pretrained checkpoint (~31 GB)
   git clone https://huggingface.co/CogACT/CogACT-Large
   ```

3. Train the model on different datasets

   Before training, modify several parameters in the corresponding scripts, such as `hf_token`, `wandb_entity`, checkpoint paths, dataset paths, and log directories.

   We train on **a single node with 8× NVIDIA A100 GPUs**.

   ```bash
   # Train on the Bridge dataset
   bash script/train/bridge/train_bridge.sh
   # Train on the LIBERO-Spatial dataset
   bash script/train/libero/train_libero_spatial.sh
   # Train on the LIBERO-Object dataset
   bash script/train/libero/train_libero_object.sh
   # Train on the LIBERO-Goal dataset
   bash script/train/libero/train_libero_goal.sh
   # Train on the LIBERO-100 dataset
   bash script/train/libero/train_libero_100.sh
   # Train on the Fractal dataset
   bash script/train/fractal/train_fractal.sh
   # Train on real-world data
   bash script/train/real_world/train_real.sh
   ```

   To finetune on your own customized data, please follow the instruction [(rlds_dataset_builder)](https://github.com/kpertsch/rlds_dataset_builder) for converting your data to RLDS format. The actions should be the deltas of end effector ``EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)``. Once your customized data is ready, place the customized data directly under the ``<data_root_dir>/custom_finetuning/1.0.0`` directory. Then set ``vla.data_mix="custom_finetuning"``.



## Evaluation in SimplerEnv

We provide evaluation interfaces and scripts based on [SimplerEnv](https://simpler-env.github.io/).

1. Please follow the installation guide in the [SimplerEnv Repo](https://github.com/simpler-env/SimplerEnv) to set up the simulation environment, and make sure to place the repo under: `./third_libs/SimplerEnv`

2. Evaluation Example.

   ```bash
   # Run evaluation
   bash script/eval/bridge/eval_bridge.sh
   # Summarize results
   python script/eval/bridge/extract_bridge_results.py
   ```

   > **NOTE**: Due to the instability of the SimplerEnv benchmark and diffusion process, the performance scores across different iterations can vary significantly. Please evaluate multiple checkpoints and report the best result.



## Evaluation in LIBERO

We also provide evaluation interfaces and scripts based on [LIBERO](https://libero-project.github.io/intro.html).

1. Please follow the installation guide in the [LIBERO Repo](https://github.com/Lifelong-Robot-Learning/LIBERO) to set up the simulation environment, and make sure to place the repo under: `./third_libs/LIBERO`

2. Evaluation Example.

   ```bash
   # Run evaluation
   bash script/eval/libero/eval_libero.sh
   # Summarize results
   python script/eval/libero/extract_libero_results.py
   ```

   > **NOTE:** The evaluation mechanism here is different from SimplerEnv. The process first loads the model using `develop.py`, then waits for a period before running `evaluation/libero/eval_libero.py` for testing. In addition, since performance may vary across iterations, please evaluate multiple checkpoints and report the best result.



## Deployment in the Real World

To deploy the model on your own robot, first collect corresponding real-world manipulation data (e.g., via teleoperation), and use it to fine-tune the pretrained model.

Next, set up the server and client as shown in [`deploy.py`](deploy.py), and deploy the system on your real robot.

The following command launches the **server**:
```bash
bash script/eval/real_world/deploy.sh
```

The robot acts as the client, and for each request it must send the following three items to obtain the action chunking result. The field episode_first_frame is a string ('True' or 'False') indicating whether the current frame is the first frame of the episode.

```bash
image = request.files['image']
query = request.form['text']
episode_first_frame = request.form['episode_first_frame']
```

This deployment process follows a similar design to [OpenVLA](https://github.com/openvla/openvla) and [CogACT](https://github.com/microsoft/CogACT).



## FAQ

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

**Note**: Check that the .json files correctly link to the .so file corresponding to your current NVIDIA driver version. Use `nvidia-smi` to check your driver version and locate the correct .so under /usr/lib/x86_64-linux-gnu/.

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












<!-- # Dexbotic -->

<p align="center">
  <img src="resources/logo.png" alt="logo" width="280"/>
</p>

## Introduction

Dexbotic aims to provide a one-stop VLA research service for professionals in embodied intelligence field. It offers a codebase that supports multiple mainstream VLA policies simultaneously, allowing users to reproduce various mainstream VLA methods with just a single environment setup based on the pretrained models we provide. Additionally, Dexbotic will continuously update to include more of the latest pre-trained foundation models and cutting-edge VLA models in the industry.

![intro](resources/intro.jpeg)

<details open>
<summary>Main features</summary>

+ **Unified Modular VLA Framework**

  Dexbotic centers around VLA models and is compatible with open-source interfaces of mainstream large language models. It integrates embodied manipulation and navigation, supporting multiple leading embodied manipulation and navigation policies, while also incorporating interfaces for future whole-body control.

+ **Powerful Pre-trained Foundation Models**
  
  For mainstream VLA policies such as Pi0 and CogACT, Dexbotic open-sources several more powerful pre-trained foundation models. These models bring significant performance improvements across various mainstream simulators (like SimplerEnv and CALVIN) as well as real-world robotic tasks.

+ **Experiment-Centric Development Framework**

  The experimental framework of Dexbotic adopts a "layered configuration + factory registration + entry dispatch" approach. Users can easily meet various needs such as modifying configurations, changing models, or adding tasks by simply altering the experimental Exp script. This design aligns with the Open-Closed Principle, allowing for flexibility and extensibility while maintaining stability.

+ **Cloud and Local Training Capabilities**
  
  Dexbotic fully addresses the training needs of users from different universities and enterprises. It supports large-scale cloud-based training platforms such as Alibaba Cloud and Volcano Engine. Additionally, it accommodates local training with consumer-grade GPUs, like RTX 4090 cards.

+ **Diverse Robot Support for Training and Deployment**
  For various mainstream robots, such as UR5, Franka and ALOHA, Dexbotic offers a unified data format for training. It also provides open-source, general-purpose deployment scripts, allowing users to customize their deployments. In the future, Dexbotic will continue to support additional mainstream robotic platforms.

</details>

## 🔥News!

+ [2025-10-20] Dexbotic has been released. Checkout the [paper](docs/Dexbotic_Tech_Report.pdf) and [document](https://dexbotic.com/docs/) for details.

## Open-Source Plan


| Category            | Model/Policy             | Status |
|----------------------|-------------------------|--------|
| **Pretraining Model** | Dexbotic-Base          | ✔️     |
|                      | Dexbotic-CogACT         | ✔️     |
|                      | ├─ Dexbotic-CogACT-SArm | ✔️     |
|                      | └─ Dexbotic-CogACT-HArm | ✔️     |
|                      | Dexbotic-Pi0            | ✔️     |
|                      | Dexbotic-OFT            | ✖️     |
| **Manipulation Policy** | Pi0                  | ✔️     |
|                      | OFT                     | ✔️     |
|                      | CogACT                  | ✔️     |
|                      | MemoryVLA               | ✔️     |
|                      | Pi0.5                   | ✖️     |
| **Navigation Policy**  | MUVLA                 | ✔️     |
|                      | NaVid                   | ✖️     |
|                      | NaVILA                  | ✖️     |
|                      | StreamVLN               | ✖️     |



## Installation

### 🐳 Docker (Recommended)


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

### Conda Installation

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


## Benchmark Results

### Libero

| Model     | Libero-Spatial | Libero-Object | Libero-Goal | Libero-10 | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -                |
| CogACT    | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 | - | - |
| DB-CogACT | 93.8 | 97.8 | 96.2 | 91.8 | 94.9 | [libero_cogact.py](playground/benchmarks/libero/libero_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-cogact) |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | - | - |
| DB-π0 | 97 | 98.2 | 94 | 86.4 | 93.9 | [libero_pi0.py](playground/benchmarks/libero/libero_pi0.py) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-pi0) |
| MemVLA | 98.4 | 98.4 | 96.4 | 93.4 |96.7 | - |
| DB-MemVLA | 97.2 | 99.2 | 98.4 | 93.2 | 97.0 | [libero_memvla.py](https://github.com/Dexmal/dexbotic/blob/main/playground/benchmarks/libero/libero_memvla.py) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-memvla) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-memvla) |

### CALVIN

> Our training and evaluation are conducted under the ABC->D setting.

| Model | 1 | 2 | 3 | 4 | 5 | Average Length | Config | Checkpoint  Link |
| -         | -      | - | -             | -           | -         | -       | -      | -                |
| CogACT | 83.8 | 72.9 | 64 | 55.9 | 48 | 3.246 | - | - |
| DB-CogACT | 93.5 | 86.7 | 80.3 | 76 | 69.8 | 4.063 | [calvin_cogact.py](playground/benchmarks/calvin/calvin_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/calvin-db-cogact) |
| OFT | 89.1 | 79.4 | 67.4 | 59.8 | 51.5 | 3.472 | - | - |
| DB-OFT | 92.8 | 80.7 | 69.2 | 60.2 | 51.1 | 3.540 | [calvin_oft.py](playground/benchmarks/calvin/calvin_oft.py) |  [🤗 HF](https://huggingface.co/Dexmal/calvin-db-oft) |

### SimplerEnv

> Our training uses the Bridge dataset and is tested on the WidowX environment.

| Model | Put Spoon on Towel | Put Carrot on Plate | Stack Green Block on Yellow Block |Put Eggplant in Yellow Basket | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -                |
| CogACT    | 71.7 | 50.8 | 15 |67.5 | 51.25 | - | - |
| DB-CogACT | 87.5 | 65.28 | 29.17 | 95.83 | 69.45 | [simpler_cogact.py](playground/benchmarks/simpler/simpler_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/simpler-db-cogact) |
| OFT | 12.5 | 4.2 | 4.2 | 100 | 30.23 | - | - |
| DB-OFT | 91.67 | 76.39 | 43.06 | 94.44 | 76.39 | [simpler_oft.py](playground/benchmarks/simpler/simpler_oft.py) | [🤗 HF](https://huggingface.co/Dexmal/simpler-db-oft) |
| MemVLA | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 | - | - |
| DB-MemVLA | 100.0 | 66.7 | 70.8 | 100.0 | 84.4 | [simpler_memvla.py](playground/benchmarks/simpler/simpler_memvla.py) | [🤗 HF](https://huggingface.co/Dexmal/simpler-db-memvla) |

### ManiSkill2

| Model | PickCube | StackCube | PickSingleYCB | PickSingleEGAD | PickClutterYCB | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -      | -                |
| CogACT    | 55 | 70 | 30 | 25 | 20 | 40 | - | - |
| DB-CogACT | 90 | 65 | 65 | 40 | 30 | 58 | [maniskill2_cogact.py](playground/benchmarks/maniskill2/maniskill2_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/maniskill2-db-cogact) |
| OFT | 40 | 45 | 5 | 5 | 0 | 21 | - | - |
| DB-OFT | 90 | 75 | 55 | 65 | 30 | 63 | [maniskill2_oft.py](playground/benchmarks/maniskill2/maniskill2_oft.py) | [🤗 HF](https://huggingface.co/Dexmal/maniskill2-db-oft) |
| π0 | 95 | 85 | 55 | 85 | 10 | 66 | - | - |
| DB-π0 | 95 | 85 | 65 | 50 | 30 | 65 | [maniskill2_pi0.py](playground/benchmarks/maniskill2/maniskill2_pi0.py) | [🤗 HF](https://huggingface.co/Dexmal/maniskill2-db-pi0) |

### RoboTwin2.0

> Our training uses the RoboTwin2.0 demo_clean dataset and is tested on the Aloha-AgileX demo_clean environment.

| Model | Adjust Bottle | Grab Roller | Place Empty Cup |Place Phone Stand | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -                |
| CogACT   | 87 | 72 | 11 |5 | 43.8 | - | - |
| DB-CogACT | 99 | 89 | 28 | 18 | 58.5 | [robotwin2_cogact.py](playground/benchmarks/robotwin2/robotwin2_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/robotwin-db-cogact) |

# FAQ

1. Failed to install Flash-Attention: 

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

