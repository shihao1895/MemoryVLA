import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, cast
from easydict import EasyDict
from flask import jsonify, request
from loguru import logger
from PIL import Image
import megfile

import torch
import transformers
from transformers import BaseImageProcessor, AutoImageProcessor, AutoTokenizer

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.exp.base_exp import (ActionConfig, BaseExp,
                                   ComputeNormActionConfig, DataConfig,
                                   ModelConfig, OptimizerConfig, TrainerConfig,
                                   OPENAI_CLIP_PATH)
from dexbotic.exp.base_exp import InferenceConfig as _InferenceConfig
from dexbotic.exp.utils import NumpyEncoder
from dexbotic.model.memvla.memvla_arch import (MemVLAConfig, MemVLAForCausalLM,
                                               MemVLAModel)
from dexbotic.exp.mem_trainer import DexboticMemTrainer
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token
from dexbotic.tokenization.process import LLMTokenization
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.tokenization import DummyTokenization
from dexbotic.data.dataset.dex_mem_dataset import DexMemDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference',
            'compute_norm_stats'])
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class MemVLAOptimizerConfig(OptimizerConfig):
    """
    MemoryVLA Optimizer Configuration Class - Inherits from Base Optimizer Configuration

    Modifications:
    - base_lr: Base learning rate controlling the step size for model parameter updates
    """
    # Base learning rate
    base_lr: float = field(default=2e-5)


@dataclass
class MemVLATrainerConfig(TrainerConfig):
    """
    MemoryVLA Trainer Configuration Class - Inherits from Base Trainer Configuration

    Modifications:
    - num_train_epochs: Number of training epochs, controlling the total training duration
    - save_steps: Save steps, how many steps to save the model
    - per_device_train_batch_size: Batch size per device
    - gradient_accumulation_steps: Number of gradient accumulation steps, effectively increasing the batch size
    - save_total_limit: Limit the total number of saved models, managing disk space by keeping only the most recent models
    - tune_mm_mlp_adapter: For debugging on a single GPU
    - dataloader_type: Dataloader type, 'group' or 'stream' or 'parallel_stream', now only 'group' is supported
    - group_size: Group size for 'group' dataloader_type
    """
    # Training output directory
    output_dir: Optional[str] = field(default=None)
    # Number of training epochs
    num_train_epochs: int = field(default=50)
    # Batch size per device
    per_device_train_batch_size: int = field(default=32)
    # Number of gradient accumulation steps
    gradient_accumulation_steps: int = field(default=1)
    # Save frequency in steps
    save_steps: int = field(default=2000)
    # Limit the total number of saved models
    save_total_limit: int = field(default=50)
    # For debug in a single gpu
    tune_mm_mlp_adapter: bool = field(default=False)
    # dataloader type, 'group' or 'stream' or 'parallel_stream', now only 'group' is supported
    dataloader_type : str = field(default='group')
    # group size for 'group' dataloader_type
    group_size: int = field(default=16)


@dataclass
class MemVLAActionConfig(ActionConfig):
    """
    MemoryVLA Action Configuration Class - Inherits from Base Action Configuration

    Modifications:
    This class inherits from ActionConfig, using default action processing parameters
    """
    padding_action: bool = field(default=True)
    pass


@dataclass
class ComputeNormActionConfig(ActionConfig):
    """
    Compute Action Normalization Parameter Configuration Class
    """
    def _get_dataset(self, action_process_func, dataset_name_list):
        robot_dataset_list = []
        for dataset_name in dataset_name_list:
            robot_dataset = DexMemDataset(
                data_args=EasyDict(
                    dataset_name=dataset_name,
                    num_images=1,
                    data_keys=['action'],
                    image_processor=AutoImageProcessor.from_pretrained(OPENAI_CLIP_PATH),
                    image_aspect_ratio=None,
                    aug_policy=None),
                tokenization_func=DummyTokenization(),
                action_process_func=action_process_func,
                image_process_func=DummyRGBProcessor())
            robot_dataset_list.append((dataset_name, robot_dataset))
        return robot_dataset_list


@dataclass
class MemVLADataConfig(DataConfig):
    """
    MemoryVLA Data Configuration Class - Inherits from Base Data Configuration

    Modifications:
    - action_config: Action configuration instance, using CogACT-specific action configuration
    """
    action_config: ActionConfig = field(default_factory=MemVLAActionConfig)

    def _build_dataset(self,
                       tokenizer: transformers.PreTrainedTokenizer,
                       chat_template: str,
                       image_processor: BaseImageProcessor) -> DexMemDataset:
        # FIXME: DO NOT USE EASYDICT IN NEXT VERSION
        data_args = EasyDict({
            "dataset_name": self.dataset_name,
            "num_images": self.num_images,
            "data_keys": self.data_keys,
            "images_keys": self.images_keys,
            "aug_policy": self.aug_policy,
            "image_aspect_ratio": self.image_aspect_ratio,
            "image_processor": image_processor,
            "chat_template": chat_template,
        })
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = LLMTokenization(tokenizer, data_args)
        dataset = DexMemDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func
        )
        return dataset


@dataclass
class MemVLAModelConfig(ModelConfig):
    """
    Model Configuration Class - Controls model architecture and initialization parameters

    Modifications:
    - model_name_or_path: Path to the pre-trained model or HuggingFace model name
    - action_model_type: Type of action model, e.g., 'DiT-B'
    - action_dim: Dimension of the action vector, typically 7 (position + rotation + gripper)
    - chunk_size: Size of action chunks for processing
    - freeze_action_head: Whether to freeze the action head during training
    - dataloader_type: Dataloader type, 'group' or 'stream' or 'parallel_stream', now only 'group' is supported
    - group_size: Group size for 'group' dataloader_type
    - per_token_size: Output dimension of perception compression
    - mem_length: Length of the memory
    - retrieval_layers: Number of layers for memory retrieval
    - use_timestep_pe: Whether to use timestep positional encoding
    - fusion_type: Type of memory fusion, 'gate' or 'add'
    - consolidate_type: Type of memory consolidation, 'fifo' or 'tome'
    - update_fused: Whether to update fused representation to memory
    """
    # Pre-trained model path
    model_name_or_path: str = field(default=None)
    # Action model type
    action_model_type: str = field(default='DiT-B')
    # Action dimension, controlling the length of the action vector, typically 7 (position + rotation + gripper)
    action_dim: int = field(default=7)
    # Action chunk size
    chunk_size: int = field(default=16)
    # Whether to freeze action head
    freeze_action_head: bool = field(default=False)
    # Dataloader type
    dataloader_type: str = MemVLATrainerConfig.dataloader_type
    # Group size for 'group' dataloader_type
    group_size: int = MemVLATrainerConfig.group_size
    # Output dimension of perception compression
    per_token_size: int = field(default=256)
    # Memory length
    mem_length: int = field(default=16)
    # Layer count for memory retrieval
    retrieval_layers: int = field(default=2)
    # Whether to use timestep positional encoding
    use_timestep_pe: bool = field(default=True)
    # Type of memory fusion, 'gate' or 'add'
    fusion_type: str = field(default='gate')  # 'gate' or 'add'
    # Type of memory consolidation, 'fifo' or 'tome'
    consolidate_type: str = field(default='tome')  # 'fifo' or 'tome'
    # Whether to update fused representation to memory
    update_fused:str = field(default=True)

    def build_model(self) -> MemVLAForCausalLM:

        if self.from_llm:
            model_config_args = {
                "llm_config": self.model_name_or_path,
                "chat_template": self.chat_template,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "init_llm_weights": True,
                "dataloader_type": self.dataloader_type,
                "group_size": self.group_size,
                'per_token_size': self.per_token_size,
                'mem_length': self.mem_length,
                'retrieval_layers': self.retrieval_layers,
                'use_timestep_pe': self.use_timestep_pe,
                'fusion_type': self.fusion_type,
                'consolidate_type': self.consolidate_type,
                'update_fused': self.update_fused,
            }
            model_config = MemVLAConfig(**model_config_args)
            model = MemVLAForCausalLM(model_config)
        else:
            model_config_args = {
                "model_name_or_path": self.model_name_or_path,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "dataloader_type": self.dataloader_type,
                "group_size": self.group_size,
                'per_token_size': self.per_token_size,
                'mem_length': self.mem_length,
                'retrieval_layers': self.retrieval_layers,
                'use_timestep_pe': self.use_timestep_pe,
                'fusion_type': self.fusion_type,
                'consolidate_type': self.consolidate_type,
                'update_fused': self.update_fused,
            }
            model = MemVLAForCausalLM.from_pretrained(self.model_name_or_path)
            model.model.initialize_model(model_config_args)

        self._freeze_model(model)

        return model

    def _freeze_model(self, model: MemVLAForCausalLM):
        model.model = cast(MemVLAModel, model.model)

        # set requires_grad to True for all parameters
        for param in model.model.parameters():
            param.requires_grad = True

        if self.freeze_llm:
            for param in model.model.backbone.parameters():
                param.requires_grad = False
        if self.freeze_mm_projector:
            for param in model.model.mm_projector_module.parameters():
                param.requires_grad = False
        if self.freeze_mm_vision:
            for param in model.model.mm_vision_module.parameters():
                param.requires_grad = False
        if self.freeze_action_head:
            for param in model.model.action_head_module.parameters():
                param.requires_grad = False


@dataclass
class InferenceConfig(_InferenceConfig):
    """
    Inference Configuration Class - Controls parameters for inference service

    Modifications:
    - model_name_or_path: Path to the inference model
    - port: Port number for the inference service
    - save_image: Whether to save input images for debugging
    - save_image_dir: Directory to save input images
    - norm_stats: Action normalization statistics for denormalizing actions
    """
    # Inference model path
    model_name_or_path: Optional[str] = field(default=None)
    # Inference port
    port: int = field(default=7891)
    # Whether to save input images
    save_image: bool = field(default=False)
    # Directory to save input images
    save_image_dir: str = field(default='./debug_data')
    # Action normalization statistics for denormalizing actions
    norm_stats: Optional[dict] = field(default=None)

    def process_frame(self) -> None:
        results = self._get_response(
            text=request.form.get('text'),
            images=request.files.getlist('image'),
            episode_first_frame=request.form.get('episode_first_frame'),
        )
        return jsonify({'response': results})

    def _get_response(self,
                      text: str,
                      images: list[str],
                      episode_first_frame: str,
                      ) -> str:
        t0 = time.monotonic()
        if len(images) == 1:
            images = [Image.open(images[0]).convert('RGB')]
            image_tensor = self.model.process_images(images).to(dtype=self.model.dtype)
        else:
            images = [Image.open(image).convert('RGB') for image in images]
            image_tensor = self.model.process_images(
                images).to(dtype=self.model.dtype).unsqueeze(0)

        self._save_image(images, text)

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + text)
        conv.append_message(conv.roles[1], ' ')
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to(
            self.model.device)
        logger.debug(f'input_ids: {input_ids}')
        inference_args = {
            'cfg_scale': 1.5,
            'num_ddim_steps': 10,
            'action_norms': self.norm_stats}

        outputs = self.model.inference_action(
            input_ids,
            image_tensor,
            episode_first_frame=episode_first_frame,
            inference_args=inference_args,
        )
        logger.info(f'prompt: <start>{prompt}<end>\naction: {outputs}')
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs

    def _load_model(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = MemVLAForCausalLM.from_pretrained(self.model_name_or_path,
                                                  torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True,
                                                  trust_remote_code=True,
                                                  device_map='auto').to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        logger.info(f"Model loaded successfully")

    def _initialize_inference(self) -> None:
        self._load_model()
        self.prev_prompt = None
        self.timestep = 0
        self.episode = 0

        if self.norm_stats is None:
            primary_file = os.path.join(self.model_name_or_path, "norm_stats.json")
            secondary_file = os.path.join(os.path.dirname(self.model_name_or_path), "norm_stats.json")
            if megfile.smart_exists(primary_file):
                norm_stats_file = primary_file
            elif megfile.smart_exists(secondary_file):
                norm_stats_file = secondary_file
            else:
                raise FileNotFoundError(
                    f"Norm stats file not found in either {primary_file} or {secondary_file}"
                )

            self.norm_stats = self.read_normalization_stats(norm_stats_file)
        elif isinstance(self.norm_stats, str):
            self.norm_stats = self.read_normalization_stats(self.norm_stats)
        logger.info(f"Normalization stats: {self.norm_stats}")


@dataclass
class MemVLAExp(BaseExp):
    """
    Basic Experiment Class - Integrates all configurations and executes training/inference

    Modifications:
    This class contains instances of all configurations, integrating model, optimizer, trainer, data, and
    """
    # Model configuration
    model_config: MemVLAModelConfig = field(default_factory=MemVLAModelConfig)
    # Optimizer configuration
    optimizer_config: MemVLAOptimizerConfig = field(
        default_factory=MemVLAOptimizerConfig)
    # Trainer configuration
    trainer_config: MemVLATrainerConfig = field(default_factory=MemVLATrainerConfig)
    # Data configuration
    data_config: MemVLADataConfig = field(default_factory=MemVLADataConfig)
    # Inference configuration
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        # Step 0: compute norm stats
        self._auto_compute_norm_stats()

        # Step 1: build tokenizer
        tokenizer_kwargs = {
            "model_max_length": self.trainer_config.model_max_length,
            "padding_side": "right",
            "use_fast": True,
        }
        tokenizer = self.tokenizer_config.build_tokenizer(
            self.model_config.model_name_or_path, **tokenizer_kwargs)
        self.tokenizer = tokenizer

        # Step 2: build model
        model = self.model_config.build_model()
        self.model = model
        self.tokenizer = self.tokenizer_config.add_special_tokens(
            self.data_config.action_config.string_format,
            self.data_config.action_config.vocab_size,
            self.tokenizer,
            self.model)
        self.model.config.use_cache = False
        self.model.model.llm.config.use_cache = False

        # Step 3: build dataloader
        train_dataset, data_collator = self.data_config.build_data(
            self.tokenizer, self.model_config.chat_template, self.model.model.mm_vision_module.image_processor, )

        # step 4: build trainer
        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "exp_config": self,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        trainer = DexboticMemTrainer(**trainer_kwargs)
        self.trainer = trainer

        # step 5: save action norm config
        logger.info(
            f"Saving action norm config to {self.trainer_config.output_dir}/norm_stats.json")
        os.makedirs(self.trainer_config.output_dir, exist_ok=True)
        action_norm_config = train_dataset.action_process_func.statistic_mapping
        with open(os.path.join(self.trainer_config.output_dir, "norm_stats.json"), "w") as f:
            json.dump(action_norm_config, f, indent=2, cls=NumpyEncoder)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)


if __name__ == "__main__":
    args = parse_args()
    exp = MemVLAExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'compute_norm_stats':
        exp.compute_norm_stats()
