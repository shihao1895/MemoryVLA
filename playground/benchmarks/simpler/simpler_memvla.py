import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.memvla_exp import (MemVLADataConfig, MemVLAExp,
                                     MemVLAModelConfig, MemVLATrainerConfig,
                                     MemVLAActionConfig, InferenceConfig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference',
            'inference_single'])
    parser.add_argument(
        '--image_path',
        type=str,
        default=None)
    parser.add_argument(
        '--prompt',
        type=str,
        default=None)
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class SimplerMemVLATrainerConfig(MemVLATrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/simpler_memvla/{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_simpler_memvla')
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=32)
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=50)
    dataloader_type: str = field(default='group')
    group_size: int = field(default=16)


@dataclass
class SimplerMemVLAActionConfig(MemVLAActionConfig):
    padding_action: bool = field(default=True)
    pass


@dataclass
class SimplerMemVLADataConfig(MemVLADataConfig):
    dataset_name: str = field(default='simpler_all')
    action_config: SimplerMemVLAActionConfig = field(default_factory=SimplerMemVLAActionConfig)


@dataclass
class SimplerMemVLAModelConfig(MemVLAModelConfig):
    dataloader_type: str = MemVLATrainerConfig.dataloader_type
    group_size: int = MemVLATrainerConfig.group_size
    # You should put the pre-trained model path here
    # NOTE: here, we use a powerful pre-trained model
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base-cont')
    action_model_type: str = field(default='DiT-B')
    per_token_size: int = field(default=256)
    mem_length: int = field(default=16)
    retrieval_layers: int = field(default=2)
    use_timestep_pe: bool = field(default=True)
    fusion_type: str = field(default='gate')
    consolidate_type: str = field(default='tome')


@dataclass
class SimplerMemVLAInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/simpler/simpler_memvla')
    port: int = field(default=7891)


@dataclass
class SimplerMemVLAExp(MemVLAExp):
    model_config: SimplerMemVLAModelConfig = field(
        default_factory=SimplerMemVLAModelConfig)
    trainer_config: SimplerMemVLATrainerConfig = field(
        default_factory=SimplerMemVLATrainerConfig)
    data_config: SimplerMemVLADataConfig = field(
        default_factory=SimplerMemVLADataConfig)
    inference_config: SimplerMemVLAInferenceConfig = field(
        default_factory=SimplerMemVLAInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = SimplerMemVLAExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
