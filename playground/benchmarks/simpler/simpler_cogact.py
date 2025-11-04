import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.cogact_exp import (CogACTDataConfig, CogACTExp,
                                     CogACTModelConfig, CogACTTrainerConfig,
                                     CogACTActionConfig, InferenceConfig)


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
class SimplerCogActTrainerConfig(CogACTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/simpler_cogact/{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_simpler_cogact')
    num_train_epochs: int = field(default=5)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)


@dataclass
class SimplerCogActActionConfig(CogACTActionConfig):
    padding_action: bool = field(default=True)

@dataclass
class SimplerCogActDataConfig(CogACTDataConfig):
    dataset_name: str = field(default='simpler_all')
    action_config: SimplerCogActActionConfig = field(default_factory=SimplerCogActActionConfig)

@dataclass
class SimplerCogActModelConfig(CogACTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class SimplerCogActInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/simpler/simpler_cogact')
    port: int = field(default=7891)
        

@dataclass
class SimplerCogActExp(CogACTExp):
    model_config: SimplerCogActModelConfig = field(
        default_factory=SimplerCogActModelConfig)
    trainer_config: SimplerCogActTrainerConfig = field(
        default_factory=SimplerCogActTrainerConfig)
    data_config: SimplerCogActDataConfig = field(
        default_factory=SimplerCogActDataConfig)
    inference_config: SimplerCogActInferenceConfig = field(
        default_factory=SimplerCogActInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = SimplerCogActExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
