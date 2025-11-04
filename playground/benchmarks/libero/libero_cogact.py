import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.cogact_exp import (CogACTDataConfig, CogACTExp,
                                     CogACTModelConfig, CogACTTrainerConfig,
                                     InferenceConfig)


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
class LiberoCogActTrainerConfig(CogACTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/libero_all_cogact/all-{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_libero_cogact')
    num_train_epochs: int = field(default=25)


@dataclass
class LiberoCogActDataConfig(CogACTDataConfig):
    dataset_name: str = field(default='libero_goal+libero_10+libero_spatial+libero_object')

@dataclass
class LiberoCogActModelConfig(CogACTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class LiberoCogActInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/libero/libero_cogact')
    port: int = field(default=7891)
        

@dataclass
class LiberoCogActExp(CogACTExp):
    model_config: LiberoCogActModelConfig = field(
        default_factory=LiberoCogActModelConfig)
    trainer_config: LiberoCogActTrainerConfig = field(
        default_factory=LiberoCogActTrainerConfig)
    data_config: LiberoCogActDataConfig = field(
        default_factory=LiberoCogActDataConfig)
    inference_config: LiberoCogActInferenceConfig = field(
        default_factory=LiberoCogActInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = LiberoCogActExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
