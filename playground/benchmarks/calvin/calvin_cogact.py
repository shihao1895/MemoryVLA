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
class CalvinCogActTrainerConfig(CogACTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/calvin_all_cogact/ABC-{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_calvin_cogact')
    num_train_epochs: int = field(default=5)


@dataclass
class CalvinCogActDataConfig(CogACTDataConfig):
    dataset_name: str = field(default='calvin_ABC')
    auto_norm_method: str = field(default='minmax')


@dataclass
class CalvinCogActModelConfig(CogACTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class CalvinCogActInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/calvin/calvin_cogact')
    port: int = field(default=7891)
        

@dataclass
class CalvinCogActExp(CogACTExp):
    model_config: CalvinCogActModelConfig = field(
        default_factory=CalvinCogActModelConfig)
    trainer_config: CalvinCogActTrainerConfig = field(
        default_factory=CalvinCogActTrainerConfig)
    data_config: CalvinCogActDataConfig = field(
        default_factory=CalvinCogActDataConfig)
    inference_config: CalvinCogActInferenceConfig = field(
        default_factory=CalvinCogActInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = CalvinCogActExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
