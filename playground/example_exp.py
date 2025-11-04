import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.cogact_exp import CogACTActionConfig as _CogACTActionConfig
from dexbotic.exp.cogact_exp import CogACTDataConfig as _CogACTDataConfig
from dexbotic.exp.cogact_exp import CogACTExp as _CogACTExp
from dexbotic.exp.cogact_exp import CogACTModelConfig as _CogACTModelConfig
from dexbotic.exp.cogact_exp import CogACTTrainerConfig as _CogACTTrainerConfig
from dexbotic.exp.cogact_exp import InferenceConfig as _CogACTInferenceConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference'])
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class CogActTrainerConfig(_CogACTTrainerConfig):
    """
    Trainer configuration class - controls training process parameters

    Modification instructions:
    - output_dir: Training output directory, recommended to modify to your local path
    - wandb_project: Weights & Biases project name for experiment tracking
    - num_train_epochs: Number of training epochs, adjust based on dataset size
    """
    # Training output directory - modify to your local path, format: /path/to/your/output/directory
    output_dir: str = field(
        default=f'./user_checkpoints/{datetime.now().strftime("%m%d")}-example')
    # Weights & Biases project name - for experiment tracking and visualization
    wandb_project: str = field(default='dexbotic_example')
    # Number of training epochs - adjust based on dataset size and convergence
    num_train_epochs: int = field(default=30)


@dataclass
class CogActActionConfig(_CogACTActionConfig):
    """
    Action configuration class - controls action-related parameters
    """
    pass


@dataclass
class CogActDataConfig(_CogACTDataConfig):
    """
    Data configuration class - controls dataset and data processing parameters

    Modification instructions:
    - dataset_name: Dataset name, needs to match your dataset
    - action_config: Action configuration instance
    """
    # Dataset name - modify to your dataset name
    dataset_name: str = field(default='libero_goal')
    # Action configuration - contains action-related parameter settings
    action_config: CogActActionConfig = field(default_factory=CogActActionConfig)


@dataclass
class CogActModelConfig(_CogACTModelConfig):
    """
    Model configuration class - controls model-related parameters

    Modification instructions:
    - model_name_or_path: Pre-trained model path or HuggingFace model name
    """
    # Pre-trained model path
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class CogActInferenceConfig(_CogACTInferenceConfig):
    """
    Inference configuration class - controls inference service parameters

    Modification instructions:
    - model_name_or_path: Model path for inference (if empty, uses model from training config)
    - port: Inference service port number
    """
    # Inference model path - if empty, uses model path from training config
    model_name_or_path: str = field(default='')
    # Inference service port number - modify to an available port
    port: int = field(default=7891)


@dataclass
class CogActExp(_CogACTExp):
    """
    Main experiment class - integrates all configurations and executes training or inference

    Modification instructions:
    This class contains instances of all configurations, you can customize parameters by modifying the configuration classes above
    """
    # Model configuration - controls model-related parameters
    model_config: CogActModelConfig = field(default_factory=CogActModelConfig)
    # Trainer configuration - controls training process parameters
    trainer_config: CogActTrainerConfig = field(default_factory=CogActTrainerConfig)
    # Data configuration - controls dataset and data processing parameters
    data_config: CogActDataConfig = field(default_factory=CogActDataConfig)
    # Inference configuration - controls inference service parameters
    inference_config: CogActInferenceConfig = field(
        default_factory=CogActInferenceConfig)


if __name__ == "__main__":
    """
    Main program entry point

    Usage:
    - Training mode: deepspeed playground/example_exp.py
    - Inference mode: python playground/example_exp.py --task inference

    Before running, modify the paths and parameters in each configuration class
    """
    args = parse_args()
    exp = CogActExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
