import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.memvla_exp import MemVLAActionConfig as _MemVLAActionConfig
from dexbotic.exp.memvla_exp import MemVLADataConfig as _MemVLADataConfig
from dexbotic.exp.memvla_exp import MemVLAExp as _MemVLAExp
from dexbotic.exp.memvla_exp import MemVLAModelConfig as _MemVLAModelConfig
from dexbotic.exp.memvla_exp import MemVLATrainerConfig as _MemVLATrainerConfig
from dexbotic.exp.memvla_exp import InferenceConfig as _MemVLAInferenceConfig


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
class MemVLATrainerConfig(_MemVLATrainerConfig):
    """
    Trainer configuration class - controls training process parameters

    Modification instructions:
    - output_dir: Training output directory, recommended to modify to your local path
    - wandb_project: Weights & Biases project name for experiment tracking
    - num_train_epochs: Number of training epochs, adjust based on dataset size
    """
    # Training output directory - modify to your local path, format: /path/to/your/output/directory
    output_dir: str = field(
        default=f'./user_checkpoints/{datetime.now().strftime("%m%d")}-memvla-example')
    # Weights & Biases project name - for experiment tracking and visualization
    wandb_project: str = field(default='memvla_example')
    # Number of training epochs - adjust based on dataset size and convergence
    num_train_epochs: int = field(default=50)
    # Batch size per device - adjust based on your GPU memory
    per_device_train_batch_size: int = field(default=32)
    # Save checkpoint every N steps
    save_steps: int = field(default=1000)
    # Maximum number of checkpoints to save
    save_total_limit: int = field(default=50)
    # DataLoader type, now only 'group' is supported, 'stream' and 'parallel_stream' are under development
    dataloader_type: str = field(default='group')
    # Group size for 'group' dataloader type
    group_size: int = field(default=16)


@dataclass
class MemVLAActionConfig(_MemVLAActionConfig):
    """
    Action configuration class - controls action-related parameters
    """
    pass


@dataclass
class MemVLADataConfig(_MemVLADataConfig):
    """
    Data configuration class - controls dataset and data processing parameters

    Modification instructions:
    - dataset_name: Dataset name, needs to match your dataset
    - action_config: Action configuration instance
    """
    # Dataset name - modify to your dataset name
    dataset_name: str = field(default='libero_goal')
    # Action configuration - contains action-related parameter settings
    action_config: MemVLAActionConfig = field(default_factory=MemVLAActionConfig)


@dataclass
class MemVLAModelConfig(_MemVLAModelConfig):
    """
    Model configuration class - controls model-related parameters

    Modification instructions:
    - model_name_or_path: Pre-trained model path or HuggingFace model name
    """
    dataloader_type: str = MemVLATrainerConfig.dataloader_type
    group_size: int = MemVLATrainerConfig.group_size
    # Pre-trained model path
    model_name_or_path: str = field(
        default='')
    # Action model type
    action_model_type: str = field(default='DiT-B')
    # Output dimension of perception compression
    per_token_size: int = field(default=256)
    # Memory length
    mem_length: int = field(default=16)
    # Layer count for memory retrieval
    retrieval_layers: int = field(default=2)
    # Whether to use timestep positional encoding
    use_timestep_pe: bool = field(default=True)
    # Type of memory fusion, 'gate' or 'add'
    fusion_type: str = field(default='gate')
    # Type of memory consolidation, 'fifo' or 'tome'
    consolidate_type: str = field(default='tome')


@dataclass
class MemVLAInferenceConfig(_MemVLAInferenceConfig):
    """
    Inference configuration class - controls inference service parameters

    Modification instructions:
    - model_name_or_path: Model path for inference (if empty, uses model from training config)
    - port: Inference service port number
    """
    # Inference model path - if empty, uses model path from training config
    model_name_or_path: str = field(default='')
    # Inference service port number - modify to an available port
    port: int = field(default=7892)


@dataclass
class MemVLAExp(_MemVLAExp):
    """
    Main experiment class - integrates all configurations and executes training or inference

    Modification instructions:
    This class contains instances of all configurations, you can customize parameters by modifying the configuration classes above
    """
    # Model configuration - controls model-related parameters
    model_config: MemVLAModelConfig = field(default_factory=MemVLAModelConfig)
    # Trainer configuration - controls training process parameters
    trainer_config: MemVLATrainerConfig = field(default_factory=MemVLATrainerConfig)
    # Data configuration - controls dataset and data processing parameters
    data_config: MemVLADataConfig = field(default_factory=MemVLADataConfig)
    # Inference configuration - controls inference service parameters
    inference_config: MemVLAInferenceConfig = field(
        default_factory=MemVLAInferenceConfig)


if __name__ == "__main__":
    """
    Main program entry point
    
    Usage:
    - Training mode: deepspeed playground/example_exp.py
    - Inference mode: python playground/example_exp.py --task inference
    
    Before running, modify the paths and parameters in each configuration class
    """
    args = parse_args()
    exp = MemVLAExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
