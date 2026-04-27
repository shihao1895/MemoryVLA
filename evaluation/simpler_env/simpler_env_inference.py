import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import yaml
from argparse import Namespace

os.environ["DISPLAY"] = ""
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

from simpler_env.evaluation.argparse import get_args

from evaluation.simpler_env.maniskill2_evaluator import maniskill2_evaluator
from evaluation.simpler_env import VLAInference


def deep_update(base: dict, updates: dict):
    """Recursively update a dictionary with another dictionary."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_local_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use_bf16", action="store_true")
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    local_args, remaining_argv = parse_local_args(sys.argv[1:])

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_argv]
        args = get_args()
    finally:
        sys.argv = original_argv

    with open(os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'config.yaml'), 'r') as f:
        yaml_args = yaml.safe_load(f) or {}

    cli_args = vars(args)
    if local_args.use_bf16 is not None:
        cli_args["use_bf16"] = local_args.use_bf16
    merged_args = deep_update(yaml_args.copy(), cli_args)
    args = Namespace(**merged_args)
    if not hasattr(args, "use_bf16"):
        args.use_bf16 = False

    exclude_keys = {"pretrained_checkpoint", "use_bf16"}
    filtered_args = {k: v for k, v in vars(args).items() if k not in exclude_keys}

    args.logging_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'eval_simpler')

    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    assert args.ckpt_path is not None
    model = VLAInference(
        saved_model_path=args.ckpt_path,
        cfg_scale=1.5, # cfg from 1.5 to 7 also performs well
        use_bf16=args.use_bf16,
        **filtered_args,
    )

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
