import os
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Tuple
import tyro
import math
import json
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import Namespace

import torch
torch.backends.cuda.preferred_linalg_library("magma")
import tensorflow as tf

from robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
)
from mikasa_utils import get_mikasa_eval_env
from vla import load_vla
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler


# prevent a single process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")

if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=512)],
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _scale_and_resize(image: Image.Image, target_size=(224, 224), scale=0.9,
                      margin_w_ratio=0.5, margin_h_ratio=0.5) -> Image.Image:
    # center crop and resize
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image

def resize_image_for_policy(image_tf_tensor, size=(224, 224), shift_to_left=0) -> Image.Image:
    '''
    Converts a TensorFlow tensor or NumPy array representing an image to a PIL Image,
    performs center cropping to a square (with optional left shift), resizes it to the specified
    size, and applies scaling and resizing.
    Args:
        image_tf_tensor: A TensorFlow tensor or NumPy array of shape (H, W, 3) representing the image.
        size: A tuple (width, height) specifying the target size for resizing.
        shift_to_left: An integer specifying the number of pixels to shift the crop to the left.
    Returns:
        A PIL Image after processing.
    '''
    if hasattr(image_tf_tensor, 'numpy'):
        arr = image_tf_tensor.numpy()
    else:
        arr = np.asarray(image_tf_tensor)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)

    im = Image.fromarray(arr)

    # center crop to square (with optional left shift)
    w, h = im.size
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    im = im.crop((left_margin, 0, left_margin + h, h))

    im = im.resize(size, resample=Image.LANCZOS)
    im = _scale_and_resize(im, target_size=size, scale=0.9,
                           margin_w_ratio=0.5, margin_h_ratio=0.5)
    return im

def deep_update(base: dict, updates: dict):
    """Recursively merge two dictionaries, with updates taking precedence but preserving keys from base."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

# =========================
# Task Registry
# =========================
TASK_CONFIGS = {
    # Shell Game
    "SG": {
        "env_id": "ShellGameTouch-v0",
        "language_instruction": "Memorize the position of the cup covering the ball, then pick that cup",
        "num_eval_steps": 90,
    },

    # Intercept
    "IM": {
        "env_id": "InterceptMedium-v0",
        "language_instruction": "Track the ball’s movement, estimate its velocity, then aim the ball at the target",
        "num_eval_steps": 90,
    },

    # Remember Color
    "RC3": {
        "env_id": "RememberColor3-v0",
        "language_instruction": "Remember the color of the cube and then pick the matching one",
        "num_eval_steps": 60,
    },
    "RC5": {
        "env_id": "RememberColor5-v0",
        "language_instruction": "Remember the color of the cube and then pick the matching one",
        "num_eval_steps": 60,
    },
    "RC9": {
        "env_id": "RememberColor9-v0",
        "language_instruction": "Remember the color of the cube and then pick the matching one",
        "num_eval_steps": 60,
    },
}


REQUIRED_ARGS_WITHOUT_TASK = sorted({
    k
    for task_cfg in TASK_CONFIGS.values()
    for k in task_cfg.keys()
})


@dataclass
class Args:
    pretrained_checkpoint: Union[str, Path] = ""
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    task: str = ""   # SG / IM / RC3
    load_wrist: bool = False
    action_scale: float = 1.0

    unnorm_key: Union[str, Path] = "mikasa_dataset"
    cfg_scale: float = 1.5
    use_bf16: bool = False
    use_ddim: bool = True
    num_ddim_steps: int = 10

    action_ensemble: bool = False
    action_ensemble_horizon: int = 7
    adaptive_ensemble_alpha: float = 0.1
    action_chunking: bool = True
    action_chunking_window: Optional[int] = 8

    local_log_dir: str = ""

    # MIKASA Parameters
    env_id: str = ""
    language_instruction: str = ""
    num_eval_steps: int = 90
    num_eval_episodes: int = 100
    env_img_res: int = 256
    record_dir: str = "videos"
    shader: str = "default"
    seed: int = 0
    reset_by_episode_id: bool = True
    info_on_video: bool = False
    save_video: bool = False
    device: str = 'cuda:0'
    camera_width: Optional[int] = 128
    camera_height: Optional[int] = 128
    include_oracle: bool = False
    noop_steps: int = 1
    include_rgb: bool = True
    include_joints: bool = False
    reward_mode: str = 'normalized_dense'
    control_mode: Optional[str] = "pd_ee_delta_pose"
    render_mode: str = "all"
    include_state: bool = False
    sim_backend: str = 'gpu'


class MemVLAWrapper:
    def __init__(self,
                 saved_model_path: str,
                 unnorm_key: str,
                 image_size: Tuple[int, int] = (224, 224),
                 cfg_scale: float = 1.5,
                 use_ddim: bool = True,
                 num_ddim_steps: int = 10,
                 use_bf16: bool = False,
                 action_ensemble: bool = False,
                 adaptive_ensemble_alpha: float = 0.1,
                 action_ensemble_horizon: int = 2,
                 action_chunking: bool = False,
                 action_chunking_window: Optional[int] = None,
                 args=None,
                 ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if hasattr(args, "pretrained_checkpoint"):
            delattr(args, "pretrained_checkpoint")

        self.unnorm_key = unnorm_key

        self.vla = load_vla(
            model_id_or_path=saved_model_path,
            load_for_training=False,
            **vars(args),
        )

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        device = torch.device("cuda")
        self.vla = self.vla.to(device).to(dtype).eval()
        print(f"Model loaded to {device} with dtype {dtype}.")

        if self.vla.use_world_model:
            self.vla.wm_imagine.move_to(device, dtype=dtype)

        torch.cuda.empty_cache()

        self.cfg_scale = float(cfg_scale)

        self.image_size = tuple(image_size)
        self.use_ddim = bool(use_ddim)
        self.num_ddim_steps = int(num_ddim_steps)
        self.action_ensemble = bool(action_ensemble)
        self.adaptive_ensemble_alpha = float(adaptive_ensemble_alpha)
        self.action_ensemble_horizon = int(action_ensemble_horizon)
        self.action_chunking = bool(action_chunking)
        self.action_chunking_window = int(action_chunking_window) if action_chunking_window is not None else None

        if self.action_ensemble and self.action_chunking:
            raise ValueError("action_ensemble 与 action_chunking 不能同时为 True。")

        self.action_ensembler = None
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)

        if hasattr(self.vla, "norm_stats"):
            self.norm_stats = self.vla.norm_stats

    def reset(self):
        if self.action_ensemble and self.action_ensembler is not None:
            self.action_ensembler.reset()

    @torch.no_grad()
    def predict_actions(self,
                        resized_image: Image.Image,
                        instruction: str,
                        resized_image_wrist: Optional[Image.Image] = None,
                        episode_first_frame: str = 'False',
                        ) -> List[np.ndarray]:
        assert episode_first_frame in ['True', 'False']

        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image,
            image_wrist=resized_image_wrist,
            instruction=instruction,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
        )

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            action = unnormed_actions.tolist()
        elif self.action_chunking:
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # Output the first action in the chunking. Can be modified to output multiple actions at once.
            unnormed_actions = unnormed_actions[0]
            action = unnormed_actions.tolist()

        return action


def initialize_model(args):
    cli_args = vars(args).copy()

    cli_args["saved_model_path"] = cli_args.pop("pretrained_checkpoint")

    required_cfg_keys = [
        'saved_model_path',
        'unnorm_key',
        'image_size',
        'cfg_scale',
        'use_ddim',
        'num_ddim_steps',
        'use_bf16',
        'action_ensemble',
        'adaptive_ensemble_alpha',
        'action_ensemble_horizon',
        'action_chunking',
        'action_chunking_window',
        # NOTE: we DO NOT input 'load_wrist' into the config file
    ]

    # pop out args that are not in the config file
    for key in list(cli_args.keys()):
        if key not in required_cfg_keys:
            cli_args.pop(key)

    with open(os.path.join(os.path.dirname(os.path.dirname(cli_args['saved_model_path'])), "config.yaml"), "r") as f:
        yaml_args = yaml.safe_load(f)

    merged_args = deep_update(yaml_args.copy(), cli_args)

    merged_args = Namespace(**merged_args)

    model = MemVLAWrapper(
        saved_model_path=merged_args.saved_model_path,
        unnorm_key=merged_args.unnorm_key,
        image_size=tuple(merged_args.image_size) if isinstance(merged_args.image_size, (list, tuple)) else (224, 224),
        cfg_scale=merged_args.cfg_scale,
        use_ddim=bool(merged_args.use_ddim),
        num_ddim_steps=merged_args.num_ddim_steps if merged_args.num_ddim_steps > 0 else 10,
        use_bf16=merged_args.use_bf16,
        action_ensemble=merged_args.action_ensemble,
        adaptive_ensemble_alpha=merged_args.adaptive_ensemble_alpha,
        action_ensemble_horizon=merged_args.action_ensemble_horizon,
        action_chunking=merged_args.action_chunking,
        action_chunking_window=merged_args.action_chunking_window,
        args=merged_args,
    )

    return model, merged_args.load_wrist


if __name__ == "__main__":

    args = tyro.cli(Args)

    # =========================
    # Resolve task list
    # =========================
    task = (args.task or "").strip()

    if task == "all":
        task_list = list(TASK_CONFIGS.keys())
    elif task in TASK_CONFIGS:
        task_list = [task]
    elif task == "":
        # custom args mode
        missing = []
        for k in REQUIRED_ARGS_WITHOUT_TASK:
            v = getattr(args, k, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                missing.append(k)

        if missing:
            raise ValueError(
                "When --task is not provided, the following arguments are required:\n"
                + "\n".join([f"  --{k}" for k in missing])
            )

        task_list = [args.env_id]

    else:
        raise ValueError(
            f"--task must be 'all' or one of {list(TASK_CONFIGS.keys())}. Got: '{args.task}'"
        )

    set_seed_everywhere(args.seed)

    model, load_wrist = initialize_model(args)

    args.load_wrist = load_wrist

    if args.local_log_dir == "":
        args.local_log_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.pretrained_checkpoint)),
            'eval_mikasa',
            os.path.basename(args.pretrained_checkpoint),
        )
    else:
        logger.info(f"Using custom log dir: {args.local_log_dir}")

    os.makedirs(args.local_log_dir, exist_ok=True)

    results_path = os.path.join(
        args.local_log_dir,
        f"results_{DATE_TIME}.jsonl"
    )

    results_file = open(results_path, "a")

    config_record = {
        "action_chunking_window": args.action_chunking_window,
        "action_scale": args.action_scale,
        "load_wrist": args.load_wrist,
    }

    results_file.write(
        json.dumps(config_record, ensure_ascii=False, indent=2) + "\n"
    )
    results_file.flush()

    logger.info("[Config] ==================================")
    logger.info(f"[Config] action_chunking_window={args.action_chunking_window}")
    logger.info(f"[Config] action_scale={args.action_scale}")
    logger.info(f"[Config] load_wrist={args.load_wrist}")

    task_success_map = {}

    for task_name in (task_list if task_list else [None]):
        logger.info(f"[Task={task_name}] ==================================")

        # =========================
        # Apply task configs or validate manual args
        # =========================
        if task_name in TASK_CONFIGS:
            task_cfg = TASK_CONFIGS[task_name]
            for k, v in task_cfg.items():
                setattr(args, k, v)
            logger.info(f"[Task={task_name}] Running predefined task")
        else:
            logger.info(f"[Task={task_name}] Running custom task")

        logger.info(f"[Task={task_name}] env_id={args.env_id}")
        logger.info(f"[Task={task_name}] lang={args.language_instruction}")
        logger.info(f"[Task={task_name}] steps={args.num_eval_steps}")
        logger.info(f"[Task={task_name}] num_eval={args.num_eval_episodes}")

        # =========================
        # Evaluation
        # =========================
        args.record_dir = os.path.join(
            args.local_log_dir,
            f"{task_name}-{DATE_TIME}",
        )
        os.makedirs(args.record_dir, exist_ok=True)

        env = get_mikasa_eval_env(args)

        eval_metrics = defaultdict(list)

        seeds = list(range(1, args.num_eval_episodes + 1))

        pbar = tqdm(
            range(args.num_eval_episodes),
            desc=f"[Task={task_name}]",
            dynamic_ncols=True,
        )

        for j in pbar:
            if hasattr(model, "reset"):
                model.reset()

            action_queue = deque(
                maxlen=args.action_chunking_window
                if args.action_chunking
                else 1)

            obs, info = env.reset(seed=seeds[j], options={})

            for i in range(args.num_eval_steps):
                observation = {}

                image_primary = tf.convert_to_tensor(obs['image_primary'].cpu().numpy()[0])
                pil_primary = resize_image_for_policy(image_primary, args.image_size)
                observation['image'] = pil_primary

                if args.load_wrist:
                    image_wrist = tf.convert_to_tensor(obs['image_wrist'].cpu().numpy()[0])
                    pil_wrist = resize_image_for_policy(image_wrist, args.image_size)
                    observation['image_wrist'] = pil_wrist

                if len(action_queue) == 0:
                    actions = model.predict_actions(
                        resized_image=observation['image'],
                        resized_image_wrist=observation['image_wrist'] if args.load_wrist else None,
                        instruction=args.language_instruction,
                        episode_first_frame='True' if i == 0 else 'False',
                    )

                    action_queue.extend(actions)

                action = np.array(action_queue.popleft(), dtype=np.float32)

                action[:3] = action[:3] * args.action_scale

                obs, reward, done, trunc, info = env.step(action)

                if "final_info" in info:
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(float(v.item()))
                    break

            if 'success_once' in eval_metrics and len(eval_metrics['success_once']) > 0:
                mean_succ = np.mean(eval_metrics['success_once'])
            else:
                mean_succ = float("nan")

            pbar.set_postfix({
                "ep": j + 1,
                "cur success": f"{mean_succ:.2f}",
            })

        if 'success_once' in eval_metrics and len(eval_metrics['success_once']) > 0:
            mean_succ = np.mean(eval_metrics['success_once'])
        else:
            mean_succ = float("nan")

        task_success_map[task_name] = mean_succ

        result_record = {
            "task": task_name,
            "env_id": args.env_id,
            "lang": args.language_instruction,
            "steps": args.num_eval_steps,
            "num_eval": len(eval_metrics["success_once"]),
            "success": float(mean_succ),
            "action_chunking_window": args.action_chunking_window,
            "action_scale": args.action_scale,
            "load_wrist": args.load_wrist,
        }

        results_file.write(json.dumps(result_record, ensure_ascii=False, indent=2) + "\n")
        results_file.flush()

        logger.info(f"[Task={task_name}] mean_success={mean_succ:.3f} over {len(eval_metrics['success_once'])} evals")

    valid_success = [
        v for v in task_success_map.values()
        if not np.isnan(v)
    ]

    avg_success = float(np.mean(valid_success)) if len(valid_success) > 0 else float("nan")

    summary_record = {
        "task_success": task_success_map,
        "avg_success": avg_success,
        "num_tasks": len(task_success_map),
    }

    results_file.write(
        json.dumps(summary_record, ensure_ascii=False, indent=2) + "\n"
    )
    results_file.flush()

    results_file.close()

    logger.info("[Result] ==================================")
    logger.info(f"[Result] Avg success over {len(valid_success)} tasks: {avg_success:.3f}")
    logger.info(f"[Result] Result file saved to {results_path}")
