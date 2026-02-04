import numpy as np
from PIL import Image
from typing import Optional
import os
import argparse
import yaml
from argparse import Namespace
import math
from flask import Flask, request, jsonify
import tempfile

import torch

from vla import load_vla
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler

app = Flask(__name__)


class MemVLAService:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10,
        use_ddim: bool = True,
        use_bf16: bool = False,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        args=None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")

        kwargs = vars(args).copy()
        for k in [
            "model_id_or_path", "saved_model_path", "pretrained_checkpoint",
        ]:
            kwargs.pop(k, None)

        self.vla = load_vla(
          model_id_or_path=saved_model_path,
          load_for_training=False,
          **kwargs,
        )
        device = torch.device("cuda")
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        self.vla = self.vla.to(device).to(dtype).eval()
        print(f"Model loaded to {device} with dtype {dtype}.")

        torch.cuda.empty_cache()

        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        self.args = args
        self.reset()

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()

    def step(
        self,
        image: str,
        image_wrist: str = None,
        task_description: str = None,
        episode_first_frame: str = 'False',
        *args, **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: Path to the image file
            task_description: Optional[str], task description
            episode_first_frame: 'True' or 'False', whether the current frame is the first frame of an episode
        Output:
            action: list[float], the ensembled 7-DoFs action of End-effector and gripper

        """

        assert episode_first_frame in ['True', 'False']

        if episode_first_frame == 'True':
            self.reset()

        image: Image.Image = Image.open(image)
        if image_wrist is not None:
            image_wrist: Image.Image = Image.open(image_wrist)
        else:
            image_wrist = None

        # [IMPORTANT!]: Please process the input images here in exactly the same way as the images
        # were processed during finetuning to ensure alignment between inference and training.
        # Make sure, as much as possible, that the gripper is visible in the processed images.
        resized_image = resize_image(image, size=self.image_size)

        # save resized image for debugging
        resized_image.save("resized_image.png")

        if image_wrist is not None:
            resized_image_wrist = resize_image(image_wrist, size=self.image_size)
            resized_image_wrist.save("resized_image_wrist.png")
        else:
            resized_image_wrist = None

        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image,
            image_wrist=resized_image_wrist,
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
        )

        unnormed_actions[:, 6] = np.clip(unnormed_actions[:, 6], -1, 1)
        unnormed_actions[:, 6] = np.where(unnormed_actions[:, 6] < 0.5, 0, 1)

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # Translate the value of the gripper's open/close state to 0 or 1.
            # Please adjust this line according to the control mode of different grippers.
            unnormed_actions[6] = unnormed_actions[6] > 0.5
            action = unnormed_actions.tolist()
        elif self.action_chunking:
            # [IMPORTANT!]: Please modify the code here to output multiple actions at once.
            # The code below only outputs the first action in the chunking.
            # The chunking window size can be adjusted by modifying the 'action_chunking_window' parameter.
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

        print(f"Instruction: {task_description}")
        # print(f"Model path: {self.args.saved_model_path} at port {self.args.port}")
        return action


# [IMPORTANT!]: Please modify the image processing code here to ensure that the input images
# are handled in exactly the same way as during the finetuning phase.
# Make sure, as much as possible, that the gripper is visible in the processed images.
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    #assert h < w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)

    image = scale_and_resize(image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image

# Here the image is first center cropped and then resized back to its original size
# because random crop data augmentation was used during finetuning.
def scale_and_resize(image : Image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5):
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


parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, default="")
parser.add_argument("--unnorm_key", type=str, default='custom_finetuning')
parser.add_argument("--image_size", type=list[int], default=[224, 224])
parser.add_argument("--cfg_scale", type=float, default=1.5)
parser.add_argument("--port", type=int, default=2345)
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--action_ensemble", action="store_true")
parser.add_argument("--action_ensemble_horizon", type=int, default=2)
parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)
parser.add_argument("--action_chunking", action="store_true")
parser.add_argument("--action_chunking_window", type=int, default=None)

args = parser.parse_args()

with open(os.path.join(os.path.dirname(os.path.dirname(args.saved_model_path)), "config.yaml"), "r") as f:
    yaml_args = yaml.safe_load(f) or {}

def deep_update(base: dict, updates: dict):
    """Recursively merge two dictionaries, with updates taking precedence but preserving keys from base."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

cli_args = vars(args)
merged_args = deep_update(yaml_args.copy(), cli_args)

args = Namespace(**merged_args)

inferencer = MemVLAService(
    saved_model_path=args.saved_model_path,
    unnorm_key=args.unnorm_key,
    image_size=args.image_size,
    cfg_scale=args.cfg_scale,
    use_bf16=args.use_bf16,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    action_ensemble_horizon=args.action_ensemble_horizon,
    action_chunking=args.action_chunking,
    action_chunking_window=args.action_chunking_window,
    args=args,
)


@app.route('/process_frame', methods=['POST'])
def inference():
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']

    if 'image_wrist' in request.files:
        image_wrist = request.files['image_wrist']
    else:
        image_wrist = None

    # Check if text is provided
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    query = request.form['text']

    # Check if episode_first_frame is provided
    if 'episode_first_frame' not in request.form:
        return jsonify({'error': 'No episode_first_frame provided'}), 400
    episode_first_frame = request.form['episode_first_frame']

    # Save image to temporary file and resize to expected dimensions
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        image.save(temp_image.name)
        temp_image_path = temp_image.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_image_wrist:
        if image_wrist is not None:
            image_wrist.save(temp_image_wrist.name)
            temp_image_wrist_path = temp_image_wrist.name
        else:
            temp_image_wrist_path = None

    # Construct input query and prepare for inference
    input_query = {
        'task_description': query,
        'episode_first_frame': episode_first_frame,
    }

    # Run inference
    answer = inferencer.step(temp_image_path, temp_image_wrist_path, **input_query)
    print(answer)

    return jsonify({'response': answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=args.port)
