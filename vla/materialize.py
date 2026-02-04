"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, Union

from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, GroupRLDSDataset, StreamRLDSDataset
from vla.action_tokenizer import ActionTokenizer


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = False,
    future_action_window_size: int = 15,
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    dataloader_type: str = "group",
    group_size: int = 16,
    load_wrist: bool = False,
    load_bi_wrist: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        load_wrist=load_wrist,
        load_bi_wrist=load_bi_wrist,
    )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side,
    )

    # Build RLDS Iterable Dataset
    if dataloader_type == "normal":
        dataset = RLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            load_wrist=load_wrist,
            load_bi_wrist=load_bi_wrist,
        )
    elif dataloader_type == "group":
        assert group_size > 1, "Group size must be greater than 1 for grouped dataset"
        dataset = GroupRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            group_size=group_size,
            load_wrist=load_wrist,
            load_bi_wrist=load_bi_wrist,
        )
    elif dataloader_type == "stream":
        dataset = StreamRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            load_wrist=load_wrist,
            load_bi_wrist=load_bi_wrist,
        )

    else:
        raise NotImplementedError(f"Dataset type {dataloader_type} not implemented.")

    return dataset, action_tokenizer, collator
