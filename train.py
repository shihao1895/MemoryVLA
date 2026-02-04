import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml
import draccus

import torch
import torch.distributed as dist

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed

from training import VLAMetrics, get_train_strategy
from conf.vla import VLAConfig, VLARegistry
from vla import load, load_vla
from vla import MemoryVLA
from vla import get_vla_dataset_and_collator
from vla.datasets.rlds.utils.data_utils import save_dataset_statistics


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path("") # Path to dataset directory
    run_root_dir: Path = Path("runs") # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Union[str, Path]] = None # Absolute Path to Checkpoint
    is_resume: bool = True # Whether we are continuing a prior training run, only applicable given pretrained checkpoint
    resume_step: Optional[int] = None # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None # Extra note for logging, Weights & Biases
    save_interval: int = 2500 # Interval for saving checkpoints (in steps)
    image_aug: bool = True # Whether to enable image augmentations
    seed: int = 42 # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token") # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb") # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "" # Name of W&B project to log to (use default!)
    wandb_entity: str = "" # Name of entity to log under

    # Model Parameters
    repeated_diffusion_steps: int = 4 # Repeated steps for training action model (a diffusion model)
    load_all_data_for_training: bool = True # Load all training data
    future_action_window_size: int = 15 # Action chunking, predicting future actions + current action
    action_model_type: str = 'DiT-L' # Action model type, chose from ['DiT-S', 'DiT-B', 'DiT-L']
    use_ema: bool = False # EMA version of action model
    action_dim: int = 7 # Dimension of action space
    dataloader_type: str = "group" # Type of dataloader, chose from ['group', 'stream', 'parallel_stream']
    group_size: int = 16 # Group size for 'group' dataloader
    per_token_size: int = 256 # Token size for perception compression
    mem_length: int = 16 # Memory length
    retrieval_layers: int = 2 # Number of layers of memory retrieval
    use_timestep_pe: bool = True # Whether to use timestep positional encoding
    fusion_type: str = 'gate' # Memory fusion type, chose from ['gate', 'add']
    consolidate_type: str = 'tome' # Memory consolidate type, chose from ['fifo', 'tome']
    update_fused: bool = False # Whether to update fused memory
    load_wrist: bool = False
    load_bi_wrist: bool = False

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("MemoryVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    dist.barrier()
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            step_match = re.search(r"step-(\d+)-", cfg.pretrained_checkpoint)
            epoch_match = re.search(r"epoch-(\d+)-", cfg.pretrained_checkpoint)

            if step_match and epoch_match:
                step = int(step_match.group(1))  # remove leading zeros
                epoch = int(epoch_match.group(1))  # remove leading zeros
                assert step == cfg.resume_step, f"Mismatch in step: {step} != {cfg.resume_step}"
                assert epoch == cfg.resume_epoch, f"Mismatch in epoch: {epoch} != {cfg.resume_epoch}"
            else:
                raise ValueError(f"Checkpoint filename format incorrect: {cfg.pretrained_checkpoint}")

        overwatch.info("Loading VLA Checkpoint")
        if cfg.use_ema:
            overwatch.info("Loading EMA of Diffusion")
        kwargs = vars(cfg)
        model_id_or_path = kwargs.pop("pretrained_checkpoint")
        vla = load_vla(model_id_or_path=model_id_or_path, load_for_training=True, **kwargs)

    else:
        vlm = load(model_id_or_path=cfg.vla.base_vlm, hf_token=cfg.hf_token, load_for_training=True)
        overwatch.info("Creating VLA from Base VLM")
        if cfg.use_ema:
            overwatch.info("Creating EMA for Diffusion")
        vla = MemoryVLA(vlm=vlm, **vars(cfg))
        del vlm
        overwatch.info("DO NOT specify image_resize_strategy, use resize-naive")

    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "full-finetune"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "finetune"  # Frozen vision encoder
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        stage = "align"  # Fine-tuning projector
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        data_root_dir=cfg.data_root_dir,
        data_mix=cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        dataloader_type=cfg.dataloader_type,
        group_size=cfg.group_size,
        load_wrist=cfg.load_wrist,
        load_bi_wrist=cfg.load_bi_wrist,
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    
    dist.barrier()
    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
    if cfg.pretrained_checkpoint is not None and cfg.is_resume:
        train_strategy.load_optimizer_and_scheduler(cfg.pretrained_checkpoint)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        metrics,
        save_interval=cfg.save_interval,
        action_model=True,
        repeated_diffusion_steps=cfg.repeated_diffusion_steps,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
