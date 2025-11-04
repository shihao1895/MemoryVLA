import os
from typing import Dict, List, Tuple, Iterator
import random
from collections import defaultdict

from torch.utils.data import DataLoader, Sampler

from dexbotic.exp.trainer import DexboticTrainer


EpKey = Tuple[int, int]  # (dataset_idx, file_idx)


def _get_rank_world():
    """
    Return (RANK, WORLD_SIZE) using common env vars.

    Priority:
      - RANK (set by torch.distributed/torchrun/accelerate)
      - fallback to LOCAL_RANK if RANK is absent
      - WORLD_SIZE defaults to 1 when not set

    NOTE:
    - We intentionally avoid importing torch.distributed here to keep this
      utility callable before the process group is initialized.
    """
    try:
        r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        r = 0
    try:
        w = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        w = 1
    return r, w

class CollatePassThrough:
    """
    Collate wrapper that preserves sample ordering metadata (indexes) across workers.

    Behavior:
      - Each sample may carry a 'indexes' tuple of (dataset_idx, file_idx, frame_idx)
      - We pop 'indexes' off samples and attach a consolidated list at batch-level
        as 'out["indexes"]' so the trainer can log exact per-sample ordering.

    NOTE:
    - The actual collate is delegated to 'base_collate'.
    - Do not forward 'indexes' to the model; trainers should pop it before compute.
    """
    def __init__(self, base_collate):
        self.base = base_collate

    def __call__(self, batch):
        tlist = []
        for s in batch:
            t = s.pop("indexes", None)
            if t is not None:
                tlist.append(tuple(int(x) for x in t))
        out = self.base(batch)
        if tlist:
            out["indexes"] = tlist
        return out


class _RankBatchLogger:
    """
    Rank-aware logger that writes one file per rank to avoid interleaved stdout.

    File path:
      ./debug_logs/rank{rank}.log

    Usage:
      - Call log_batch(tlist) with a list of (di, fi, fidx) tuples to append an
        annotated "BATCH" section per training step for this rank only.
    """
    def __init__(self):
        self.rank, self.world = _get_rank_world()
        self.path = os.path.join(os.getcwd(), f"debug_logs/rank{self.rank}.log")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Truncate on first init
        with open(self.path, "w") as f:
            f.write("")

    def log_batch(self, tlist: List[Tuple[int,int,int]]):
        # Append-only to avoid mixing logs across ranks
        with open(self.path, "a") as f:
            f.write(f"=== BATCH [rank {self.rank}/{self.world}] ===\n")
            for di, fi, fidx in tlist:
                f.write(f"{di}\t{fi}\t{fidx}\n")


class _EpisodeScheduleBuilder:
    """
    Build a list of batches (List[List[int]]), each inner list is a batch
    of global sample indices.

    Guarantees:
      - In-episode monotonicity (frames do not go backward within an episode)
      - Episode-level shuffle only; we do not permute frames within an episode
      - Drop-last behavior: only full batches are returned

    Modes:
      - stream: sequentially drain episodes into batches after episode-level shuffle
      - group: per batch, sample E = B // G episodes; from each pick G frames
               (no replacement within an episode), sorted by time; pad last frame if short
      - parallel_stream: interleave B episodes in parallel, one frame per slot per step

    Seeding:
      - All constructions use Random(base_seed + epoch) for reproducibility.

    DDP aware:
      - Episodes are sharded by rank BEFORE building batches via eps[rank::world].
        This ensures no inter-rank duplication.
    """
    def __init__(self, dataset, batch_size: int, group_size: int, base_seed: int = 42, verify: bool = True):
        self.ds = dataset
        self.B = int(batch_size)
        self.G = int(group_size)
        self.base_seed = int(base_seed)
        self.verify = bool(verify)

    def _shard_eps(self, eps, rank: int, world: int):
        """
        Evenly shard episode list across ranks.

        Rationale:
          - Shard by episode boundary rather than by post-built batches to avoid
            cross-rank duplication of frames.
        """
        if world <= 1:
            return eps
        return eps[rank::world]

    def build_group_batches(self, epoch: int, rank: int, world: int) -> List[List[int]]:
        """
        GROUP mode (rank-aware):

        per batch:
          - E = B // G episodes are sampled (without replacement if possible)
          - From each episode, pick G frames without replacement; sorted by time
          - If episode has < G frames, pad with its last frame to reach G
          - Concatenate E chunks to form a batch

        Guarantees:
          - Monotonic order per-episode within the SAME batch
          - Cross-batch monotonicity is NOT guaranteed (by design)

        Drop-last: yes
        """
        assert self.G > 0 and self.B % self.G == 0
        E = self.B // self.G
        rng = random.Random(self.base_seed + int(epoch))
        eps = self._build_episode_map()
        rng.shuffle(eps)
        eps = self._shard_eps(eps, rank, world)
        if not eps:
            return []
        key_to_idxs = {k: v for k, v in eps}
        keys = [k for k, _ in eps]
        total_frames = sum(len(v) for v in key_to_idxs.values())
        num_batches = max(1, total_frames // max(self.B, 1))
        batches = []
        for _ in range(num_batches):
            chosen = rng.sample(keys, E) if len(keys) >= E else rng.choices(keys, k=E)
            batch = []
            for k in chosen:
                frames = key_to_idxs.get(k, [])
                m = len(frames)
                if m == 0:
                    # If an empty episode sneaks in (should not), skip it.
                    continue
                if m >= self.G:
                    pos = sorted(rng.sample(range(m), self.G))
                    take = [frames[i] for i in pos]
                else:
                    # Not enough frames: take all and pad with the last frame.
                    take = frames[:] + [frames[-1]] * (self.G - m)
                batch.extend(take)
            if len(batch) == self.B:
                batches.append(batch)
        if self.verify:
            self._assert_monotonic_within_batch(batches)
        return batches

    def _build_episode_map(self) -> List[Tuple[EpKey, List[int]]]:
        """
        Build [(EpKey, [global_indices in ascending frame order])].

        Notes:
          - Respects 'predict_length' by trimming tail frames if necessary
          - global_index is expected to contain (di, fi, fidx)
        """
        predict_len = int(getattr(self.ds.action_process_func, "predict_length", 0) or 0)
        buckets: Dict[EpKey, List[Tuple[int, int]]] = defaultdict(list)
        for gidx, (di, fi, fidx) in enumerate(self.ds.global_index):
            buckets[(int(di), int(fi))].append((int(fidx), int(gidx)))

        eps: List[Tuple[EpKey, List[int]]] = []
        for k, pairs in buckets.items():
            pairs.sort(key=lambda x: x[0])  # ascending by frame idx
            if predict_len > 0:
                eff = max(len(pairs) - predict_len, 0)
                pairs = pairs[:eff]
            if not pairs:
                continue
            idxs = [g for _, g in pairs]
            eps.append((k, idxs))
        return eps

    def _assert_monotonic(self, batches: List[List[int]]) -> None:
        """
        Validate that frames never go backwards within each episode across
        the whole sequence of returned batches (global monotonicity).
        """
        last_fidx: Dict[EpKey, int] = {}
        for batch in batches:
            for g in batch:
                di, fi, fidx = self.ds.global_index[g]
                key = (int(di), int(fi))
                pv = last_fidx.get(key, -1)
                if int(fidx) < pv:
                    raise RuntimeError(
                        f"Non-monotonic within episode {key}: {fidx} < {pv} (global_idx={g})"
                    )
                last_fidx[key] = int(fidx)

    def _assert_monotonic_within_batch(self, batches: List[List[int]]) -> None:
        """
        Validate that within a SINGLE batch, frames from the same episode never
        go backwards (used by GROUP mode).
        """
        for batch in batches:
            last_fidx: Dict[EpKey, int] = {}
            for g in batch:
                di, fi, fidx = self.ds.global_index[g]
                key = (int(di), int(fi))
                prev = last_fidx.get(key, -1)
                if int(fidx) < prev:
                    raise RuntimeError(
                        f"Non-monotonic within one batch for episode {key}: {fidx} < {prev} (global_idx={g})"
                    )
                last_fidx[key] = int(fidx)


class EpisodeBatchSampler(Sampler[List[int]]):
    """
    Batch-level sampler for DataLoader(batch_sampler=...).

    Highlights:
      - Single code path for single- and multi-GPU: always build batches
        "for_rank"; when WORLD_SIZE=1, sharding is effectively no-op.
      - Epoch-wise reshuffle/rebuild is handled internally:
          * DataLoader creates a fresh iterator each epoch
          * We bump an internal 'epoch' counter inside __iter__()
          * Seed = base_seed + epoch ensures deterministic reshuffle per epoch
      - set_epoch remains available if an external trainer prefers to call it.

    Returns:
      - Iterator over lists of global indices, each list size == batch_size
    """
    def __init__(
        self,
        dataset,
        mode: str,
        batch_size: int,
        group_size: int,
        seed: int = 42,
        verify: bool = True,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.mode = str(mode).lower()
        assert self.mode in {"stream", "group", "parallel_stream"}
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.base_seed = int(seed)
        self.verify = bool(verify)

        self.rank, self.world = _get_rank_world()
        self.builder = _EpisodeScheduleBuilder(
            dataset, batch_size, group_size, base_seed=seed, verify=verify
        )

        self.epoch = 0       # current epoch (internal)
        self._passes = 0     # number of times __iter__ has been called
        self._batches: List[List[int]] = []
        self._rebuild()

    def _build_for_rank(self, epoch: int) -> List[List[int]]:
        """
        Dispatch to the correct builder with rank-aware episode sharding.
        """
        if self.mode == "group":
            return self.builder.build_group_batches(epoch, self.rank, self.world)
        elif self.mode == "stream" or self.mode == "parallel_stream":
            raise NotImplementedError(f"{self.mode} is not implemented yet.")
        else:
            raise ValueError(self.mode)

    def _rebuild(self) -> None:
        """
        Build batches for the current (self.epoch).
        """
        self._batches = self._build_for_rank(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        """
        Optional external control of epoch. Not required if relying on
        DataLoader's per-epoch iterator creation (we bump 'epoch' in __iter__).
        """
        self.epoch = int(epoch)
        self._rebuild()

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Called by DataLoader at the beginning of each epoch.
        We advance the internal epoch counter and rebuild batches deterministically.
        """
        self.epoch = self._passes
        self._batches = self._build_for_rank(self.epoch)
        self._passes += 1
        return iter(self._batches)


class DexboticMemTrainer(DexboticTrainer):
    """
    Trainer override that swaps in the EpisodeBatchSampler and logs true batch
    ordering (per-rank) via _RankBatchLogger for debugging/verification.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._order_logger = _RankBatchLogger()  # rank-specific debug logger

    def get_train_dataloader(self):
        """
        Build a DataLoader that:
          - uses EpisodeBatchSampler to produce batch index lists
          - wraps the collator with CollatePassThrough to propagate 'indexes'
        """
        batch_sampler = EpisodeBatchSampler(
            self.train_dataset,
            mode=self.exp_config.trainer_config.dataloader_type,  # expected: 'stream' | 'group' | 'parallel_stream'
            batch_size=self.args.per_device_train_batch_size,
            group_size=self.exp_config.trainer_config.group_size,
            seed=int(getattr(self.args, "seed", 42) or 42),
            verify=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=CollatePassThrough(self.data_collator),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=(self.args.dataloader_num_workers > 0),
        )

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Get and log 'indexes' (if present) before delegating to the base compute_loss.
        'indexes' contains the ground-truth order for this batch on this rank.
        """

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        return (loss, outputs) if return_outputs else loss
