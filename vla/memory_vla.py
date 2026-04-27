from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
import torch.nn.functional as F
from transformers import LlamaTokenizerFast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from action_model.action_model import ActionModel
from action_model.models import DiT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossTransformerBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # Feed‑Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self,
                query: torch.Tensor, # (B, N, D)
                k: torch.Tensor, # (B, M, D)
                v: torch.Tensor, # (B, M, D)
                ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # residual + LN
        x = self.attn_norm(query + attn_out)

        # FFN + LN
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class BottleneckSE(nn.Module):
    def __init__(self, C_in, C_mid, C_out):
        super().__init__()
        self.C_in = C_in
        self.C_mid = C_mid
        self.C_out = C_out

        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, C_mid//16, 1),
            nn.ReLU(),
            nn.Conv2d(C_mid//16, C_mid, 1),
            nn.Sigmoid()
        )

        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)

    def forward(self, x):
        _b, _n, _c = x.shape
        _h = _w = int(math.sqrt(_n))
        assert _h * _h == _n, "Input feature has no spatial structure"

        x = x.reshape(_b, _h, _w, _c).permute(0, 3, 1, 2)  # (B, C_in, H, W)
        z = self.act(self.reduce(x))
        w = self.excite(z)

        final = self.expand(z * w)
        final = final.reshape(_b, self.C_out, _n).permute(0, 2, 1)
        return final


class GateFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(
            self.proj(
                torch.cat([x1, x2],
                dim=-1)
            )
        )

        fused = scale * x1 + (1 - scale) * x2
        return fused


class CogMemBank(nn.Module):
    def __init__(self,
                 dataloader_type: str,
                 group_size: int,
                 token_size: int,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 ):
        super().__init__()
        assert dataloader_type in ('stream', 'group')
        assert fusion_type in ('gate', 'add')
        assert consolidate_type in ('fifo', 'tome')

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList([
            CrossTransformerBlock(self.token_size)
            for _ in range(self.retrieval_layers)
        ])

        if self.fusion_type == 'gate':
            self.gate_fusion_blocks = GateFusion(self.token_size)

        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size,
                frequency_embedding_size=self.token_size // 4)
        else:
            self.timestep_encoder = None

        self.reset()

    def reset(self):
        # bank[episode_id] = [(timestep, feat[N,D]), ...]
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]

        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i+1].flatten(1) if feats[i+1].dim() > 1 else feats[i+1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())

        timestep_i, feat_i = bank[idx_max]
        timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)

        bank[idx_max] = (timestep_i, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
            self,
            episode_id,
            feat: torch.Tensor,
            timestep: Optional[torch.Tensor]):
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError

    def process_batch(
        self,
        tokens: torch.Tensor, # [B, N, D_role]
        episode_ids: np.array,
        timesteps: np.array,
    ) -> torch.Tensor:
        assert episode_ids is not None, "episode_ids must be provided during training"

        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided during training"

        B, N, D = tokens.shape
        outputs = []

        if self.training:
            if self.dataloader_type == 'group':
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == 'stream':
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            # 1) episode management
            eid = episode_ids[i]
            if self.training:
                if self.dataloader_type == 'group':
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
                if self.dataloader_type == 'stream':
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]

            # 2) memory retrieval
            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)

            hist = self.bank.get(eid, [])
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)  # (1, T*N, D)

                if self.use_timestep_pe:
                    hist_timesteps = [t for t, _ in hist]
                    hist_timesteps = torch.tensor(hist_timesteps).to(working_mem.device)
                    pe = self.timestep_encoder(hist_timesteps).unsqueeze(0)  # (1, T, D)
                    pe = pe.repeat_interleave(N, dim=1) # (1, T*N, D)
                else:
                    pe = torch.zeros_like(episode_mem)

                query = working_mem
                for block in self.retrieval_blocks:
                    query = block(query, episode_mem + pe, episode_mem)

                retrieved_episode_mem = query

            else:
                # without history：working memory as episode memory
                retrieved_episode_mem = working_mem  # (1, N, D)

            # 3) memory adaptive fusion
            if self.fusion_type == 'add':
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            elif self.fusion_type == 'gate':
                fused_feats = self.gate_fusion_blocks(working_mem, retrieved_episode_mem)

            outputs.append(fused_feats)

            # 4) memory consolidate
            timestep_i = timesteps[i] if self.use_timestep_pe else None

            if self.update_fused:
                self._memory_consolidate(eid, fused_feats.squeeze(0), timestep_i)
            else:
                self._memory_consolidate(eid, tokens[i], timestep_i)

        return torch.cat(outputs, dim=0)  # [B, N, D_role]


class PerMemBank(CogMemBank):
    def __init__(self,
                 dataloader_type: str,
                 group_size: int,
                 token_size: int,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 ):
        super().__init__(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
        )


class MemoryVLA(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = 'DiT-L',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        dataloader_type: str = "group",
        group_size: int = 16,
        per_token_size: int = 256,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = 'gate',
        consolidate_type: str = 'tome',
        update_fused: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.use_ema = use_ema
        self.norm_stats = norm_stats

        self.cog_token_size = token_size

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.per_token_size = per_token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.cur_timestep = 0

        self.vision_dim = self.vlm.vision_backbone.dino_featurizer.patch_embed.proj.weight.shape[0] + \
                 self.vlm.vision_backbone.siglip_featurizer.patch_embed.proj.weight.shape[0]


        self.per_compr = BottleneckSE(
            C_in=self.vision_dim,
            C_mid=self.per_token_size * 2,
            C_out=self.per_token_size,
        )

        self.cog_mem_bank = CogMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.cog_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )

        self.per_mem_bank = PerMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.per_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )

        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            future_action_window_size=future_action_window_size,
            use_per_attn=True,
            per_token_size=per_token_size,
        )

        self.all_module_keys = []
        self._trainable_module_keys = []

        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys.append('ema_diffusion')

        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        for name, module in self.named_children():
            if name != "vlm" and any(p.requires_grad for p in module.parameters()):
                self.all_module_keys.append(name)
                self._trainable_module_keys.append(name)

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_masks: Optional[torch.FloatTensor] = None,
        timesteps: np.array = None,
        episode_ids: np.array = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # extract the visual token number
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        # extract the last hidden state and the learnable EOS token feature
        last_hidden_state = output.hidden_states[-1]
        last_hidden_state = last_hidden_state[:, num_patch :]

        # extract the cognition feature
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden_state.size(-1))

        cog_tokens = last_hidden_state.gather(
            1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        vision_feats = self.vlm.vision_feats
        per_tokens = self.per_compr(vision_feats)

        cog_tokens = self.cog_mem_bank.process_batch(
            tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        per_tokens = self.per_mem_bank.process_batch(
            tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)

        cog_tokens_repeated = cog_tokens.repeat(
            repeated_diffusion_steps, 1, 1)

        per_tokens_repeated = per_tokens.repeat(
            repeated_diffusion_steps, 1, 1)

        # Action model forward and compute loss
        loss = self.action_model.loss(
            actions_repeated,
            cog_tokens_repeated,
            per_tokens_repeated,
        )

        return loss, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        action_model_type: str = 'DiT-L',
        use_ema: bool = False,
        norm_stats = None,
        use_bf16: bool = False,
        **kwargs,
    ) -> MemoryVLA:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        if use_bf16:
            raw_state = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
            for k in raw_state:
                for subk in raw_state[k]:
                    raw_state[k][subk] = raw_state[k][subk].to(torch.bfloat16)

            model_state_dict = raw_state
        else:
            model_state_dict = torch.load(
                pretrained_checkpoint, map_location="cuda"
            )["model"]

        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize
        memory_vla = MemoryVLA(vlm,
                        token_size = vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        **kwargs,
                        )

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            memory_vla.action_model.load_state_dict(model_state_dict["action_model"], strict=False)
            assert use_ema is False, "Does not support using EMA weights from pretrained checkpoint."
            if "ema_diffusion" in model_state_dict and use_ema:
                memory_vla.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                memory_vla.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")

        # load other weights
        for key, sub_state in model_state_dict.items():
            if key not in {"projector", "llm_backbone", "vision_backbone",
                           "action_model", "ema_diffusion"}:
                module = getattr(memory_vla, key, None)
                module.load_state_dict(sub_state, strict=True)

        del model_state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if use_bf16:
            memory_vla = memory_vla.to("cuda", dtype=torch.bfloat16)

        return memory_vla

    @torch.inference_mode()
    def predict_action(
        self, image: Image, 
        instruction: str,
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        episode_first_frame: str = 'False',
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        model_dtype = next(self.parameters()).dtype

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device, dtype=model_dtype)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device, dtype=model_dtype) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float32

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(autocast_dtype == torch.bfloat16)):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs,
            )
            # fmt: on

        model_dtype = next(self.action_model.net.parameters()).dtype
        cog_tokens = output.hidden_states[-1][-1][:,-1,:]
        assert (cog_tokens.shape[0], cog_tokens.shape[1]) == (1,4096), "Batch size must be 1 for action prediction"

        cog_tokens = cog_tokens.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        vision_feats = self.vlm.vision_feats
        per_tokens = self.per_compr(vision_feats)

        assert episode_first_frame in ['True', 'False'], "episode_first_frame must be 'True' or 'False'"
        if episode_first_frame == 'True':
            print(" ** reset memory ** ")
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            self.cur_timestep = 0

        episode_ids = [0]
        timesteps = [torch.tensor(self.cur_timestep, device=cog_tokens.device)]
        self.cur_timestep += 1

        cog_tokens = self.cog_mem_bank.process_batch(
            tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        per_tokens = self.per_mem_bank.process_batch(
            tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        # Sample random noise
        B = cog_tokens.shape[0]
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cog_tokens.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[k, D]
            uncondition = uncondition.expand(B, *uncondition.shape[1:]) #[B, k, D]
            z = torch.cat([cog_tokens, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
            model_kwargs.update({'per_token': per_tokens.repeat(2, 1, 1)})  # Repeat for unconditioned and conditioned samples
        else:
            model_kwargs = dict(z=cog_tokens)
            sample_fn = self.action_model.net.forward
            model_kwargs.update({'per_token': per_tokens})

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cog_tokens.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cog_tokens.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions


    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
