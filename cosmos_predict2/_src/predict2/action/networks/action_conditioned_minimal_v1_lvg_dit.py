# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT
from wm4vla.configs.wm_conditioning import DELAY_SCALAR_DIM


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActionMLPTemporalMixerBlock(nn.Module):
    def __init__(self, num_slots: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        token_hidden_dim = max(num_slots * 2, 16)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = Mlp(
            in_features=num_slots,
            hidden_features=token_hidden_dim,
            out_features=num_slots,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=drop,
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            out_features=hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=drop,
        )

    def forward(self, slot_features: torch.Tensor, valid_mask_bool: torch.Tensor) -> torch.Tensor:
        slot_mask = valid_mask_bool.unsqueeze(-1).type_as(slot_features)

        token_mixed = self.token_norm(slot_features) * slot_mask
        token_mixed = rearrange(token_mixed, "b t c -> b c t")
        token_mixed = self.token_mlp(token_mixed)
        token_mixed = rearrange(token_mixed, "b c t -> b t c") * slot_mask
        slot_features = slot_features + token_mixed

        channel_mixed = self.channel_norm(slot_features)
        channel_mixed = self.channel_mlp(channel_mixed) * slot_mask
        slot_features = slot_features + channel_mixed
        return slot_features * slot_mask


def _split_legacy_single_action(
    action: torch.Tensor,
    delay_scalar: Optional[torch.Tensor],
    expected_slot_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Support legacy wm4vla inputs shaped as one action with delay packed in the last channel."""
    if delay_scalar is not None:
        return action, delay_scalar
    if action.ndim != 3 or action.shape[1] != 1 or action.shape[2] < 2:
        raise ValueError("delay_scalar must be provided unless using legacy one-slot packed action format")

    raw_action = action[..., :-1]
    packed_delay = action[..., -1]
    if raw_action.shape[2] != expected_slot_dim - 1:
        raise ValueError(
            f"Legacy packed action expects raw dim {expected_slot_dim - 1}, got {raw_action.shape[2]}"
        )

    valid_mask = torch.ones_like(packed_delay).unsqueeze(-1)
    upgraded_action = torch.cat([raw_action, valid_mask], dim=-1)
    upgraded_delay = packed_delay.unsqueeze(-1)
    return upgraded_action, upgraded_delay


class ActionConditionedMinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask

        action_dim = kwargs.get("action_dim", 10 * 8)
        if "action_dim" in kwargs:
            del kwargs["action_dim"]

        num_action_per_chunk = kwargs.get("num_action_per_chunk", 12)
        if "num_action_per_chunk" in kwargs:
            del kwargs["num_action_per_chunk"]

        self.timestep_scale = timestep_scale
        log.info(f"timestep_scale: {timestep_scale}")

        super().__init__(*args, **kwargs)

        self._action_slot_dim = action_dim
        self._num_action_slots = num_action_per_chunk
        self._action_slot_hidden_dim = max(self.model_channels // 4, 128)
        self.action_slot_embedder = Mlp(
            in_features=self._action_slot_dim,
            hidden_features=self.model_channels * 2,
            out_features=self._action_slot_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_position_embedding = nn.Parameter(torch.zeros(1, self._num_action_slots, self._action_slot_hidden_dim))
        nn.init.normal_(self.action_position_embedding, std=0.02)

        self.action_temporal_mixer = nn.ModuleList(
            [
                ActionMLPTemporalMixerBlock(
                    num_slots=self._num_action_slots,
                    hidden_dim=self._action_slot_hidden_dim,
                    drop=0.0,
                )
                for _ in range(2)
            ]
        )
        self.action_pooling_score = nn.Sequential(
            nn.LayerNorm(self._action_slot_hidden_dim),
            nn.Linear(self._action_slot_hidden_dim, 1, bias=False),
        )
        self.action_summary_norm = nn.LayerNorm(self._action_slot_hidden_dim)
        self.action_summary_to_B_D = Mlp(
            in_features=self._action_slot_hidden_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_summary_to_B_3D = Mlp(
            in_features=self._action_slot_hidden_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.delay_embedder_B_D = Mlp(
            in_features=DELAY_SCALAR_DIM,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.delay_embedder_B_3D = Mlp(
            in_features=DELAY_SCALAR_DIM,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def _encode_action_prefix(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3:
            raise ValueError(f"Expected action with shape [B, T, D], got {tuple(action.shape)}")
        if action.shape[2] != self._action_slot_dim:
            raise ValueError(f"Expected action slot dim {self._action_slot_dim}, got {action.shape[2]}")
        if action.shape[1] > self._num_action_slots:
            raise ValueError(f"Expected at most {self._num_action_slots} action slots, got {action.shape[1]}")

        valid_mask_bool = action[..., -1] > 0
        slot_features = self.action_slot_embedder(action)
        slot_features = slot_features + self.action_position_embedding[:, : action.shape[1], :].type_as(slot_features)

        if action.shape[1] < self._num_action_slots:
            pad_slots = self._num_action_slots - action.shape[1]
            pad = torch.zeros(
                action.shape[0],
                pad_slots,
                self._action_slot_hidden_dim,
                dtype=slot_features.dtype,
                device=slot_features.device,
            )
            slot_features = torch.cat([slot_features, pad], dim=1)
            valid_mask_bool = torch.cat(
                [
                    valid_mask_bool,
                    torch.zeros(action.shape[0], pad_slots, dtype=torch.bool, device=action.device),
                ],
                dim=1,
            )

        if not valid_mask_bool.any(dim=1).all():
            raise ValueError("Each action prefix must contain at least one valid action")

        slot_mask = valid_mask_bool.unsqueeze(-1).type_as(slot_features)
        slot_features = slot_features * slot_mask

        for layer in self.action_temporal_mixer:
            slot_features = layer(slot_features, valid_mask_bool)

        pool_logits = self.action_pooling_score(slot_features).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~valid_mask_bool, torch.finfo(slot_features.dtype).min)
        pool_weights = torch.softmax(pool_logits, dim=1).unsqueeze(-1)
        pooled_features = (slot_features * pool_weights).sum(dim=1, keepdim=True)
        pooled_features = self.action_summary_norm(pooled_features)
        return pooled_features

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        delay_scalar: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        timesteps_B_T = timesteps_B_T * self.timestep_scale

        assert action is not None, "action must be provided"
        action, delay_scalar = _split_legacy_single_action(action, delay_scalar, expected_slot_dim=self._action_slot_dim)
        action_summary = self._encode_action_prefix(action)
        if delay_scalar.ndim == 1:
            delay_scalar = delay_scalar.unsqueeze(1)
        if delay_scalar.ndim == 2:
            delay_scalar = delay_scalar.unsqueeze(1)
        action_emb_B_D = self.action_summary_to_B_D(action_summary)
        action_emb_B_3D = self.action_summary_to_B_3D(action_summary)
        delay_emb_B_D = self.delay_embedder_B_D(delay_scalar)
        delay_emb_B_3D = self.delay_embedder_B_3D(delay_scalar)

        intermediate_feature_ids = None

        assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, "extra_image_context_dim must be set if img_context_emb is provided"
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D + delay_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D + delay_emb_B_3D

            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp


class ActionChunkConditionedMinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask

        action_dim = kwargs.get("action_dim", 10 * 8)
        if "action_dim" in kwargs:
            del kwargs["action_dim"]

        self._num_action_per_latent_frame = kwargs.get("temporal_compression_ratio", 4)
        if "temporal_compression_ratio" in kwargs:
            del kwargs["temporal_compression_ratio"]

        if "num_action_per_chunk" in kwargs:
            del kwargs["num_action_per_chunk"]

        self._hidden_dim_in_action_embedder = kwargs.get("hidden_dim_in_action_embedder", None)
        if "hidden_dim_in_action_embedder" in kwargs:
            del kwargs["hidden_dim_in_action_embedder"]

        self.timestep_scale = timestep_scale

        super().__init__(*args, **kwargs)

        if self._hidden_dim_in_action_embedder is None:
            self._hidden_dim_in_action_embedder = self.model_channels * 4

        log.info(f"hidden_dim_in_action_embedder: {self._hidden_dim_in_action_embedder}")

        self.action_embedder_B_D = Mlp(
            in_features=action_dim * self._num_action_per_latent_frame,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=action_dim * self._num_action_per_latent_frame,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.delay_embedder_B_D = Mlp(
            in_features=DELAY_SCALAR_DIM,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.delay_embedder_B_3D = Mlp(
            in_features=DELAY_SCALAR_DIM,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        delay_scalar: Optional[torch.Tensor] = None,
        intermediate_feature_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        timesteps_B_T = timesteps_B_T * self.timestep_scale

        assert action is not None, "action must be provided"
        action, delay_scalar = _split_legacy_single_action(action, delay_scalar, expected_slot_dim=self.action_embedder_B_D.fc1.in_features // self._num_action_per_latent_frame)
        num_actions = action.shape[1]
        action = rearrange(action, "b t d -> b 1 (t d)")
        action = rearrange(action, "b 1 (t d) -> b t d", t=num_actions // self._num_action_per_latent_frame)
        if delay_scalar.ndim == 1:
            delay_scalar = delay_scalar.unsqueeze(1)
        if delay_scalar.ndim == 2:
            delay_scalar = delay_scalar.unsqueeze(1)
        action_emb_B_D = self.action_embedder_B_D(action)
        action_emb_B_3D = self.action_embedder_B_3D(action)
        delay_emb_B_D = self.delay_embedder_B_D(delay_scalar)
        delay_emb_B_3D = self.delay_embedder_B_3D(delay_scalar)

        zero_pad_action_emb_B_D = torch.zeros_like(action_emb_B_D[:, :1, :], device=action_emb_B_D.device)
        zero_pad_action_emb_B_3D = torch.zeros_like(action_emb_B_3D[:, :1, :], device=action_emb_B_3D.device)

        action_emb_B_D = torch.cat([zero_pad_action_emb_B_D, action_emb_B_D], dim=1)
        action_emb_B_3D = torch.cat([zero_pad_action_emb_B_3D, action_emb_B_3D], dim=1)

        assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, "extra_image_context_dim must be set if img_context_emb is provided"
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D + delay_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D + delay_emb_B_3D

            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp
