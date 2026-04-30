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

from copy import deepcopy
import os

from hydra.core.config_store import ConfigStore  # type: ignore[import]

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.interactive.configs.registry_defaults.teacher_model_paths import (
    ACTION_CONDITIONED_TEACHER_CKPT_2B_256X320,
)
from cosmos_predict2._src.interactive.configs.registry_experiment.experiments_dmd2_predict2p5 import (
    make_experiment,
)
from wm4vla.configs.wm_conditioning import (
    ACTION_CHUNK_LEN,
    LIBERO_ACTION_SLOT_DIM,
    METAWORLD_ACTION_SLOT_DIM,
)

# Bridge dataset - 13 frame prediction at 256x320 resolution
dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320 = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320",
    data_train="bridge_13frame_480_640_train",
    net="cosmos_v1_2B_action_chunk_conditioned_student",
    net_teacher="cosmos_v1_2B_action_chunk_conditioned_teacher",
    net_fake_score="cosmos_v1_2B_action_chunk_conditioned_fake_score",
    conditioner="action_conditioned_video_conditioner",
    resolution="256",
    cp_size=1,
    overrides=dict(
        model=dict(
            config=dict(
                state_t=4,
                use_clean_cond_timesteps=False,
                conditional_frames_probs={0: 0.0, 1: 1.0, 2: 0.0},
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_fake_score=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_256X320,
                teacher_guidance=0,
                student_update_freq=10,
            ),
        ),
        dataloader_train=dict(
            batch_size=40,
            sampler=dict(
                dataset=dict(
                    video_size=[256, 320],
                    num_action_per_chunk=12,
                    fps_downsample_ratio=1,
                    gripper_rescale_factor=1,
                ),
            ),
            dataset=dict(
                video_size=[256, 320],
                num_action_per_chunk=12,
                fps_downsample_ratio=1,
                gripper_rescale_factor=1,
            ),
        ),
    ),
)
# Remove the nested dataloaders structure inherited from base make_experiment
del dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320["dataloader_train"][
    "dataloaders"
]

# Bridge dataset - 13 frame prediction at 480x640 resolution (if you have a 480p teacher)
dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640 = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640",
    data_train="bridge_13frame_480_640_train",
    net="cosmos_v1_2B_action_chunk_conditioned_student",
    net_teacher="cosmos_v1_2B_action_chunk_conditioned_teacher",
    net_fake_score="cosmos_v1_2B_action_chunk_conditioned_fake_score",
    conditioner="action_conditioned_video_conditioner",
    resolution="480",
    cp_size=1,
    # NOTE: Update this to your 480p teacher checkpoint if you have one
    overrides=dict(
        model=dict(
            config=dict(
                state_t=4,
                use_clean_cond_timesteps=False,
                conditional_frames_probs={0: 0.0, 1: 1.0, 2: 0.0},
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_fake_score=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_256X320,
                teacher_guidance=0,
                student_update_freq=10,
            ),
        ),
        dataloader_train=dict(
            batch_size=10,
            sampler=dict(
                dataset=dict(
                    video_size=[480, 640],
                    num_action_per_chunk=12,
                    fps_downsample_ratio=1,
                    gripper_rescale_factor=1,
                ),
            ),
            dataset=dict(
                video_size=[480, 640],
                num_action_per_chunk=12,
                fps_downsample_ratio=1,
                gripper_rescale_factor=1,
            ),
        ),
    ),
)
# Remove the nested dataloaders structure inherited from base make_experiment
del dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640["dataloader_train"][
    "dataloaders"
]


def _build_no_s3_run(job: LazyDict) -> LazyDict:
    """
    Build a no S3 run of the given job.
    """
    no_s3_job = deepcopy(job)

    teacher_load_path = no_s3_job["model"]["config"]["teacher_load_from"]["load_path"]
    # Keep teacher checkpoint path unresolved during config import.
    # This avoids network downloads as a side effect when users only run
    # distilled-model evaluation with skip_teacher_init=True.
    resolved_path = teacher_load_path
    no_s3_job["model"]["config"]["teacher_load_from"] = {
        "load_path": resolved_path,
        "credentials": None,
    }

    no_s3_job["job"]["name"] = f"{job['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    no_s3_job["upload_reproducible_setup"] = False

    no_s3_job["checkpoint"]["save_to_object_store"]["enabled"] = False
    no_s3_job["checkpoint"]["load_from_object_store"]["enabled"] = False

    no_s3_job["trainer"]["straggler_detection"] = {"enabled": False}
    no_s3_job["trainer"]["callbacks"] = {
        "heart_beat": {"save_s3": False},
        "iter_speed": {"save_s3": False},
        "device_monitor": {"save_s3": False},
        "every_n_sample_reg": {"save_s3": False, "every_n": 500},
        "every_n_sample_ema": {"save_s3": False, "every_n": 500},
        "wandb": {"save_s3": False},
        "wandb_10x": {"save_s3": False},
        "dataloader_speed": {"save_s3": False},
    }

    return no_s3_job


def _teacher_load_from_env(env_var: str) -> dict:
    """Resolve teacher checkpoint path from env without hard-coding local paths."""
    return {
        "load_path": os.getenv(env_var, ""),
        "credentials": None,
    }


def _make_wm_libero_distill_experiment(
    *,
    name: str,
    data_train: str,
    teacher_ckpt_env_var: str,
    action_slot_dim: int = LIBERO_ACTION_SLOT_DIM,
    tokenizer_name: str = "wan2pt1_tokenizer",
    tokenizer_overrides: dict | None = None,
    batch_size: int = 2,
) -> LazyDict:
    """Factory for 256x256 wm4vla/student distillation experiments."""
    if tokenizer_overrides is None and tokenizer_name == "wan2pt1_tokenizer":
        tokenizer_overrides = {
            "vae_pth": "hf://nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
        }

    model_config = dict(
        # paired/single-view 5 pixel frames -> 2 latent frames (temporal compression = 4)
        state_t=2,
        # Teacher trained without clean cond timesteps (conditional_frame_timestep=-1.0)
        use_clean_cond_timesteps=False,
        # Teacher trained with adjust_video_noise=False -> multiplier must be 1.0
        multiply_noise_by_video_len=False,
        # Always 1 conditional frame (view_t in each sample)
        conditional_frames_probs={0: 0.0, 1: 1.0, 2: 0.0},
        min_num_conditional_frames=1,
        max_num_conditional_frames=1,
        # 4-GPU single-node training
        fsdp_shard_size=4,
        # Use precomputed T5 embeddings from data batch
        text_encoder_config=None,
        net=dict(
            action_dim=action_slot_dim,
            num_action_per_chunk=ACTION_CHUNK_LEN,
            use_crossattn_projection=False,
        ),
        net_fake_score=dict(
            action_dim=action_slot_dim,
            num_action_per_chunk=ACTION_CHUNK_LEN,
            use_crossattn_projection=False,
        ),
        net_teacher=dict(
            action_dim=action_slot_dim,
            num_action_per_chunk=ACTION_CHUNK_LEN,
            use_crossattn_projection=False,
        ),
        teacher_load_from=_teacher_load_from_env(teacher_ckpt_env_var),
        teacher_guidance=0,
        student_update_freq=5,
    )
    if tokenizer_overrides is not None:
        model_config["tokenizer"] = tokenizer_overrides

    experiment = make_experiment(
        name=name,
        data_train=data_train,
        net="cosmos_v1_2B_action_conditioned_student",
        net_teacher="cosmos_v1_2B_action_conditioned_teacher",
        net_fake_score="cosmos_v1_2B_action_conditioned_fake_score",
        conditioner="action_conditioned_video_conditioner",
        tokenizer=tokenizer_name,
        resolution="256",
        cp_size=1,
        overrides=dict(
            model=dict(config=model_config),
            dataloader_train=dict(batch_size=batch_size),
            # Disable all S3 I/O for local training (no credentials needed)
            upload_reproducible_setup=False,
            checkpoint=dict(
                save_to_object_store=dict(enabled=False),
                load_from_object_store=dict(enabled=False),
            ),
            trainer=dict(
                straggler_detection=dict(enabled=False),
                callbacks=dict(
                    # Disable val-prompt sampling: requires online text encoder which we don't have
                    every_n_sample_reg=dict(do_sample_val_prompts=False, save_s3=False),
                    every_n_sample_ema=dict(do_sample_val_prompts=False, save_s3=False),
                    heart_beat=dict(save_s3=False),
                    iter_speed=dict(save_s3=False),
                    device_monitor=dict(save_s3=False),
                    wandb=dict(save_s3=False),
                    wandb_10x=dict(save_s3=False),
                    dataloader_speed=dict(save_s3=False),
                ),
            ),
        ),
    )
    # Remove nested dataloaders structure inherited from make_experiment
    del experiment["dataloader_train"]["dataloaders"]
    return experiment


# LIBERO LeRobot 256×256 dual-cam task0 - paired 5 frame prediction (state_t=2)
# Teacher: ActionConditionedMinimalV1LVGDiT, action_dim=8, num_action_per_chunk=8
dmd2_trigflow_distill_wm_libero_lerobot_256_task0 = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_libero_lerobot_256_task0",
    data_train="lerobot_libero_dual_cam_256_task0_train",
    teacher_ckpt_env_var="WM4VLA_LIBERO_TASK0_TEACHER_CKPT",
)


dmd2_trigflow_distill_wm_pi_libero_256_all = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_pi_libero_256_all",
    data_train="pi_libero_all_256_train",
    teacher_ckpt_env_var="WM4VLA_PI_LIBERO_TEACHER_CKPT_ALL",
)


dmd2_trigflow_distill_wm_pi_libero_256_10 = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_pi_libero_256_10",
    data_train="pi_libero_10_256_train",
    teacher_ckpt_env_var="WM4VLA_PI_LIBERO_TEACHER_CKPT_10",
)


dmd2_trigflow_distill_wm_pi_libero_256_goal = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_pi_libero_256_goal",
    data_train="pi_libero_goal_256_train",
    teacher_ckpt_env_var="WM4VLA_PI_LIBERO_TEACHER_CKPT_GOAL",
)


dmd2_trigflow_distill_wm_pi_libero_256_object = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_pi_libero_256_object",
    data_train="pi_libero_object_256_train",
    teacher_ckpt_env_var="WM4VLA_PI_LIBERO_TEACHER_CKPT_OBJECT",
)


dmd2_trigflow_distill_wm_pi_libero_256_spatial = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_pi_libero_256_spatial",
    data_train="pi_libero_spatial_256_train",
    teacher_ckpt_env_var="WM4VLA_PI_LIBERO_TEACHER_CKPT_SPATIAL",
)


# MetaWorld MT50 256x256 single-cam - 5 frame prediction (state_t=2)
# Teacher: ActionConditionedMinimalV1LVGDiT, action_dim=5, num_action_per_chunk=8
dmd2_trigflow_distill_wm_metaworld_mt50_256 = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_metaworld_mt50_256",
    data_train="metaworld_mt50_256_train",
    teacher_ckpt_env_var="WM4VLA_METAWORLD_MT50_TEACHER_CKPT",
    action_slot_dim=METAWORLD_ACTION_SLOT_DIM,
    tokenizer_name="wan2pt1_lightvae_tokenizer",
)


dmd2_trigflow_distill_wm_metaworld_mt50_256_task0 = _make_wm_libero_distill_experiment(
    name="dmd2_trigflow_distill_wm_metaworld_mt50_256_task0",
    data_train="metaworld_mt50_256_task0_train",
    teacher_ckpt_env_var="WM4VLA_METAWORLD_TASK0_TEACHER_CKPT",
    action_slot_dim=METAWORLD_ACTION_SLOT_DIM,
    tokenizer_name="wan2pt1_lightvae_tokenizer",
)


# Kinetix 128×128 - 9 frame prediction (state_t=3)
# Teacher: ActionConditionedMinimalV1LVGDiT, action_dim=7, num_action_per_chunk=1
dmd2_trigflow_distill_wm_kinetix_128_9frame = make_experiment(
    name="dmd2_trigflow_distill_wm_kinetix_128_9frame",
    data_train="kinetix_5frame_128_train",
    net="cosmos_v1_2B_action_conditioned_student",
    net_teacher="cosmos_v1_2B_action_conditioned_teacher",
    net_fake_score="cosmos_v1_2B_action_conditioned_fake_score",
    conditioner="action_conditioned_video_conditioner",
    # Teacher was trained with wan2pt1_tokenizer (DEFAULT_CHECKPOINT.experiment overrides wan2pt2→wan2pt1)
    tokenizer="wan2pt1_tokenizer",
    resolution="128",
    cp_size=1,
    overrides=dict(
        model=dict(
            config=dict(
                # 9 pixel frames → 3 latent frames (temporal compression = 4, +1 blank)
                state_t=3,
                # Teacher trained without clean cond timesteps (conditional_frame_timestep=-1.0)
                use_clean_cond_timesteps=False,
                # Teacher trained with adjust_video_noise=False → multiplier must be 1.0
                multiply_noise_by_video_len=False,
                # Always 2 conditional frames (blank + obs_t)
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 1.0},
                min_num_conditional_frames=2,
                max_num_conditional_frames=2,
                # 4-GPU single-node training
                fsdp_shard_size=4,
                # Kinetix has no text condition (uses zero T5 embeddings from data batch)
                text_encoder_config=None,
                # Use Hugging Face tokenizer checkpoint so cache location follows HF_* env vars.
                tokenizer=dict(
                    vae_pth="hf://nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
                ),
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=1,
                    use_crossattn_projection=False,
                ),
                net_fake_score=dict(
                    action_dim=7,
                    num_action_per_chunk=1,
                    use_crossattn_projection=False,
                ),
                net_teacher=dict(
                    action_dim=7,
                    num_action_per_chunk=1,
                    use_crossattn_projection=False,
                ),
                teacher_load_from=_teacher_load_from_env("WM4VLA_KINETIX_TEACHER_CKPT"),
                teacher_guidance=0,
                student_update_freq=5,
            ),
        ),
        dataloader_train=dict(batch_size=4),
        # Disable all S3 I/O for local training (no credentials needed)
        upload_reproducible_setup=False,
        checkpoint=dict(
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                # Disable val-prompt sampling: requires online text encoder which we don't have
                every_n_sample_reg=dict(do_sample_val_prompts=False, save_s3=False),
                every_n_sample_ema=dict(do_sample_val_prompts=False, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
    ),
)
# Remove nested dataloaders structure inherited from make_experiment
del dmd2_trigflow_distill_wm_kinetix_128_9frame["dataloader_train"]["dataloaders"]


cs = ConfigStore.instance()
"""
4-GPU single-node distillation commands:

LIBERO task0 (256x256, paired 5 frames):
torchrun --nproc_per_node=4 --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- experiment=dmd2_trigflow_distill_wm_libero_lerobot_256_task0 \
  model.config.teacher_load_from.load_path=/path/to/model_ema_bf16.pt \
  job.wandb_mode=disabled

PI-LIBERO benchmark/all (256x256, paired 5 frames):
torchrun --nproc_per_node=4 --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- experiment=dmd2_trigflow_distill_wm_pi_libero_256_all \
  model.config.teacher_load_from.load_path=/path/to/model_ema_bf16.pt \
  job.wandb_mode=disabled

MetaWorld MT50 (256x256, single-cam 5 frames):
torchrun --nproc_per_node=4 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- experiment=dmd2_trigflow_distill_wm_metaworld_mt50_256 \
  model.config.teacher_load_from.load_path=/path/to/model_ema_bf16.pt \
  job.wandb_mode=disabled

Kinetix (128x128, 9 frames):
torchrun --nproc_per_node=4 --master_port=12342 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- experiment=dmd2_trigflow_distill_wm_kinetix_128_9frame \
  model.config.teacher_load_from.load_path=/path/to/model_ema_bf16.pt \
  job.wandb_mode=disabled

2B (original Bridge):
torchrun --nproc_per_node=4 --master_port=12340 -m scripts.train --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py -- experiment=dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V
"""
for _item in [
    dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320,
    dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640,
    dmd2_trigflow_distill_wm_libero_lerobot_256_task0,
    dmd2_trigflow_distill_wm_pi_libero_256_all,
    dmd2_trigflow_distill_wm_pi_libero_256_10,
    dmd2_trigflow_distill_wm_pi_libero_256_goal,
    dmd2_trigflow_distill_wm_pi_libero_256_object,
    dmd2_trigflow_distill_wm_pi_libero_256_spatial,
    dmd2_trigflow_distill_wm_metaworld_mt50_256,
    dmd2_trigflow_distill_wm_metaworld_mt50_256_task0,
    dmd2_trigflow_distill_wm_kinetix_128_9frame,
]:
    cs.store(
        group="experiment",
        package="_global_",
        name=f"{_item['job']['name']}",
        node=_item,
    )

    cs.store(
        group="experiment",
        package="_global_",
        name=f"{_item['job']['name']}_no_s3",
        node=_build_no_s3_run(_item),
    )
