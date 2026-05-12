"""wm4vla experiment configurations for skip-dynamics world model training.

Extracted from cosmos_predict2/experiments/base/action.py.
Registers eight experiments into Hydra ConfigStore:
  - ac_libero_lerobot_256_pixels_2b  (LIBERO LeRobot, 256×256, paired 5-frame views)
  - ac_libero_lerobot_256_pixels_2b_task0   (single-task ablation)
  - ac_libero_lerobot_256_pixels_2b_task01  (two-task ablation)
  - ac_pi_libero_256_pixels_2b_all          (physical-intelligence/libero, all 40 tasks)
  - ac_pi_libero_256_pixels_2b_10           (Libero-10 benchmark)
  - ac_pi_libero_256_pixels_2b_goal         (Libero-Goal benchmark)
  - ac_pi_libero_256_pixels_2b_object       (Libero-Object benchmark)
  - ac_pi_libero_256_pixels_2b_spatial      (Libero-Spatial benchmark)
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
from wm4vla.configs.wm_conditioning import (
    ACTION_CHUNK_LEN,
    LIBERO_ACTION_SLOT_DIM,
)

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]

# Shared trainer callback settings (disable S3 upload).
_CALLBACKS_NO_S3 = dict(
    heart_beat=dict(save_s3=False),
    iter_speed=dict(hit_thres=100, save_s3=False),
    device_monitor=dict(save_s3=False),
    wandb=dict(save_s3=False),
    wandb_10x=dict(save_s3=False),
    dataloader_speed=dict(save_s3=False),
)

_CHECKPOINT_BASE = dict(
    save_iter=1_000,
    # pyrefly: ignore  # missing-attribute
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_training_state=False,
    strict_resume=False,
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
)


# ── LIBERO LeRobot paired-view short-video (256×256, 5 frames, state_t=2) ──

def _libero_lerobot_256_base(
    data_train: str,
    data_val: str,
    job_name: str,
    sample_every_n: int = 500,
) -> LazyDict:
    """Factory for LIBERO LeRobot 256×256 experiments (shared structure)."""
    return LazyDict(
        dict(
            defaults=[
                DEFAULT_CHECKPOINT.experiment,
                {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
                {"override /net": "cosmos_v1_2B_action_conditioned"},
                {"override /conditioner": "action_conditioned_video_conditioner"},
                {"override /data_train": data_train},
                {"override /data_val": data_val},
                "_self_",
            ],
            job=dict(
                project="cosmos_predict2_action_conditioned",
                group="cosmos_predict_v2p5",
                name=job_name,
            ),
            optimizer=dict(lr=2 ** (-14.5), weight_decay=0.1),
            checkpoint=_CHECKPOINT_BASE,
            trainer=dict(
                straggler_detection=dict(enabled=False),
                callbacks=dict(
                    every_n_sample_reg=dict(
                        every_n=sample_every_n, do_x0_prediction=False,
                        guidance=[0, 3, 7], fps=10, save_s3=False,
                    ),
                    every_n_sample_ema=dict(
                        every_n=sample_every_n, do_x0_prediction=False,
                        guidance=[0, 3, 7], fps=10, save_s3=False,
                    ),
                    **_CALLBACKS_NO_S3,
                ),
            ),
            model_parallel=dict(context_parallel_size=1),
            model=dict(
                config=dict(
                    min_num_conditional_frames=1,
                    max_num_conditional_frames=1,
                    conditional_frames_probs=None,
                    state_t=2,
                    text_encoder_config=None,
                    net=dict(
                        action_dim=LIBERO_ACTION_SLOT_DIM,
                        num_action_per_chunk=ACTION_CHUNK_LEN,
                        use_crossattn_projection=False,
                    ),
                ),
            ),
            dataloader_train=dict(batch_size=2),
        ),
        flags={"allow_objects": True},
    )


ac_libero_lerobot_256_pixels_2b = _libero_lerobot_256_base(
    data_train="lerobot_libero_dual_cam_256_train",
    data_val="lerobot_libero_dual_cam_256_val",
    job_name="2b_libero_object_lerobot_256_skip_dynamics_dual_cam_b32",
)

ac_libero_lerobot_256_pixels_2b_task0 = _libero_lerobot_256_base(
    data_train="lerobot_libero_dual_cam_256_task0_train",
    data_val="lerobot_libero_dual_cam_256_task0_val",
    job_name="2b_libero_10_lerobot_256_skip_dynamics_dual_cam_task0",
)

ac_libero_lerobot_256_pixels_2b_task01 = _libero_lerobot_256_base(
    data_train="lerobot_libero_dual_cam_256_task01_train",
    data_val="lerobot_libero_dual_cam_256_task01_val",
    job_name="2b_libero_lerobot_256_skip_dynamics_dual_cam_task01",
)

ac_pi_libero_256_pixels_2b_all = _libero_lerobot_256_base(
    data_train="pi_libero_all_256_train",
    data_val="pi_libero_all_256_val",
    job_name="2b_pi_libero_256_skip_dynamics_dual_cam_all",
)

ac_pi_libero_256_pixels_2b_10 = _libero_lerobot_256_base(
    data_train="pi_libero_10_256_train",
    data_val="pi_libero_10_256_val",
    job_name="2b_pi_libero_256_skip_dynamics_dual_cam_10",
)

ac_pi_libero_256_pixels_2b_goal = _libero_lerobot_256_base(
    data_train="pi_libero_goal_256_train",
    data_val="pi_libero_goal_256_val",
    job_name="2b_pi_libero_256_skip_dynamics_dual_cam_goal",
)

ac_pi_libero_256_pixels_2b_object = _libero_lerobot_256_base(
    data_train="pi_libero_object_256_train",
    data_val="pi_libero_object_256_val",
    job_name="2b_pi_libero_256_skip_dynamics_dual_cam_object",
)

ac_pi_libero_256_pixels_2b_spatial = _libero_lerobot_256_base(
    data_train="pi_libero_spatial_256_train",
    data_val="pi_libero_spatial_256_val",
    job_name="2b_pi_libero_256_skip_dynamics_dual_cam_spatial",
)


# ── Registration ─────────────────────────────────────────────────────────────

_WM4VLA_EXPERIMENTS = [
    ac_libero_lerobot_256_pixels_2b,
    ac_libero_lerobot_256_pixels_2b_task0,
    ac_libero_lerobot_256_pixels_2b_task01,
    ac_pi_libero_256_pixels_2b_all,
    ac_pi_libero_256_pixels_2b_10,
    ac_pi_libero_256_pixels_2b_goal,
    ac_pi_libero_256_pixels_2b_object,
    ac_pi_libero_256_pixels_2b_spatial,
]


def register_wm4vla_experiments():
    """Register all wm4vla experiment configs into Hydra ConfigStore."""
    cs = ConfigStore.instance()
    for _item in _WM4VLA_EXPERIMENTS:
        experiment_name = [
            name.lower()
            for name, value in globals().items()
            if value is _item and not name.startswith("_")
        ][0]
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
