"""wm4vla experiment configurations for skip-dynamics world model training.

Extracted from cosmos_predict2/experiments/base/action.py.
Registers five experiments into Hydra ConfigStore:
  - ac_kinetix_pixels_2b             (Kinetix, 128×128, 9 frames)
  - ac_libero_pixels_2b              (LIBERO HDF5, 128×128, 13 frames)
  - ac_libero_lerobot_256_pixels_2b  (LIBERO LeRobot, 256×256, paired 5-frame views)
  - ac_libero_lerobot_256_pixels_2b_task0   (single-task ablation)
  - ac_libero_lerobot_256_pixels_2b_task01  (two-task ablation)
"""
import os

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]
_WM4VLA_WAN21_VAE_PATH = os.getenv("WM4VLA_WAN21_VAE_PATH", "/home/kyji/public/models/lightx2v/vae/Wan2.1_VAE.pth")
_WM4VLA_LIGHTVAE_PATH = os.getenv(
    "WM4VLA_LIGHTVAE_PATH", "/home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth"
)
_WM4VLA_VAE_BACKEND = os.getenv("WM4VLA_VAE_BACKEND", "lightvae").strip().lower()
_WM4VLA_VAE_PATH_BY_BACKEND = {
    "wan2pt1": _WM4VLA_WAN21_VAE_PATH,
    "lightvae": _WM4VLA_LIGHTVAE_PATH,
}
if _WM4VLA_VAE_BACKEND not in _WM4VLA_VAE_PATH_BY_BACKEND:
    raise ValueError(
        f"Invalid WM4VLA_VAE_BACKEND={_WM4VLA_VAE_BACKEND!r}. "
        "Expected one of {'wan2pt1', 'lightvae'}."
    )
_WM4VLA_VAE_PATH = _WM4VLA_VAE_PATH_BY_BACKEND[_WM4VLA_VAE_BACKEND]

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
    save_iter=2_000,
    # pyrefly: ignore  # missing-attribute
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_training_state=False,
    strict_resume=False,
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
)


# ── Kinetix (128×128, 9 frames, state_t=3) ──────────────────────────────────

ac_kinetix_pixels_2b = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "kinetix_5frame_128_train"},
            {"override /data_val": "kinetix_5frame_128_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_kinetix_pixels_skip_dynamics",
        ),
        optimizer=dict(lr=2 ** (-14.5), weight_decay=0.1),
        checkpoint=_CHECKPOINT_BASE,
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000, do_x0_prediction=False,
                    guidance=[0, 3, 7], fps=4, save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=5000, do_x0_prediction=False,
                    guidance=[0, 3, 7], fps=4, save_s3=False,
                ),
                **_CALLBACKS_NO_S3,
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=2,
                max_num_conditional_frames=2,
                conditional_frames_probs=None,
                state_t=3,
                text_encoder_config=None,
                tokenizer=dict(
                    vae_pth=_WM4VLA_VAE_PATH,
                ),
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=1,
                    use_crossattn_projection=False,
                ),
            ),
        ),
        dataloader_train=dict(batch_size=4),
    ),
    flags={"allow_objects": True},
)


# ── LIBERO HDF5 dual-camera (128×128, 13 frames, state_t=4) ─────────────────

ac_libero_pixels_2b = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "libero_dual_cam_128_train"},
            {"override /data_val": "libero_dual_cam_128_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_libero_skip_dynamics_dual_cam",
        ),
        optimizer=dict(lr=2 ** (-14.5), weight_decay=0.1),
        checkpoint=_CHECKPOINT_BASE,
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000, do_x0_prediction=False,
                    guidance=[0, 3, 7], fps=10, save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=5000, do_x0_prediction=False,
                    guidance=[0, 3, 7], fps=10, save_s3=False,
                ),
                **_CALLBACKS_NO_S3,
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=2,
                max_num_conditional_frames=2,
                conditional_frames_probs=None,
                state_t=4,
                text_encoder_config=None,
                tokenizer=dict(
                    vae_pth=_WM4VLA_VAE_PATH,
                ),
                net=dict(
                    action_dim=8,
                    num_action_per_chunk=1,
                    use_crossattn_projection=False,
                ),
            ),
        ),
        dataloader_train=dict(batch_size=4),
    ),
    flags={"allow_objects": True},
)


# ── LIBERO LeRobot paired-view short-video (256×256, 5 frames, state_t=2) ──

def _libero_lerobot_256_base(
    data_train: str,
    data_val: str,
    job_name: str,
    sample_every_n: int = 5000,
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
                    tokenizer=dict(
                        vae_pth=_WM4VLA_VAE_PATH,
                    ),
                    net=dict(
                        action_dim=8,
                        num_action_per_chunk=1,
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
    sample_every_n=2_000,
)

ac_libero_lerobot_256_pixels_2b_task01 = _libero_lerobot_256_base(
    data_train="lerobot_libero_dual_cam_256_task01_train",
    data_val="lerobot_libero_dual_cam_256_task01_val",
    job_name="2b_libero_lerobot_256_skip_dynamics_dual_cam_task01",
    sample_every_n=2_000,
)


# ── Registration ─────────────────────────────────────────────────────────────

_WM4VLA_EXPERIMENTS = [
    ac_kinetix_pixels_2b,
    ac_libero_pixels_2b,
    ac_libero_lerobot_256_pixels_2b,
    ac_libero_lerobot_256_pixels_2b_task0,
    ac_libero_lerobot_256_pixels_2b_task01,
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
