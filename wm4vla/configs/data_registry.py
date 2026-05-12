"""wm4vla dataloader registrations for Hydra ConfigStore.

Extracted from cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py.
Registers LIBERO LeRobot and PI LIBERO dataloaders.
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
import torch
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from wm4vla.configs.wm_conditioning import TRAIN_DELAY_MAX
from wm4vla.datasets.dataset_lerobot_libero import LeRobotLiberoDataset
from wm4vla.datasets.dataset_pi_libero import PILiberoDataset


def _get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def _collate_paired_view_batch(batch):
    """Flatten dataset-produced paired view samples into one training batch.

    Each dataset item already contains a strong paired mini-batch with shape:
      video: [2, 3, 5, H, W]
      action: [2, max_delay, slot_dim]
      delay_scalar: [2, 1]
      ...

    When DataLoader batch_size = N, `batch` is a list of N such dicts.
    This collate function concatenates along the leading paired-view dimension
    so the trainer receives standard tensors:
      video: [2N, 3, 5, H, W]
      action: [2N, max_delay, slot_dim]
      delay_scalar: [2N, 1]
    """
    assert len(batch) > 0, "Empty batch is not allowed"

    collated = {}
    tensor_cat_keys = {
        "video",
        "action",
        "delay_scalar",
        "fps",
        "image_size",
        "padding_mask",
        "t5_text_embeddings",
    }

    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key == "__key__":
            flat_keys = []
            for value in values:
                if isinstance(value, list):
                    flat_keys.extend(value)
                else:
                    flat_keys.append(value)
            collated[key] = flat_keys
        elif key in tensor_cat_keys:
            collated[key] = torch.cat(values, dim=0)
        elif key == "num_frames":
            collated[key] = values[0]
        else:
            collated[key] = values

    return collated


def register_wm4vla_data():
    """Register all wm4vla dataloaders into Hydra ConfigStore."""
    cs = ConfigStore.instance()

    # ── LIBERO LeRobot paired-view short-video (256×256, state_t=2) ─────────
    _lerobot_libero_data_root = os.environ.get(
        "LEROBOT_LIBERO_DATA_ROOT",
        "/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot"
        "/lerobot--libero_spatial_image@v2.0",
    )
    _t5_emb_path = os.environ.get("LEROBOT_LIBERO_T5_EMB_PATH", None)

    def _lerobot_pair(task_indices=None):
        """Create a (train, val) dataset pair for LeRobot LIBERO."""
        train_ds = L(LeRobotLiberoDataset)(
            data_root=_lerobot_libero_data_root,
            max_delay=TRAIN_DELAY_MAX, val_ratio=0.1, mode="train", seed=0,
            t5_emb_path=_t5_emb_path, task_indices=task_indices,
        )
        val_ds = L(LeRobotLiberoDataset)(
            data_root=_lerobot_libero_data_root,
            max_delay=TRAIN_DELAY_MAX, val_ratio=0.1, mode="val", seed=0,
            t5_emb_path=_t5_emb_path, task_indices=task_indices,
        )
        train_dl = L(DataLoader)(
            dataset=train_ds,
            sampler=L(_get_sampler)(dataset=train_ds),
            batch_size=1, drop_last=True,
            collate_fn=_collate_paired_view_batch,
        )
        val_dl = L(DataLoader)(
            dataset=val_ds,
            sampler=L(_get_sampler)(dataset=val_ds),
            batch_size=1, drop_last=True,
            collate_fn=_collate_paired_view_batch,
        )
        return train_dl, val_dl

    # Full dataset (all tasks)
    train_dl, val_dl = _lerobot_pair()
    cs.store(group="data_train", package="dataloader_train",
             name="lerobot_libero_dual_cam_256_train", node=train_dl)
    cs.store(group="data_val", package="dataloader_val",
             name="lerobot_libero_dual_cam_256_val", node=val_dl)

    # Task 0 only
    train_dl, val_dl = _lerobot_pair(task_indices=[0])
    cs.store(group="data_train", package="dataloader_train",
             name="lerobot_libero_dual_cam_256_task0_train", node=train_dl)
    cs.store(group="data_val", package="dataloader_val",
             name="lerobot_libero_dual_cam_256_task0_val", node=val_dl)

    # Task 0 + 1
    train_dl, val_dl = _lerobot_pair(task_indices=[0, 1])
    cs.store(group="data_train", package="dataloader_train",
             name="lerobot_libero_dual_cam_256_task01_train", node=train_dl)
    cs.store(group="data_val", package="dataloader_val",
             name="lerobot_libero_dual_cam_256_task01_val", node=val_dl)

    # ── physical-intelligence/libero paired-view short-video (256x256) ──────
    _pi_libero_data_root = os.environ.get(
        "PI_LIBERO_DATA_ROOT",
        "/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/"
        "physical-intelligence/libero",
    )
    _pi_t5_emb_path = os.environ.get("PI_LIBERO_T5_EMB_PATH", None)

    def _pi_libero_pair(benchmark=None):
        """Create a (train, val) dataset pair for physical-intelligence/libero."""
        train_ds = L(PILiberoDataset)(
            data_root=_pi_libero_data_root,
            max_delay=TRAIN_DELAY_MAX, val_ratio=0.1, mode="train", seed=0,
            t5_emb_path=_pi_t5_emb_path, benchmark=benchmark,
        )
        val_ds = L(PILiberoDataset)(
            data_root=_pi_libero_data_root,
            max_delay=TRAIN_DELAY_MAX, val_ratio=0.1, mode="val", seed=0,
            t5_emb_path=_pi_t5_emb_path, benchmark=benchmark,
        )
        train_dl = L(DataLoader)(
            dataset=train_ds,
            sampler=L(_get_sampler)(dataset=train_ds),
            batch_size=1, drop_last=True,
            collate_fn=_collate_paired_view_batch,
        )
        val_dl = L(DataLoader)(
            dataset=val_ds,
            sampler=L(_get_sampler)(dataset=val_ds),
            batch_size=1, drop_last=True,
            collate_fn=_collate_paired_view_batch,
        )
        return train_dl, val_dl

    for benchmark_name, store_prefix in [
        (None, "pi_libero_all_256"),
        ("libero_10", "pi_libero_10_256"),
        ("libero_goal", "pi_libero_goal_256"),
        ("libero_object", "pi_libero_object_256"),
        ("libero_spatial", "pi_libero_spatial_256"),
    ]:
        train_dl, val_dl = _pi_libero_pair(benchmark=benchmark_name)
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"{store_prefix}_train",
            node=train_dl,
        )
        cs.store(
            group="data_val",
            package="dataloader_val",
            name=f"{store_prefix}_val",
            node=val_dl,
        )
