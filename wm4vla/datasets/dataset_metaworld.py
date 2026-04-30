# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter for MetaWorld MT50 stored in LeRobot v3 parquet format.

MetaWorld differs from the LIBERO adapters in two important ways:

* It has one camera only: ``observation.image``.
* Images are stored as 480x480 PNG bytes and are resized here to 256x256.

Each sample returns one single-view short video:

  video       : [1, 3, 5, 256, 256] uint8
  action      : [1, max_delay, 5] float32  # 4D action + valid mask
  delay_scalar: [1, 1] float32

The leading view dimension is kept so the shared wm4vla collate function can
flatten single-view and paired-view datasets in the same way.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import pickle
import traceback
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from wm4vla.conditioning import normalize_delay_scalar, pack_masked_action_sequence
from wm4vla.configs.wm_conditioning import ACTION_CHUNK_LEN, TRAIN_DELAY_MIN

_DEFAULT_DATA_ROOT = "/mnt/cpfs/yangboxue/vla/wm4vla/data/dataset/lerobot/metaworld_mt50"

_IMAGE_KEY = "observation.image"
_ACT_KEY = "action"

_TEMPORAL_COMPRESSION = 4
_SEQUENCE_LENGTH = 1 + _TEMPORAL_COMPRESSION
_TARGET_IMAGE_SIZE = 256
_MAX_RETRIES = 16

_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def _decode_image(cell, image_size: int) -> np.ndarray:
    """Decode a LeRobot image cell into an RGB uint8 array with fixed size."""
    if not isinstance(cell, dict):
        raise TypeError(f"Expected dict image cell, got {type(cell)}")

    raw = cell.get("bytes")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"Unexpected image cell format: {list(cell.keys())}")

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), _BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _load_task_map(data_root: pathlib.Path) -> dict[int, str]:
    tasks_file = data_root / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        return {}
    return {
        item["task_index"]: item["task"]
        for item in (json.loads(line) for line in tasks_file.read_text().strip().splitlines())
    }


class MetaWorldDataset(Dataset):
    """PyTorch dataset for single-camera MetaWorld skip-dynamics training."""

    def __init__(
        self,
        data_root: str | None = None,
        max_delay: int = ACTION_CHUNK_LEN,
        sampled_delay_max: int | None = None,
        fixed_delay: int | None = None,
        delay_normalization_max: int | None = None,
        val_ratio: float = 0.1,
        mode: str = "train",
        seed: int = 0,
        t5_emb_path: str | None = None,
        task_indices: Sequence[int] | None = None,
        image_size: int = _TARGET_IMAGE_SIZE,
    ):
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        if data_root is None:
            data_root = os.environ.get("METAWORLD_DATA_ROOT", _DEFAULT_DATA_ROOT)

        self.data_root = pathlib.Path(data_root)
        self.max_delay = int(max_delay)
        self.sampled_delay_max = self.max_delay if sampled_delay_max is None else int(sampled_delay_max)
        self.fixed_delay = None if fixed_delay is None else int(fixed_delay)
        self.delay_normalization_max = (
            self.max_delay if delay_normalization_max is None else int(delay_normalization_max)
        )
        self.mode = mode
        self.sequence_length = _SEQUENCE_LENGTH
        self.task_indices = set(task_indices) if task_indices is not None else None
        self.image_size = int(image_size)
        self.fps = self._read_fps()

        if not (TRAIN_DELAY_MIN <= self.sampled_delay_max <= self.max_delay):
            raise ValueError(
                f"sampled_delay_max must be in [{TRAIN_DELAY_MIN}, {self.max_delay}], "
                f"got {self.sampled_delay_max}"
            )
        if self.fixed_delay is not None and not (TRAIN_DELAY_MIN <= self.fixed_delay <= self.max_delay):
            raise ValueError(
                f"fixed_delay must be in [{TRAIN_DELAY_MIN}, {self.max_delay}], got {self.fixed_delay}"
            )
        if self.fixed_delay is not None and self.sampled_delay_max < self.fixed_delay:
            raise ValueError(
                "sampled_delay_max must be >= fixed_delay, "
                f"got {self.sampled_delay_max} < {self.fixed_delay}"
            )
        effective_delay_max = self.fixed_delay if self.fixed_delay is not None else self.sampled_delay_max
        if self.delay_normalization_max < effective_delay_max:
            raise ValueError(
                "delay_normalization_max must be >= effective delay max, "
                f"got {self.delay_normalization_max} < {effective_delay_max}"
            )
        self._window_delay_max = effective_delay_max

        self._windows = self._build_windows(val_ratio=val_ratio, seed=seed)
        if not self._windows:
            fixed_delay_str = f", fixed_delay={self.fixed_delay}" if self.fixed_delay is not None else ""
            raise RuntimeError(
                f"No valid MetaWorld windows found (mode={mode}, max_delay={max_delay}, "
                f"sampled_delay_max={self.sampled_delay_max}{fixed_delay_str})."
            )

        task_filter_str = f", task_indices={sorted(self.task_indices)}" if self.task_indices else ""
        fixed_delay_str = f", fixed_delay={self.fixed_delay}" if self.fixed_delay is not None else ""
        print(
            f"[MetaWorldDataset] mode={mode}: {len(self._windows):,} windows "
            f"(max_delay={max_delay}, sampled_delay_max={self.sampled_delay_max}"
            f"{fixed_delay_str}{task_filter_str}, image_size={self.image_size}, data_root={data_root})"
        )

        self._task_index_to_str = _load_task_map(self.data_root)
        self._t5_embs = self._load_t5_embeddings(t5_emb_path)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                return self._load_sample(index)
            except Exception:
                warnings.warn(f"[MetaWorldDataset] error at index={index} (attempt {attempt})")
                warnings.warn(traceback.format_exc())
                index = int(np.random.randint(len(self)))
        return self._load_sample(index)

    def _read_fps(self) -> float:
        info_file = self.data_root / "meta" / "info.json"
        if not info_file.exists():
            return 80.0
        info = json.loads(info_file.read_text())
        return float(info.get("fps", 80.0))

    def _build_windows(self, val_ratio: float, seed: int) -> list[tuple[pathlib.Path, int, int]]:
        data_dir = self.data_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"MetaWorldDataset: data directory not found: {data_dir}")

        all_files = sorted(data_dir.glob("chunk-*/*.parquet"))
        if not all_files:
            raise FileNotFoundError(f"No MetaWorld parquet files found in {data_dir}/chunk-*")

        episode_infos: list[tuple[pathlib.Path, int, int, int]] = []
        for file_path in all_files:
            df_meta = pd.read_parquet(file_path, columns=["episode_index", "frame_index", "task_index"])
            for ep_index, ep_df in df_meta.groupby("episode_index", sort=False):
                task_index = int(ep_df["task_index"].iloc[0])
                episode_infos.append((file_path, int(ep_index), task_index, len(ep_df)))

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(episode_infos))
        n_val = max(1, int(len(episode_infos) * val_ratio))
        val_set = set(perm[:n_val].tolist())

        windows: list[tuple[pathlib.Path, int, int]] = []
        for i, (file_path, ep_index, task_index, total_frames) in enumerate(episode_infos):
            if (i in val_set) != (self.mode == "val"):
                continue
            if self.task_indices is not None and task_index not in self.task_indices:
                continue
            max_valid_start = total_frames - self._window_delay_max
            for start_t in range(max_valid_start):
                windows.append((file_path, ep_index, start_t))

        return windows

    def _load_t5_embeddings(self, t5_emb_path: str | None) -> dict | None:
        if t5_emb_path == "":
            print("[MetaWorldDataset] T5 embeddings disabled (t5_emb_path='')")
            return None

        pkl_path = pathlib.Path(t5_emb_path) if t5_emb_path else (self.data_root / "meta" / "t5_embeddings.pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                t5_embs = pickle.load(f)
            print(f"[MetaWorldDataset] Loaded T5 embeddings for {len(t5_embs)} tasks from {pkl_path}")
            return t5_embs

        warnings.warn(
            f"[MetaWorldDataset] T5 embeddings not found at {pkl_path}. "
            "Text conditioning will use zero embeddings."
        )
        return None

    def _load_sample(self, index: int) -> dict:
        file_path, ep_index, start_t = self._windows[index]

        if self.fixed_delay is not None:
            delay = self.fixed_delay
        else:
            delay = int(np.random.randint(TRAIN_DELAY_MIN, self.sampled_delay_max + 1))
        pred_t = start_t + delay

        df = pd.read_parquet(file_path)
        ep_df = (
            df[df["episode_index"] == ep_index]
            .sort_values("frame_index")
            .reset_index(drop=True)
        )
        if pred_t >= len(ep_df):
            raise IndexError(f"pred_t={pred_t} out of episode length {len(ep_df)}")

        img_t = _decode_image(ep_df[_IMAGE_KEY].iloc[start_t], image_size=self.image_size)
        img_pred = _decode_image(ep_df[_IMAGE_KEY].iloc[pred_t], image_size=self.image_size)

        frames = np.stack([img_t, img_pred, img_pred, img_pred, img_pred], axis=0)
        video = torch.from_numpy(frames[None].astype(np.uint8)).permute(0, 4, 1, 2, 3)

        act_seq = np.stack(
            [np.asarray(ep_df[_ACT_KEY].iloc[start_t + offset], dtype=np.float32) for offset in range(delay)],
            axis=0,
        )
        action_seq = pack_masked_action_sequence(
            actions=act_seq,
            delay=delay,
            chunk_len=self.max_delay,
        )
        delay_scalar = normalize_delay_scalar(delay=delay, max_delay=self.delay_normalization_max)
        action = torch.from_numpy(action_seq).unsqueeze(0)
        delay_scalar = torch.from_numpy(delay_scalar).unsqueeze(0)

        task_index = int(ep_df["task_index"].iloc[0])
        key = [f"metaworld_mt50/task{task_index:02d}/ep{ep_index:04d}/t{start_t:04d}_d{delay}"]

        if self._t5_embs is not None:
            task_str = self._task_index_to_str[task_index]
            t5_emb = self._t5_embs[task_str].squeeze(0).float()
        else:
            t5_emb = torch.zeros(512, 1024, dtype=torch.float32)
        t5_emb = t5_emb.unsqueeze(0)

        return {
            "video": video,
            "action": action,
            "delay_scalar": delay_scalar,
            "__key__": key,
            "fps": torch.full((1,), self.fps, dtype=torch.float32),
            "image_size": torch.full((1, 4), float(self.image_size), dtype=torch.float32).cuda(),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(1, 1, self.image_size, self.image_size).cuda(),
            "t5_text_embeddings": t5_emb,
        }
