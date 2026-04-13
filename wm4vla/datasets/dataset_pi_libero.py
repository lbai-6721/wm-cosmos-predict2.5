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

"""Dataset adapter for physical-intelligence/libero parquet episodes.

This loader keeps the exact same training interface as LeRobotLiberoDataset:

  video             : [2, 3, 5, 256, 256] uint8
  action            : [2, max_delay, 8] float32
  delay_scalar      : [2, 1] float32
  t5_text_embeddings: [2, 512, 1024] float32

The only differences are:
  - It reads from physical-intelligence/libero/{data,meta}
  - It adapts column names:
      image       -> agent-view image
      wrist_image -> wrist image
      actions     -> action
  - It supports benchmark-level filtering over the 40-task merged dataset.
"""

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

_DEFAULT_DATA_ROOT = (
    "/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/"
    "physical-intelligence/libero"
)

_CAM1_KEY = "image"
_CAM2_KEY = "wrist_image"
_ACT_KEY = "actions"

_TEMPORAL_COMPRESSION = 4
_SEQUENCE_LENGTH = 1 + _TEMPORAL_COMPRESSION
_MAX_RETRIES = 16

PI_LIBERO_BENCHMARK_TASKS = {
    "libero_10": tuple(range(0, 10)),
    "libero_goal": tuple(range(10, 20)),
    "libero_object": tuple(range(20, 30)),
    "libero_spatial": tuple(range(30, 40)),
}


def _decode_image(cell) -> np.ndarray:
    """Decode a LeRobot image cell (dict with 'bytes' key) -> (H, W, 3) uint8."""
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, (bytes, bytearray)):
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        raise ValueError(f"Unexpected image cell format: {list(cell.keys())}")
    raise TypeError(f"Expected dict image cell, got {type(cell)}")


def _resolve_task_filter(
    benchmark: str | None,
    task_indices: Sequence[int] | None,
) -> set[int] | None:
    """Combine benchmark-level and explicit task filters into one set."""
    benchmark_tasks = None
    if benchmark is not None:
        if benchmark not in PI_LIBERO_BENCHMARK_TASKS:
            valid = ", ".join(sorted(PI_LIBERO_BENCHMARK_TASKS))
            raise ValueError(f"Unknown benchmark {benchmark!r}; valid benchmarks: {valid}")
        benchmark_tasks = set(PI_LIBERO_BENCHMARK_TASKS[benchmark])

    explicit_tasks = set(task_indices) if task_indices is not None else None
    if benchmark_tasks is None:
        return explicit_tasks
    if explicit_tasks is None:
        return benchmark_tasks
    return benchmark_tasks & explicit_tasks


class PILiberoDataset(Dataset):
    """PyTorch dataset for physical-intelligence/libero paired-view training."""

    def __init__(
        self,
        data_root: str | None = None,
        max_delay: int = ACTION_CHUNK_LEN,
        sampled_delay_max: int | None = None,
        delay_normalization_max: int | None = None,
        val_ratio: float = 0.1,
        mode: str = "train",
        seed: int = 0,
        t5_emb_path: str | None = None,
        task_indices: Sequence[int] | None = None,
        benchmark: str | None = None,
    ):
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        if data_root is None:
            data_root = os.environ.get("PI_LIBERO_DATA_ROOT", _DEFAULT_DATA_ROOT)

        self.data_root = pathlib.Path(data_root)
        self.max_delay = max_delay
        self.sampled_delay_max = max_delay if sampled_delay_max is None else int(sampled_delay_max)
        self.delay_normalization_max = (
            max_delay if delay_normalization_max is None else int(delay_normalization_max)
        )
        self.mode = mode
        self.sequence_length = _SEQUENCE_LENGTH
        self.benchmark = benchmark
        self.task_indices = _resolve_task_filter(benchmark=benchmark, task_indices=task_indices)

        if not (TRAIN_DELAY_MIN <= self.sampled_delay_max <= self.max_delay):
            raise ValueError(
                f"sampled_delay_max must be in [{TRAIN_DELAY_MIN}, {self.max_delay}], "
                f"got {self.sampled_delay_max}"
            )
        if self.delay_normalization_max < self.sampled_delay_max:
            raise ValueError(
                "delay_normalization_max must be >= sampled_delay_max, "
                f"got {self.delay_normalization_max} < {self.sampled_delay_max}"
            )

        data_dir = self.data_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"PILiberoDataset: data directory not found: {data_dir}")

        all_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
        if not all_files:
            raise FileNotFoundError(f"No episode parquet files found in {data_dir}/chunk-*")

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_files))
        n_val = max(1, int(len(all_files) * val_ratio))
        val_set = set(perm[:n_val].tolist())

        selected_files = [
            all_files[i]
            for i in range(len(all_files))
            if (i in val_set) == (mode == "val")
        ]

        self._windows: list[tuple[pathlib.Path, int]] = []
        for file_path in selected_files:
            df_meta = pd.read_parquet(file_path, columns=["frame_index", "task_index"])
            if self.task_indices is not None:
                ep_task_index = int(df_meta["task_index"].iloc[0])
                if ep_task_index not in self.task_indices:
                    continue
            total_frames = len(df_meta)
            max_valid_start = total_frames - self.sampled_delay_max
            for start_t in range(max_valid_start):
                self._windows.append((file_path, start_t))

        if not self._windows:
            raise RuntimeError(
                f"No valid windows found (mode={mode}, max_delay={max_delay}, "
                f"sampled_delay_max={self.sampled_delay_max}, benchmark={benchmark!r})."
            )

        filter_parts = []
        if self.benchmark is not None:
            filter_parts.append(f"benchmark={self.benchmark}")
        if self.task_indices is not None:
            filter_parts.append(f"task_indices={sorted(self.task_indices)}")
        filter_str = f", {', '.join(filter_parts)}" if filter_parts else ""
        print(
            f"[PILiberoDataset] mode={mode}: {len(self._windows):,} windows "
            f"(max_delay={max_delay}, sampled_delay_max={self.sampled_delay_max}"
            f"{filter_str}, data_root={data_root})"
        )

        if t5_emb_path == "":
            self._t5_embs: dict | None = None
            print("[PILiberoDataset] T5 embeddings disabled (t5_emb_path='')")
        else:
            t5_path = pathlib.Path(t5_emb_path) if t5_emb_path else (self.data_root / "meta" / "t5_embeddings.pkl")
            if t5_path.exists():
                with open(t5_path, "rb") as f:
                    self._t5_embs = pickle.load(f)
                tasks_file = self.data_root / "meta" / "tasks.jsonl"
                self._task_index_to_str: dict[int, str] = {
                    item["task_index"]: item["task"]
                    for item in (json.loads(line) for line in tasks_file.read_text().strip().splitlines())
                }
                print(
                    f"[PILiberoDataset] Loaded T5 embeddings for "
                    f"{len(self._t5_embs)} tasks from {t5_path}"
                )
            else:
                self._t5_embs = None
                warnings.warn(
                    f"[PILiberoDataset] T5 embeddings not found at {t5_path}. "
                    "Text conditioning will use zero embeddings."
                )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                return self._load_sample(index)
            except Exception:
                warnings.warn(f"[PILiberoDataset] error at index={index} (attempt {attempt})")
                warnings.warn(traceback.format_exc())
                index = int(np.random.randint(len(self)))
        return self._load_sample(index)

    def _load_sample(self, index: int) -> dict:
        file_path, start_t = self._windows[index]

        d = int(np.random.randint(TRAIN_DELAY_MIN, self.sampled_delay_max + 1))
        pred_t = start_t + d

        df = pd.read_parquet(file_path)

        cam1_t = _decode_image(df[_CAM1_KEY].iloc[start_t])
        cam2_t = _decode_image(df[_CAM2_KEY].iloc[start_t])
        cam1_pred = _decode_image(df[_CAM1_KEY].iloc[pred_t])
        cam2_pred = _decode_image(df[_CAM2_KEY].iloc[pred_t])

        act_seq = np.stack(
            [np.asarray(df[_ACT_KEY].iloc[start_t + offset], dtype=np.float32) for offset in range(d)],
            axis=0,
        )

        cam1_imgs = np.stack([cam1_t, cam1_pred, cam1_pred, cam1_pred, cam1_pred], axis=0)
        cam2_imgs = np.stack([cam2_t, cam2_pred, cam2_pred, cam2_pred, cam2_pred], axis=0)
        paired_imgs = np.stack([cam1_imgs, cam2_imgs], axis=0)

        video = torch.from_numpy(paired_imgs.astype(np.uint8)).permute(0, 4, 1, 2, 3)

        action_seq = pack_masked_action_sequence(actions=act_seq, delay=d, chunk_len=self.max_delay)
        delay_scalar = normalize_delay_scalar(delay=d, max_delay=self.delay_normalization_max)
        action = torch.from_numpy(action_seq).unsqueeze(0).repeat(2, 1, 1)
        delay_scalar = torch.from_numpy(delay_scalar).unsqueeze(0).repeat(2, 1)

        task_index = int(df["task_index"].iloc[0])
        ep_index = int(df["episode_index"].iloc[0])
        benchmark_prefix = self.benchmark if self.benchmark is not None else "all"
        key = [
            f"pi_libero/{benchmark_prefix}/ep{ep_index:04d}/t{start_t:04d}_d{d}_cam1",
            f"pi_libero/{benchmark_prefix}/ep{ep_index:04d}/t{start_t:04d}_d{d}_cam2",
        ]

        if self._t5_embs is not None:
            task_str = self._task_index_to_str[task_index]
            t5_emb = self._t5_embs[task_str].squeeze(0).float()
        else:
            t5_emb = torch.zeros(512, 1024, dtype=torch.float32)
        t5_emb = t5_emb.unsqueeze(0).repeat(2, 1, 1)

        return {
            "video": video,
            "action": action,
            "delay_scalar": delay_scalar,
            "__key__": key,
            "fps": torch.full((2,), 10.0, dtype=torch.float32),
            "image_size": (256 * torch.ones(2, 4)).cuda(),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(2, 1, 256, 256).cuda(),
            "t5_text_embeddings": t5_emb,
        }
