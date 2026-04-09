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

"""Dataset for paired-view LIBERO parquet trajectories.

Supported dataset layouts:

1. ``lerobot/lerobot--libero_10_image@v2.0``
2. ``physical-intelligence/libero``

Each dataset item returns a strong paired mini-batch with two single-view
samples:

  Sample 0 (cam1): [cam1_t, cam1_{t+d} x4]
  Sample 1 (cam2): [cam2_t, cam2_{t+d} x4]

Tensor layout:
  video        : [2, 3, 5, 256, 256] uint8
  action       : [2, max_delay, 8] float32
  delay_scalar : [2, 1] float32
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
    "/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot"
    "/lerobot--libero_10_image@v2.0"
)

_SCHEMA_CANDIDATES = (
    {
        "name": "lerobot_v2",
        "cam1": "observation.images.image",
        "cam2": "observation.images.wrist_image",
        "state": "observation.state",
        "action": "action",
    },
    {
        "name": "physical_intelligence_flat",
        "cam1": "image",
        "cam2": "wrist_image",
        "state": "state",
        "action": "actions",
    },
)

_TEMPORAL_COMPRESSION = 4
_SEQUENCE_LENGTH = 1 + _TEMPORAL_COMPRESSION
_MAX_RETRIES = 16


def _decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, (bytes, bytearray)):
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        raise ValueError(f"Unexpected image cell format: {list(cell.keys())}")
    raise TypeError(f"Expected dict image cell, got {type(cell)}")


def _collect_episode_files(data_root: pathlib.Path) -> list[pathlib.Path]:
    data_dir = data_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"LeRobotLiberoDataset: data directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No episode parquet files found in {data_dir}")
    return all_files


def _infer_schema_from_info(data_root: pathlib.Path) -> dict[str, str] | None:
    info_path = data_root / "meta" / "info.json"
    if not info_path.exists():
        return None

    info = json.loads(info_path.read_text())
    features = set(info.get("features", {}).keys())
    for schema in _SCHEMA_CANDIDATES:
        if schema["cam1"] in features and schema["cam2"] in features and schema["action"] in features:
            return dict(schema)
    return None


def _infer_schema_from_file(file_path: pathlib.Path) -> dict[str, str]:
    columns = set(pd.read_parquet(file_path, columns=None).columns)
    for schema in _SCHEMA_CANDIDATES:
        if schema["cam1"] in columns and schema["cam2"] in columns and schema["action"] in columns:
            return dict(schema)
    raise RuntimeError(
        f"Unsupported LIBERO parquet schema in {file_path}. "
        f"Found columns: {sorted(columns)}"
    )


class LeRobotLiberoDataset(Dataset):
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
    ):
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        if data_root is None:
            data_root = os.environ.get("LEROBOT_LIBERO_DATA_ROOT", _DEFAULT_DATA_ROOT)

        self.data_root = pathlib.Path(data_root)
        self.max_delay = max_delay
        self.sampled_delay_max = max_delay if sampled_delay_max is None else int(sampled_delay_max)
        self.delay_normalization_max = max_delay if delay_normalization_max is None else int(delay_normalization_max)
        self.mode = mode
        self.sequence_length = _SEQUENCE_LENGTH
        self.task_indices = set(task_indices) if task_indices is not None else None
        self._dataset_key_prefix = self.data_root.name.replace("/", "_")
        if not (TRAIN_DELAY_MIN <= self.sampled_delay_max <= self.max_delay):
            raise ValueError(
                f"sampled_delay_max must be in [{TRAIN_DELAY_MIN}, {self.max_delay}], got {self.sampled_delay_max}"
            )
        if self.delay_normalization_max < self.sampled_delay_max:
            raise ValueError(
                "delay_normalization_max must be >= sampled_delay_max, "
                f"got {self.delay_normalization_max} < {self.sampled_delay_max}"
            )

        all_files = _collect_episode_files(self.data_root)
        schema = _infer_schema_from_info(self.data_root) or _infer_schema_from_file(all_files[0])
        self._schema_name = schema["name"]
        self._cam1_key = schema["cam1"]
        self._cam2_key = schema["cam2"]
        self._action_key = schema["action"]

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
            episode_len = len(df_meta)
            max_valid_start = episode_len - self.sampled_delay_max
            for start_t in range(max_valid_start):
                self._windows.append((file_path, start_t))

        if not self._windows:
            raise RuntimeError(
                f"No valid windows found (mode={mode}, max_delay={max_delay}, "
                f"sampled_delay_max={self.sampled_delay_max}). "
                "All episodes may be too short or filtered out."
            )

        task_filter_str = f", task_indices={sorted(self.task_indices)}" if self.task_indices else ""
        print(
            f"[LeRobotLiberoDataset] mode={mode}: {len(self._windows):,} windows "
            f"(schema={self._schema_name}, max_delay={max_delay}, "
            f"sampled_delay_max={self.sampled_delay_max}{task_filter_str}, "
            f"data_root={data_root})"
        )

        if t5_emb_path == "":
            self._t5_embs: dict | None = None
            print("[LeRobotLiberoDataset] T5 embeddings disabled (t5_emb_path='')")
        else:
            pkl_path = pathlib.Path(t5_emb_path) if t5_emb_path else (self.data_root / "meta" / "t5_embeddings.pkl")
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self._t5_embs = pickle.load(f)
                tasks_file = self.data_root / "meta" / "tasks.jsonl"
                self._task_index_to_str: dict[int, str] = {
                    task["task_index"]: task["task"]
                    for task in (json.loads(line) for line in tasks_file.read_text().strip().splitlines())
                }
                print(
                    f"[LeRobotLiberoDataset] Loaded T5 embeddings for "
                    f"{len(self._t5_embs)} tasks from {pkl_path}"
                )
            else:
                self._t5_embs = None
                warnings.warn(
                    f"[LeRobotLiberoDataset] T5 embeddings not found at {pkl_path}. "
                    "Text conditioning will use zero embeddings. "
                    "Run: python scripts/precompute_libero_t5.py"
                )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                return self._load_sample(index)
            except Exception:
                warnings.warn(f"[LeRobotLiberoDataset] error at index={index} (attempt {attempt})")
                warnings.warn(traceback.format_exc())
                index = int(np.random.randint(len(self)))
        return self._load_sample(index)

    def _load_sample(self, index: int) -> dict:
        file_path, start_t = self._windows[index]

        d = int(np.random.randint(TRAIN_DELAY_MIN, self.sampled_delay_max + 1))
        pred_t = start_t + d

        df = pd.read_parquet(file_path)

        cam1_t = _decode_image(df[self._cam1_key].iloc[start_t])
        cam2_t = _decode_image(df[self._cam2_key].iloc[start_t])
        cam1_pred = _decode_image(df[self._cam1_key].iloc[pred_t])
        cam2_pred = _decode_image(df[self._cam2_key].iloc[pred_t])

        act_seq = np.stack(
            [np.asarray(df[self._action_key].iloc[start_t + offset], dtype=np.float32) for offset in range(d)],
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
        episode_index = int(df["episode_index"].iloc[0])
        key = [
            f"{self._dataset_key_prefix}/ep{episode_index:04d}/t{start_t:04d}_d{d}_cam1",
            f"{self._dataset_key_prefix}/ep{episode_index:04d}/t{start_t:04d}_d{d}_cam2",
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


class _EpisodeBoundaryError(Exception):
    pass
