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

"""Dataset for LIBERO pixel trajectories stored in LeRobot v2.0 parquet format.

Data layout:
  <data_root>/data/chunk-000/episode_{episode_index:06d}.parquet

Each parquet file is one episode with columns:
  observation.images.image       : dict {'bytes': <PNG bytes>}  → (256, 256, 3) uint8
  observation.images.wrist_image : dict {'bytes': <PNG bytes>}  → (256, 256, 3) uint8
  observation.state              : float32 [8]
  action                         : float32 [7]   (already in [-1, 1])

No explicit 'dones' column — each file is one complete episode.
Episode ends at the last row (index T-1).

── Dual-camera skip-dynamics (always enabled) ──────────────────────────────
Training objective:
  (cam1_t, cam2_t, task, a_{t+d}, d)  →  (cam1_{t+d+1}, cam2_{t+d+1})

17-frame video layout (state_t=5, num_conditional_frames=3):
  Frame 0       : zeros                        (latent 0 – blank anchor)
  Frames 1–4    : cam1_t ×4                   (latent 1 – conditioning, agentview)
  Frames 5–8    : cam2_t ×4                   (latent 2 – conditioning, wrist)
  Frames 9–12   : cam1_{t+d+1} ×4            (latent 3 – predicted, agentview)
  Frames 13–16  : cam2_{t+d+1} ×4            (latent 4 – predicted, wrist)

  Symmetric encoding: cam1 and cam2 each encoded from 4 pixel frames (equal quality).
  Blank anchor in latent 0 ensures all conditioning info is in full 4-frame latents.

action : [1, 8] float32  — [a_{t+d} ; d/(max_delay-1)]
video  : [3, 17, 256, 256] uint8

Eval frame extraction:
  cam1_pred = video_out[:, :, 9]   (first frame of latent 3)
  cam2_pred = video_out[:, :, 13]  (first frame of latent 4)

Max window needed: max_delay + 1 = 6 (only frames t and t+d+1).

cam1 = observation.images.image       (agentview, 3rd-person)
cam2 = observation.images.wrist_image (wrist / 1st-person)

Default data root:
  /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/lerobot/lerobot--libero_spatial_image@v2.0
  Override with LEROBOT_LIBERO_DATA_ROOT environment variable.
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

_DEFAULT_DATA_ROOT = (
    "/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/lerobot"
    "/lerobot--libero_spatial_image@v2.0"
)

_CAM1_KEY = "observation.images.image"
_CAM2_KEY = "observation.images.wrist_image"
_ACT_KEY = "action"

# VAE temporal compression factor.
_TEMPORAL_COMPRESSION = 4

# state_t=5: 17 pixel frames total.
# Layout: 1 blank + 4×cam1_t + 4×cam2_t + 4×cam1_pred + 4×cam2_pred
_SEQUENCE_LENGTH = 1 + 4 * _TEMPORAL_COMPRESSION  # = 17

_MAX_RETRIES = 16


def _decode_image(cell) -> np.ndarray:
    """Decode a LeRobot v2.0 image cell (dict with 'bytes' key) → (H, W, 3) uint8."""
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, (bytes, bytearray)):
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        raise ValueError(f"Unexpected image cell format: {list(cell.keys())}")
    raise TypeError(f"Expected dict image cell, got {type(cell)}")


class LeRobotLiberoDataset(Dataset):
    """PyTorch Dataset for LIBERO pixel trajectories in LeRobot v2.0 parquet format.

    Dual-camera skip-dynamics mode only:
      video  : [3, 17, 256, 256] uint8  (state_t=5)
      action : [1, 8] float32           [a_{t+d} ; d/(max_delay-1)]

    Use with state_t=5, num_conditional_frames=3 in the experiment config.
    """

    def __init__(
        self,
        data_root: str | None = None,
        max_delay: int = 5,
        val_ratio: float = 0.1,
        mode: str = "train",
        seed: int = 0,
        t5_emb_path: str | None = None,
        task_indices: Sequence[int] | None = None,
    ):
        """
        Args:
            data_root: Path to the lerobot dataset root (contains data/, meta/).
                Falls back to LEROBOT_LIBERO_DATA_ROOT env var, then the
                compile-time default.
            max_delay: Maximum async delay D_max; delay d is sampled from [0, max_delay-1].
                Requires episode length ≥ max_delay + 1.
            val_ratio: Fraction of episodes held out for validation.
            mode: "train" or "val".
            seed: RNG seed for the train/val split.
            t5_emb_path: Path to precomputed T5 embeddings pickle
                ({task_str: tensor(1,512,1024)}). Defaults to
                <data_root>/meta/t5_embeddings.pkl. Set to "" to disable.
            task_indices: If set, only include episodes whose task_index is in
                this list. Useful for single-task ablation (e.g. [0] for task 0
                only). None = all tasks (default).
        """
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        if data_root is None:
            data_root = os.environ.get("LEROBOT_LIBERO_DATA_ROOT", _DEFAULT_DATA_ROOT)

        self.data_root = pathlib.Path(data_root)
        self.max_delay = max_delay
        self.mode = mode
        self.sequence_length = _SEQUENCE_LENGTH  # = 17
        self.task_indices = set(task_indices) if task_indices is not None else None

        # ── collect all episode parquet files ────────────────────────────────
        data_dir = self.data_root / "data" / "chunk-000"
        if not data_dir.exists():
            raise FileNotFoundError(
                f"LeRobotLiberoDataset: data directory not found: {data_dir}"
            )

        all_files = sorted(data_dir.glob("episode_*.parquet"))
        if not all_files:
            raise FileNotFoundError(f"No episode parquet files found in {data_dir}")

        # ── train/val split at episode level ─────────────────────────────────
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_files))
        n_val = max(1, int(len(all_files) * val_ratio))
        val_set = set(perm[:n_val].tolist())

        selected_files = [
            all_files[i]
            for i in range(len(all_files))
            if (i in val_set) == (mode == "val")
        ]

        # ── build flat (file, start_t) window index ───────────────────────────
        # Valid start: start_t + max_delay < T  →  start_t ≤ T - max_delay - 1
        # (so that start_t + d + 1 ≤ start_t + max_delay < T for any d)
        self._windows: list[tuple[pathlib.Path, int]] = []
        for file_path in selected_files:
            # Read lightweight columns: frame_index (for T) + task_index (for filtering)
            df_meta = pd.read_parquet(file_path, columns=["frame_index", "task_index"])
            # Apply task_indices filter (keep all when task_indices is None)
            if self.task_indices is not None:
                ep_task_index = int(df_meta["task_index"].iloc[0])
                if ep_task_index not in self.task_indices:
                    continue
            T = len(df_meta)
            max_valid_start = T - max_delay  # start_t ∈ [0, T - max_delay - 1]
            for t in range(max_valid_start):
                self._windows.append((file_path, t))

        if not self._windows:
            raise RuntimeError(
                f"No valid windows found (mode={mode}, max_delay={max_delay}). "
                "All episodes may be too short."
            )

        task_filter_str = f", task_indices={sorted(self.task_indices)}" if self.task_indices else ""
        print(
            f"[LeRobotLiberoDataset] mode={mode}: {len(self._windows):,} windows "
            f"(max_delay={max_delay}{task_filter_str}, data_root={data_root})"
        )

        # ── load precomputed T5 text embeddings ───────────────────────────────
        # Format: {task_description_str: tensor(1, 512, 1024, bfloat16)}
        # Produced by scripts/precompute_libero_t5.py
        if t5_emb_path == "":
            self._t5_embs: dict | None = None
            print("[LeRobotLiberoDataset] T5 embeddings disabled (t5_emb_path='')")
        else:
            _pkl = pathlib.Path(t5_emb_path) if t5_emb_path else (self.data_root / "meta" / "t5_embeddings.pkl")
            if _pkl.exists():
                with open(_pkl, "rb") as f:
                    self._t5_embs = pickle.load(f)
                # build task_index → task_str mapping from tasks.jsonl
                _tasks_file = self.data_root / "meta" / "tasks.jsonl"
                self._task_index_to_str: dict[int, str] = {
                    t["task_index"]: t["task"]
                    for t in (json.loads(l) for l in _tasks_file.read_text().strip().splitlines())
                }
                print(
                    f"[LeRobotLiberoDataset] Loaded T5 embeddings for "
                    f"{len(self._t5_embs)} tasks from {_pkl}"
                )
            else:
                self._t5_embs = None
                warnings.warn(
                    f"[LeRobotLiberoDataset] T5 embeddings not found at {_pkl}. "
                    "Text conditioning will use zero embeddings. "
                    "Run: python scripts/precompute_libero_t5.py"
                )

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                return self._load_sample(index)
            except Exception:
                warnings.warn(
                    f"[LeRobotLiberoDataset] error at index={index} (attempt {attempt})"
                )
                warnings.warn(traceback.format_exc())
                index = int(np.random.randint(len(self)))
        return self._load_sample(index)

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_sample(self, index: int) -> dict:
        file_path, start_t = self._windows[index]

        # Sample delay d ∈ [0, max_delay-1]
        d = int(np.random.randint(0, self.max_delay))
        pred_t = start_t + d + 1

        # Load the full episode parquet (small file, ~10–15 MB per episode)
        df = pd.read_parquet(file_path)

        # Decode the two needed frames for each camera
        cam1_t    = _decode_image(df[_CAM1_KEY].iloc[start_t])  # [H, W, 3]
        cam2_t    = _decode_image(df[_CAM2_KEY].iloc[start_t])  # [H, W, 3]
        cam1_pred = _decode_image(df[_CAM1_KEY].iloc[pred_t])   # [H, W, 3]
        cam2_pred = _decode_image(df[_CAM2_KEY].iloc[pred_t])   # [H, W, 3]

        # Action at t+d
        act = np.asarray(df[_ACT_KEY].iloc[start_t + d], dtype=np.float32)  # [7]

        H, W = cam1_t.shape[:2]
        blank = np.zeros((H, W, 3), dtype=np.uint8)

        # Build 17-frame video (state_t=5, num_conditional_frames=3)
        # Frame 0       : zeros                (latent 0 – blank anchor)
        # Frames 1–4    : cam1_t ×4           (latent 1 – cond, agentview)
        # Frames 5–8    : cam2_t ×4           (latent 2 – cond, wrist)
        # Frames 9–12   : cam1_pred ×4        (latent 3 – predicted)
        # Frames 13–16  : cam2_pred ×4        (latent 4 – predicted)
        imgs = np.concatenate([
            blank[np.newaxis],                                               # [1, H, W, 3]  frame 0
            np.stack([cam1_t,   cam1_t,   cam1_t,   cam1_t],   axis=0),     # [4, H, W, 3]  frames 1-4
            np.stack([cam2_t,   cam2_t,   cam2_t,   cam2_t],   axis=0),     # [4, H, W, 3]  frames 5-8
            np.stack([cam1_pred, cam1_pred, cam1_pred, cam1_pred], axis=0),  # [4, H, W, 3]  frames 9-12
            np.stack([cam2_pred, cam2_pred, cam2_pred, cam2_pred], axis=0),  # [4, H, W, 3]  frames 13-16
        ], axis=0)  # [17, H, W, 3]

        # → [3, 17, H, W] uint8
        video = torch.from_numpy(imgs.astype(np.uint8)).permute(3, 0, 1, 2)

        # Action with normalised delay → [1, 8]
        d_norm = float(d) / float(max(self.max_delay - 1, 1))
        action_vec = np.concatenate([act, np.array([d_norm], dtype=np.float32)])
        action = torch.from_numpy(action_vec).unsqueeze(0)  # [1, 8]

        # Metadata
        task_index = int(df["task_index"].iloc[0])
        ep_index   = int(df["episode_index"].iloc[0])
        key = f"libero_spatial/ep{ep_index:04d}/t{start_t:04d}_d{d}"

        # T5 text embedding: (512, 1024) float32 — DataLoader stacks to (B, 512, 1024)
        if self._t5_embs is not None:
            task_str = self._task_index_to_str[task_index]
            t5_emb = self._t5_embs[task_str].squeeze(0).float()  # (512, 1024)
        else:
            t5_emb = torch.zeros(512, 1024, dtype=torch.float32)

        return {
            "video": video,                                    # [3, 13, 256, 256] uint8
            "action": action,                                  # [1, 8] float32
            "__key__": key,
            "fps": 10,
            "image_size": 256 * torch.ones(4).cuda(),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(1, 256, 256).cuda(),
            "t5_text_embeddings": t5_emb,                      # [512, 1024] float32
        }


class _EpisodeBoundaryError(Exception):
    pass
