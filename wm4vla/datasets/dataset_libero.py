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

"""Dataset for LIBERO pixel-observation trajectories (skip-dynamics WM).

Data layout (official LIBERO HDF5):
  <libero_data_dir>/<suite_name>/<task_name>_demo.hdf5

Each HDF5 contains:
  data/demo_0, data/demo_1, ...
    obs/agentview_rgb   : uint8  [T, H, W, 3]    (128×128 native)
    obs/eye_in_hand_rgb : uint8  [T, H, W, 3]    (128×128 native)
    actions             : float32 [T, 7]          already in [-1, 1]
    dones               : bool/int [T]

Frame alignment:
  actions[t] is the action taken at obs[t] which produces obs[t+1].
  The last frame has dones[T-1]=1.

── Single-camera mode (dual_camera=False) ───────────────────────────────────
Skip-dynamics sampling (max_delay=5, d ∈ [0, 4]):
  video   [3, 5, H, W] uint8  — [cam_t, cam_{t+d+1}×4]
  action  [1, 8] float32      — [a_{t+d} ; d/(max_delay-1)]
  state_t = 2   Max window = max_delay + 1 = 6 (only frames t and t+d+1).

── Dual-camera mode (dual_camera=True) ──────────────────────────────────────
Pseudo-video packing: each camera image repeated ×4 to fill latent, both
cameras concatenated to form a 13-frame sequence.

  video   [3, 13, H, W] uint8  — 13 frames (state_t=4, num_conditional_frames=2):
    Frame 0      : cam1_t                        (conditioning latent 1)
    Frames 1–4   : [cam2_t, zeros, zeros, zeros] (conditioning latent 2)
    Frames 5–8   : cam1_{t+d+1} ×4              (predicted latent 1 – agentview)
    Frames 9–12  : cam2_{t+d+1} ×4              (predicted latent 2 – wrist)
  action  [1, 8] float32      — [a_{t+d} ; d/(max_delay-1)]

  cam1 = agentview_rgb  (3rd-person fixed)
  cam2 = eye_in_hand_rgb (wrist / 1st-person)

  Max window = max_delay + 1 = 6 (only frames t and t+d+1 needed).

  At eval:
    Decode predicted latent 1 → cam1 frame 0;  latent 2 → cam2 frame 0.
    Resize 128×128 → 256×256 before feeding into policy.

Environment variable:
  LIBERO_DATA_DIR   – root that contains libero_spatial/, libero_10/, etc.
                      defaults to /home/kyji/public/dataset/LIBERO
"""

import os
import pathlib
import traceback
import warnings
from typing import Sequence

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

_DEFAULT_LIBERO_DATA_DIR = "/home/kyji/public/dataset/LIBERO"

LIBERO_SUITE_NAMES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

_MAX_RETRIES = 16

# VAE temporal compression factor (1 latent frame covers this many pixel frames).
_TEMPORAL_COMPRESSION = 4

# Single-camera: predicted pixel frames per sample.
_NUM_PRED_FRAMES = _TEMPORAL_COMPRESSION  # = 4

# Dual-camera camera keys (order determines latent assignment).
_CAM1_KEY = "agentview_rgb"
_CAM2_KEY = "eye_in_hand_rgb"


class LiberoPixelDataset(Dataset):
    """PyTorch Dataset for LIBERO pixel trajectory data (skip-dynamics).

    Single-camera mode (dual_camera=False):
      video  : [3, 5, H, W] uint8  — [cam_t, cam_{t+d+1}×4]   (state_t=2)
      action : [1, 8] float32      — [a_{t+d} ; d/(max_delay-1)]
      → use with state_t=2 in the experiment config

    Dual-camera mode (dual_camera=True):
      13-frame video (state_t=4, num_conditional_frames=2):
        Frame 0      : cam1_t                           (conditioning latent 1)
        Frames 1–4   : [cam2_t, zeros, zeros, zeros]    (conditioning latent 2)
        Frames 5–8   : cam1_{t+d+1} ×4                 (predicted latent 1 – agentview)
        Frames 9–12  : cam2_{t+d+1} ×4                 (predicted latent 2 – wrist)
      video  : [3, 13, H, W] uint8
      action : [1, 8]  float32
      → use with state_t=4, num_conditional_frames=2 in the experiment config
    """

    def __init__(
        self,
        libero_data_dir: str | None = None,
        suite_names: Sequence[str] = ("libero_spatial",),
        max_delay: int = 5,
        val_ratio: float = 0.1,
        mode: str = "train",
        seed: int = 0,
        video_size: list | None = None,
        camera_key: str = "agentview_rgb",
        dual_camera: bool = False,
    ):
        """
        Args:
            libero_data_dir: Root dir containing libero_spatial/, etc.
                Falls back to the LIBERO_DATA_DIR env var, then the
                compile-time default.
            suite_names: Which LIBERO suites to include.
            max_delay: Maximum async delay D_max; delay d is sampled from [0, max_delay-1].
            val_ratio: Fraction of demos held out for validation.
            mode: "train" or "val".
            seed: RNG seed for the train/val split.
            video_size: [H, W] to resize frames to (None = keep native size).
            camera_key: Camera to use in single-camera mode
                        ("agentview_rgb" or "eye_in_hand_rgb"). Ignored when
                        dual_camera=True (always uses agentview + eye_in_hand).
            dual_camera: If True, produce a 13-frame video with both camera
                         views as conditioning and the single predicted frame
                         (repeated ×4) per camera. Use with state_t=4.
        """
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        if libero_data_dir is None:
            libero_data_dir = os.environ.get("LIBERO_DATA_DIR", _DEFAULT_LIBERO_DATA_DIR)

        self.libero_data_dir = pathlib.Path(libero_data_dir)
        self.max_delay = max_delay
        self.dual_camera = dual_camera
        self.mode = mode
        self.video_size = video_size
        self.camera_key = camera_key

        if dual_camera:
            # Only need frames t and t+d+1
            self._max_window = max_delay + 1                       # = 6 for max_delay=5
            # state_t=4: 1 + (4-1)×4 = 13 pixel frames
            # Layout: [cam1_t | cam2_t,0,0,0 | cam1_{t+d+1}×4 | cam2_{t+d+1}×4]
            self.sequence_length = 1 + 3 * _TEMPORAL_COMPRESSION  # = 13
        else:
            # Only need frames t and t+d+1
            self._max_window = max_delay + 1                       # = 6 for max_delay=5
            self.sequence_length = 1 + _NUM_PRED_FRAMES            # = 5

        # ── collect all (hdf5_path, demo_key, T) triples ─────────────────────
        all_demos: list[tuple[pathlib.Path, str, int]] = []
        for suite in suite_names:
            suite_dir = self.libero_data_dir / suite
            if not suite_dir.exists():
                warnings.warn(f"[LiberoPixelDataset] suite dir not found: {suite_dir}")
                continue
            for hdf5_path in sorted(suite_dir.glob("*.hdf5")):
                try:
                    with h5py.File(hdf5_path, "r") as f:
                        for demo_key in sorted(f["data"].keys()):
                            T = f["data"][demo_key]["actions"].shape[0]
                            if T >= self._max_window:
                                all_demos.append((hdf5_path, demo_key, T))
                except Exception as e:
                    warnings.warn(f"[LiberoPixelDataset] skipping {hdf5_path}: {e}")

        if not all_demos:
            raise FileNotFoundError(
                f"No valid LIBERO demos found under {self.libero_data_dir} "
                f"for suites {list(suite_names)}"
            )

        # ── train/val split at demo level ─────────────────────────────────────
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_demos))
        n_val = max(1, int(len(all_demos) * val_ratio))
        val_set = set(perm[:n_val].tolist())

        selected = [
            all_demos[i]
            for i in range(len(all_demos))
            if (i in val_set) == (mode == "val")
        ]

        # ── build flat window index ───────────────────────────────────────────
        # valid_starts per demo: start_t such that start_t + max_window - 1 ≤ T - 1
        # i.e., start_t ≤ T - max_window
        self._windows: list[tuple[pathlib.Path, str, int]] = []
        for hdf5_path, demo_key, T in selected:
            valid_starts = T - self._max_window + 1
            for t in range(valid_starts):
                self._windows.append((hdf5_path, demo_key, t))

        print(
            f"[LiberoPixelDataset] mode={mode}: {len(selected)} demos, "
            f"{len(self._windows):,} windows "
            f"(max_delay={max_delay}, max_window={self._max_window})"
        )

    # ── index helpers ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    # ── data loading ─────────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                return self._load_sample(index)
            except _EpisodeBoundaryError:
                index = int(np.random.randint(len(self)))
            except Exception:
                warnings.warn(f"[LiberoPixelDataset] error at index={index} (attempt {attempt})")
                warnings.warn(traceback.format_exc())
                index = int(np.random.randint(len(self)))
        return self._load_sample(index)

    def _load_sample(self, index: int) -> dict:
        hdf5_path, demo_key, start_t = self._windows[index]

        # Sample delay d ∈ [0, max_delay-1]
        d = int(np.random.randint(0, self.max_delay))

        if self.dual_camera:
            imgs, act = self._load_dual_camera(hdf5_path, demo_key, start_t, d)
        else:
            imgs, act = self._load_single_camera(hdf5_path, demo_key, start_t, d)

        # ── images → [3, T, H, W] uint8 ──────────────────────────────────────
        video = torch.from_numpy(imgs.astype(np.uint8)).permute(3, 0, 1, 2)  # [3, T, H, W]

        if self.video_size is not None:
            tgt_h, tgt_w = self.video_size
            if video.shape[2] != tgt_h or video.shape[3] != tgt_w:
                video = video.permute(1, 0, 2, 3)
                video = torch.stack(
                    [TF.resize(video[t], [tgt_h, tgt_w], antialias=True) for t in range(video.shape[0])]
                )
                video = video.permute(1, 0, 2, 3)

        # ── action with normalised delay → [1, 8] float32 ────────────────────
        d_norm = float(d) / float(max(self.max_delay - 1, 1))
        action_vec = np.concatenate([
            act.astype(np.float32),
            np.array([d_norm], dtype=np.float32),
        ])
        action = torch.from_numpy(action_vec).unsqueeze(0)  # [1, 8]

        # ── metadata ─────────────────────────────────────────────────────────
        task_name = hdf5_path.stem.replace("_demo", "")
        cam_tag = "dual" if self.dual_camera else self.camera_key[:3]
        key = f"{hdf5_path.parent.name}/{task_name}/{demo_key}/t{start_t:04d}_d{d}_{cam_tag}"
        caption = task_name.replace("_", " ")

        return {
            "video": video,                                    # [3, T, H, W] uint8
            "action": action,                                  # [1, 8] float32
            "__key__": key,
            "fps": 10,
            "image_size": 256 * torch.ones(4).cuda(),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(1, 256, 256).cuda(),
            "ai_caption": caption,
        }

    def _load_single_camera(
        self, hdf5_path: pathlib.Path, demo_key: str, start_t: int, d: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load single-camera 5-frame window → (imgs [5,H,W,3], act [7]).

        Layout: [cam_t, cam_{t+d+1}×4]
        """
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            dones = demo["dones"][start_t : start_t + d + 1]
            if np.any(dones):
                raise _EpisodeBoundaryError()
            cond_img = demo["obs"][self.camera_key][start_t]          # [H, W, 3]
            pred_img  = demo["obs"][self.camera_key][start_t + d + 1] # [H, W, 3]
            act = demo["actions"][start_t + d]                         # [7]

        imgs = np.stack(
            [cond_img, pred_img, pred_img, pred_img, pred_img], axis=0
        )  # [5, H, W, 3]
        return imgs, act

    def _load_dual_camera(
        self, hdf5_path: pathlib.Path, demo_key: str, start_t: int, d: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load dual-camera 13-frame video → (imgs [13,H,W,3], act [7]).

        Layout (state_t=4, num_conditional_frames=2):
          Frame 0      : cam1_t                        (conditioning latent 1)
          Frames 1–4   : [cam2_t, zeros, zeros, zeros] (conditioning latent 2)
          Frames 5–8   : cam1_{t+d+1} ×4              (predicted latent 1)
          Frames 9–12  : cam2_{t+d+1} ×4              (predicted latent 2)
        """
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            dones = demo["dones"][start_t : start_t + d + 1]
            if np.any(dones):
                raise _EpisodeBoundaryError()

            H = demo["obs"][_CAM1_KEY].shape[1]
            W = demo["obs"][_CAM1_KEY].shape[2]

            # Conditioning frames at t
            cam1_t = demo["obs"][_CAM1_KEY][start_t]             # [H, W, 3]
            cam2_t = demo["obs"][_CAM2_KEY][start_t]             # [H, W, 3]

            # Single predicted frame at t+d+1
            cam1_pred = demo["obs"][_CAM1_KEY][start_t + d + 1]  # [H, W, 3]
            cam2_pred = demo["obs"][_CAM2_KEY][start_t + d + 1]  # [H, W, 3]

            act = demo["actions"][start_t + d]                    # [7]

        blank = np.zeros((H, W, 3), dtype=np.uint8)

        imgs = np.concatenate([
            cam1_t[np.newaxis],                                             # frame 0
            cam2_t[np.newaxis],                                             # frame 1
            np.stack([blank, blank, blank], axis=0),                        # frames 2-4
            np.stack([cam1_pred, cam1_pred, cam1_pred, cam1_pred], axis=0), # frames 5-8
            np.stack([cam2_pred, cam2_pred, cam2_pred, cam2_pred], axis=0), # frames 9-12
        ], axis=0)  # [13, H, W, 3]
        return imgs, act


class _EpisodeBoundaryError(Exception):
    """Raised when a sampled window crosses an episode boundary."""
