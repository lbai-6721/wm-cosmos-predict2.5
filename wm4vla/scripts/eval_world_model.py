#!/usr/bin/env python3
"""eval_world_model.py

离线评估训练好的 Libero-Spatial World Model（model_ema_bf16.pt）。

评估协议：
  - 数据集：val split（seed=0, val_ratio=0.1），与训练时完全相同的分割
  - 支持按 task_indices 过滤，用于单任务快速评估
  - 固定 d ∈ {1, 2, 3, 4}，各独立统计一组指标
  - 每个 val episode 随机抽取 20 个 start_t
  - 指标：PSNR、SSIM（cam1 + cam2 分别统计，再求平均）
  - 可选：LPIPS（需要 pip install lpips）

视频帧布局（强配对 batch，5 帧，state_t=2）：
  Batch 0:
    Frame 0    : cam1_t          ← conditioning latent 0（第三视角）
    Frames 1–4 : zeros           ← 待预测 cam1（latent 1）
  Batch 1:
    Frame 0    : cam2_t          ← conditioning latent 0（腕部）
    Frames 1–4 : zeros           ← 待预测 cam2（latent 1）

文本条件：使用 tasks.jsonl 中的真实任务描述。
  若提供 --t5-emb-path，直接使用预计算嵌入（避免在线加载 T5-11B）；
  否则通过 Video2WorldInference 在线编码 prompt 字符串。

用法：
  # 评估单任务（task 0）
  python scripts/eval_world_model.py \\
      --ckpt /path/to/model_ema_bf16.pt \\
      --task-indices 0 \\
      --experiment ac_libero_lerobot_256_pixels_2b_task0 \\
      [--t5-emb-path lerobot/.../meta/t5_embeddings.pkl] \\
      [--num-steps 10] \\
      [--samples-per-episode 20] \\
      [--output outputs/eval_wm/task0.json]

CUDA_VISIBLE_DEVICES=0 python scripts/eval_world_model.py \
    --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_libero_10_lerobot_256_skip_dynamics_dual_cam_task0/checkpoints/iter_000008000/model_ema_bf16.pt \
    --task-indices 0 \
    --experiment ac_libero_lerobot_256_pixels_2b_task0 \
    --t5-emb-path /home/kyji/public/dataset/lerobot/lerobot--libero_10_image@v2.0/meta/t5_embeddings.pkl \
    --num-steps 1 \
    --tokenizer-backend lightvae \
    --tokenizer-vae-pth /home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth \
    --save-images outputs/eval_wm_test/test_time_new/test_time_1_20260416/libero-10_task0_images \
    --output outputs/eval_wm_test/test_time_new/test_time_1_20260416/libero-10_task0.json

  # 评估全部 10 个任务
  python scripts/eval_world_model.py \\
      --ckpt /path/to/model_ema_bf16.pt \\
      --experiment ac_libero_lerobot_256_pixels_2b \\
      [--output outputs/eval_wm/all_tasks.json]

python scripts/eval_world_model.py \
    --ckpt .../model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 35 \
    --output outputs/eval_wm/task0.json


CUDA_VISIBLE_DEVICES=1 python scripts/eval_world_model.py \
    --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/single/libero-spatial_task0/6000/libero-10_task01/10000/model_ema_bf16.pt \
    --task-indices 0 1 \
    --experiment ac_libero_lerobot_256_pixels_2b_task01 \
    --t5-emb-path /home/kyji/public/dataset/lerobot/lerobot--libero_10_image@v2.0/meta/t5_embeddings.pkl \
    --output outputs/eval_wm/task01.json


注意：需要在 cosmos-predict2.5 根目录下执行，或者把根目录加入 PYTHONPATH。
"""

import argparse
import io
import json
import os
import pathlib
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wm4vla.conditioning import normalize_delay_scalar, pack_masked_action_sequence
from wm4vla.configs.wm_conditioning import (
    BYPASS_WM_WHEN_DELAY_ZERO,
    DEFAULT_EVAL_DELAYS,
    INFER_DELAY_MAX,
)

# ── 可选依赖（不影响主要指标）──────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    print("[warn] skimage not found; SSIM will be skipped. pip install scikit-image")

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False
    print("[warn] lpips not found; LPIPS will be skipped. pip install lpips")


# ── 常量（与训练/数据集一致）───────────────────────────────────────────────
_CAM1_KEY = "observation.images.image"
_CAM2_KEY = "observation.images.wrist_image"
_ACT_KEY  = "action"

_DEFAULT_DATA_ROOT = (
    "/home/kyji/public/dataset/lerobot"
    "/lerobot--libero_10_image@v2.0"
)
_EXPERIMENT_NAME_FULL   = "ac_libero_lerobot_256_pixels_2b"
_EXPERIMENT_NAME_TASK0  = "ac_libero_lerobot_256_pixels_2b_task0"
_EXPERIMENT_NAME_TASK01 = "ac_libero_lerobot_256_pixels_2b_task01"
_CONFIG_FILE      = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
_NUM_LATENT_COND  = 1    # state_t=2 → 1 conditioning latent frame
_NUM_VIDEO_FRAMES = 5    # 1 + (2-1)×4 = 5 pixel frames
_RESOLUTION       = "256,256"
_MAX_DELAY        = INFER_DELAY_MAX


# ══════════════════════════════════════════════════════════════════════════════
# 图像解码 / 编码工具
# ══════════════════════════════════════════════════════════════════════════════

def _decode_image(cell) -> np.ndarray:
    """LeRobot v2.0 parquet 图像列 → (H, W, 3) uint8。"""
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, (bytes, bytearray)):
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    raise TypeError(f"Unexpected image cell type: {type(cell)}")


def _uint8_to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [0,1]。"""
    return img.astype(np.float32) / 255.0


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [-1, 1] → (H, W, 3) uint8。"""
    img = ((t.float() + 1.0) / 2.0).clamp(0, 1)  # → [0, 1]
    img = (img * 255.0).to(torch.uint8)
    return img.permute(1, 2, 0).cpu().numpy()  # → (H, W, 3)


def build_action_inputs(action_rows: np.ndarray, delay: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the action prefix and global delay scalar for inference."""
    packed = pack_masked_action_sequence(
        actions=action_rows,
        delay=delay,
        chunk_len=_MAX_DELAY,
    )
    delay_scalar = normalize_delay_scalar(delay=delay, max_delay=_MAX_DELAY)
    action = torch.from_numpy(packed).float().unsqueeze(0).repeat(batch_size, 1, 1)
    delay_tensor = torch.from_numpy(delay_scalar).float().unsqueeze(0).repeat(batch_size, 1)
    return action, delay_tensor


# ══════════════════════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """(H, W, 3) uint8 → PSNR (dB)。"""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """(H, W, 3) uint8 → SSIM ∈ [0, 1]。需要 skimage。"""
    if not _HAS_SKIMAGE:
        return float("nan")
    return float(_ssim_fn(pred, gt, channel_axis=-1, data_range=255))


# ══════════════════════════════════════════════════════════════════════════════
# Val 数据集采样（与训练一致）
# ══════════════════════════════════════════════════════════════════════════════

def load_task_descriptions(data_root: str) -> Dict[int, str]:
    """
    从 tasks.jsonl 读取 task_index → language_instruction 映射。

    Returns:
        {task_index: description_string}
    """
    tasks_file = pathlib.Path(data_root) / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        print(f"[warn] tasks.jsonl not found at {tasks_file}. Will use generic prompts.")
        return {}
    task_map: Dict[int, str] = {}
    with open(tasks_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj.get("task_index", obj.get("index", -1)))
            desc = obj.get("task", obj.get("language_instruction", obj.get("description", "")))
            if idx >= 0 and desc:
                task_map[idx] = desc
    print(f"[data] Loaded {len(task_map)} task descriptions from tasks.jsonl")
    return task_map


def load_t5_embeddings(t5_emb_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    加载预计算的 T5 embeddings（pkl 格式）。

    Returns:
        {task_index_str: tensor (512, 1024)} 或 None（若文件不存在）
    """
    p = pathlib.Path(t5_emb_path)
    if not p.exists():
        print(f"[warn] T5 embedding file not found: {p}. Will use online encoding.")
        return None
    with open(p, "rb") as f:
        emb_dict = pickle.load(f)
    print(f"[data] Loaded T5 embeddings for {len(emb_dict)} tasks from {p}")
    return emb_dict


def build_val_episode_samples(
    data_root: str,
    val_ratio: float = 0.1,
    split_seed: int = 0,
    max_delay: int = 5,
    samples_per_episode: int = 20,
    sample_seed: int = 42,
    task_indices: Optional[Sequence[int]] = None,
) -> List[tuple]:
    """
    复现训练时的 train/val episode 分割，返回 val 集。

    Args:
        task_indices: 若指定，只保留 task_index 在此列表中的 episode（单任务评估）。
                      None 表示使用全部 val episodes。

    Returns:
        list of (parquet_file_path, task_index, [start_t, ...]) per episode
    """
    data_dir = pathlib.Path(data_root) / "data" / "chunk-000"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("episode_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No episode parquet files in {data_dir}")

    # ── 与 dataset_lerobot_libero.py 完全相同的分割逻辑 ────────────────────
    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * val_ratio))
    val_set = set(perm[:n_val].tolist())

    val_files = [all_files[i] for i in range(len(all_files)) if i in val_set]
    print(f"[data] Total episodes: {len(all_files)}, val episodes: {len(val_files)}")

    # ── 可选：按 task_indices 过滤（复现 dataset_lerobot_libero.py 行为）──
    if task_indices is not None:
        task_indices_set = set(int(t) for t in task_indices)
        filtered = []
        for fp in val_files:
            try:
                df_task = pd.read_parquet(fp, columns=["task_index"])
                ep_task = int(df_task["task_index"].iloc[0])
                if ep_task in task_indices_set:
                    filtered.append(fp)
            except Exception as e:
                print(f"[warn] Failed to read task_index from {fp.name}: {e}")
        print(f"[data] After filtering task_indices={list(task_indices_set)}: "
              f"{len(filtered)}/{len(val_files)} val episodes kept")
        val_files = filtered

    # ── 对每个 val episode 构建合法 start_t 列表，随机采样 ──────────────────
    rng_sample = random.Random(sample_seed)
    result = []
    for file_path in sorted(val_files):
        df_meta = pd.read_parquet(file_path, columns=["frame_index", "task_index"])
        T = len(df_meta)
        ep_task_index = int(df_meta["task_index"].iloc[0])
        # 合法 start_t：start_t + max_delay - 1 < T  →  start_t ≤ T - max_delay
        valid_starts = list(range(T - max_delay))
        if not valid_starts:
            print(f"[warn] Episode {file_path.name} too short (T={T}), skipped")
            continue
        chosen = rng_sample.sample(valid_starts, min(samples_per_episode, len(valid_starts)))
        result.append((file_path, ep_task_index, sorted(chosen)))

    total_windows = sum(len(s) for _, _, s in result)
    print(f"[data] {len(result)} val episodes, {total_windows} sampled windows "
          f"(×4 delays = {total_windows * 4} inference calls)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 构建 WM 输入视频张量（强配对 batch，5 帧布局，state_t=2）
# ══════════════════════════════════════════════════════════════════════════════

def build_input_video(cam1_t: np.ndarray, cam2_t: np.ndarray) -> torch.Tensor:
    """
    构建 WM 输入的强配对 5 帧视频张量（uint8，值域 [0, 255]）。

    帧布局（与 dataset_lerobot_libero.py 完全一致，state_t=2）：
      Batch 0:
        Frame 0    : cam1_t    ← conditioning latent 0
        Frames 1–4 : zeros     ← 待预测 cam1_{t+d+1}（latent 1）
      Batch 1:
        Frame 0    : cam2_t    ← conditioning latent 0
        Frames 1–4 : zeros     ← 待预测 cam2_{t+d+1}（latent 1）

    Returns:
        [2, 3, 5, H, W] uint8 tensor（CPU）
    """
    H, W = cam1_t.shape[:2]
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    cam1_frames = np.stack([cam1_t, blank, blank, blank, blank], axis=0)
    cam2_frames = np.stack([cam2_t, blank, blank, blank, blank], axis=0)
    paired_frames = np.stack([cam1_frames, cam2_frames], axis=0)  # [2, 5, H, W, 3]

    # → [2, 3, 5, H, W] uint8
    video = torch.from_numpy(paired_frames).permute(0, 4, 1, 2, 3)
    return video


# ══════════════════════════════════════════════════════════════════════════════
# 主评估流程
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 解析评估模式 ──────────────────────────────────────────────────────────
    task_indices: Optional[List[int]] = (
        [int(t) for t in args.task_indices] if args.task_indices else None
    )
    if args.experiment:
        experiment_name = args.experiment
    elif task_indices is None:
        experiment_name = _EXPERIMENT_NAME_FULL
    elif sorted(task_indices) == [0, 1]:
        experiment_name = _EXPERIMENT_NAME_TASK01
    elif sorted(task_indices) == [0]:
        experiment_name = _EXPERIMENT_NAME_TASK0
    else:
        # 其他任意子集：架构与全集相同，直接用全集实验配置
        experiment_name = _EXPERIMENT_NAME_FULL

    # ── 1. 加载 WM ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("加载 World Model …")
    print(f"  ckpt        : {args.ckpt}")
    print(f"  experiment  : {experiment_name}")
    print(f"  task_indices: {task_indices if task_indices is not None else 'all'}")
    print(f"{'='*60}\n")

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    experiment_opts: List[str] = []
    tokenizer_backend = "lightvae" if args.use_lightvae else args.tokenizer_backend
    tokenizer_vae_pth = args.tokenizer_vae_pth
    if tokenizer_backend == "lightvae":
        tokenizer_vae_pth = tokenizer_vae_pth or args.lightvae_pth
        experiment_opts.extend(
            [
                "tokenizer=wan2pt1_lightvae_tokenizer",
                f"model.config.tokenizer.vae_pth={tokenizer_vae_pth}",
            ]
        )
        if args.lightx2v_root:
            experiment_opts.append(f"+model.config.tokenizer.lightx2v_root={args.lightx2v_root}")
        print(f"[info] Tokenizer backend=lightvae, overrides: {experiment_opts}")
    else:
        if args.lightx2v_root:
            print("[warn] --lightx2v-root is ignored when tokenizer backend is wan2pt1")
        if tokenizer_vae_pth:
            experiment_opts.extend(
                [
                    "tokenizer=wan2pt1_tokenizer",
                    f"model.config.tokenizer.vae_pth={tokenizer_vae_pth}",
                ]
            )
            print(f"[info] Tokenizer backend=wan2pt1, overrides: {experiment_opts}")

    wm = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=str(args.ckpt),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=_CONFIG_FILE,
        experiment_opts=experiment_opts,
    )
    wm.model.eval()

    # ── 2. 可选：LPIPS 损失网络 ────────────────────────────────────────────
    lpips_fn = None
    if _HAS_LPIPS:
        lpips_fn = _lpips_lib.LPIPS(net="alex").cuda().eval()
        print("[info] LPIPS (AlexNet) loaded.")

    # ── 3. 加载任务描述 & T5 嵌入 ────────────────────────────────────────────
    data_root = args.data_root or _DEFAULT_DATA_ROOT
    task_descriptions = load_task_descriptions(data_root)

    # generate_vid2world 内部调用 get_text_embedding(prompt)（全局单例）。
    # 优先使用预计算 pkl：直接查表返回 tensor，完全不需要加载 T5-11B 模型（42 GB）。
    # 若未提供 --t5-emb-path，则回退到加载本地/在线 T5-11B。
    import cosmos_predict2._src.predict2.inference.get_t5_emb as _t5_module
    from cosmos_predict2._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder

    # 加载预计算嵌入 pkl：key = 任务描述字符串, value = tensor (1, 512, 1024)
    t5_emb_by_str: Dict[str, torch.Tensor] = {}
    if args.t5_emb_path:
        emb_dict = load_t5_embeddings(args.t5_emb_path)
        if emb_dict is not None:
            for k, v in emb_dict.items():
                t5_emb_by_str[str(k)] = torch.as_tensor(v, dtype=torch.float32)
            print(f"[info] 预计算 T5 嵌入已加载：{len(t5_emb_by_str)} 个任务")

    if t5_emb_by_str:
        # 用预计算缓存伪装成 CosmosT5TextEncoder，generate_vid2world 无感知
        class _CachedT5Encoder:
            """从预计算 pkl 查表，跳过在线 T5 编码。"""
            def __init__(self, cache: Dict[str, torch.Tensor]):
                self._cache = cache
                self.text_encoder = self   # 满足 offload 代码路径

            def encode_prompts(self, prompts, max_length=512, return_mask=False):
                if isinstance(prompts, str):
                    prompts = [prompts]
                embeddings = []
                for key in prompts:
                    if key not in self._cache:
                        emb = torch.zeros(1, max_length, 1024, dtype=torch.float32, device="cuda")
                    else:
                        emb = self._cache[key].to("cuda")
                    embeddings.append(emb.squeeze(0))
                emb = torch.stack(embeddings, dim=0)
                if return_mask:
                    mask = torch.ones(emb.shape[:2], dtype=torch.bool, device="cuda")
                    return emb, mask
                return emb

            # offload 相关方法（空实现，不报错）
            def to(self, device): return self
            def __call__(self, *a, **kw): return self.encode_prompts(*a, **kw)

        _t5_module.cosmos_encoder = _CachedT5Encoder(t5_emb_by_str)
        print("[info] 已用预计算嵌入替换全局 T5 编码器，推理不需加载 T5-11B")
    else:
        # 没有预计算嵌入：回退加载 T5-11B（本地路径优先）
        if _t5_module.cosmos_encoder is None:
            t5_model_path = getattr(args, "t5_model", None)
            local_path = pathlib.Path(t5_model_path) if t5_model_path else None
            if local_path is not None and local_path.is_dir():
                print(f"[info] 加载 T5-11B（本地）: {local_path}")
                _t5_module.cosmos_encoder = CosmosT5TextEncoder(
                    model_name=str(local_path), device="cuda", local_files_only=True
                )
            else:
                print("[info] 加载 T5-11B（在线）: google-t5/t5-11b …")
                _t5_module.cosmos_encoder = CosmosT5TextEncoder(
                    model_name="google-t5/t5-11b", device="cuda"
                )
        print("[info] T5-11B 编码器就绪，推理时将对 prompt 在线编码。")

    # ── 4. 构建 val 数据集采样表 ───────────────────────────────────────────
    episode_samples = build_val_episode_samples(
        data_root=data_root,
        val_ratio=0.1,
        split_seed=0,           # 与训练一致，固定！
        max_delay=_MAX_DELAY,
        samples_per_episode=args.samples_per_episode,
        sample_seed=args.seed,
        task_indices=task_indices,
    )

    # ── 5. 初始化指标容器 ─────────────────────────────────────────────────
    delays = [d for d in DEFAULT_EVAL_DELAYS if 1 <= d <= 4]
    # 汇总指标：results[d]
    results = {d: defaultdict(list) for d in delays}
    # 按任务细分：results_per_task[task_idx][d]（多任务评估时展示各任务独立指标）
    all_task_indices_seen: set = set()
    results_per_task: Dict[int, Dict[int, defaultdict]] = {}

    # 可选：保存预测帧图片
    if args.save_images:
        pathlib.Path(args.save_images).mkdir(parents=True, exist_ok=True)

    # ── 6. 评估主循环 ──────────────────────────────────────────────────────
    total_windows = sum(len(s) for _, _, s in episode_samples)
    done = 0
    t_start = time.time()

    print(
        f"\n开始评估，共 {total_windows} 个 (episode, start_t) × {len(delays)} delays = "
        f"{total_windows * len(delays)} 次推理 …\n"
    )

    for ep_idx, (file_path, ep_task_index, start_ts) in enumerate(episode_samples):
        ep_name = file_path.stem  # e.g. "episode_000042"

        # 确定 prompt（real task description from tasks.jsonl）
        # generate_vid2world 内部使用全局 cosmos_encoder 对 prompt 编码
        prompt = task_descriptions.get(
            ep_task_index, f"libero task {ep_task_index}"
        )

        # 初始化该 task 的指标容器（首次出现时）
        if ep_task_index not in results_per_task:
            results_per_task[ep_task_index] = {d: defaultdict(list) for d in delays}
        all_task_indices_seen.add(ep_task_index)

        # 读取整个 episode（~10-15 MB per parquet）
        df = pd.read_parquet(file_path)
        T = len(df)

        for start_t in start_ts:
            # 解码当前帧
            cam1_t = _decode_image(df[_CAM1_KEY].iloc[start_t])
            cam2_t = _decode_image(df[_CAM2_KEY].iloc[start_t])

            # 构建 WM 输入（一次构建，4 个 d 复用）
            vid_input = build_input_video(cam1_t, cam2_t)

            for d in delays:
                pred_t = start_t + d
                if pred_t >= T:
                    # 理论上 valid_starts 已保证合法，双重保险
                    continue

                # GT 帧
                cam1_gt = _decode_image(df[_CAM1_KEY].iloc[pred_t])
                cam2_gt = _decode_image(df[_CAM2_KEY].iloc[pred_t])

                if d == 0 and BYPASS_WM_WHEN_DELAY_ZERO:
                    cam1_pred_t = cam1_t.copy()
                    cam2_pred_t = cam2_t.copy()
                    video_out = None
                else:
                    action_rows = np.stack(
                        [np.asarray(df[_ACT_KEY].iloc[start_t + offset], dtype=np.float32) for offset in range(d)],
                        axis=0,
                    )
                    action, delay_scalar = build_action_inputs(action_rows, delay=d, batch_size=2)

                    time_video_start = time.time()
                    with torch.no_grad():
                        video_out = wm.generate_vid2world(
                            prompt=[prompt, prompt],
                            input_path=vid_input,
                            action=action,
                            delay_scalar=delay_scalar,
                            guidance=args.guidance,
                            num_video_frames=_NUM_VIDEO_FRAMES,
                            num_latent_conditional_frames=_NUM_LATENT_COND,
                            resolution=_RESOLUTION,
                            seed=args.seed,
                            num_steps=args.num_steps,
                        )
                    time_video_end = time.time()
                    print(f"Time taken for video generation: {time_video_end - time_video_start} seconds")
                    cam1_pred_t = _tensor_to_uint8(video_out[0, :, 1])
                    cam2_pred_t = _tensor_to_uint8(video_out[1, :, 1])

                # ── 计算指标 ────────────────────────────────────────────
                psnr1 = compute_psnr(cam1_pred_t, cam1_gt)
                psnr2 = compute_psnr(cam2_pred_t, cam2_gt)
                results[d]["cam1_psnr"].append(psnr1)
                results[d]["cam2_psnr"].append(psnr2)
                results_per_task[ep_task_index][d]["cam1_psnr"].append(psnr1)
                results_per_task[ep_task_index][d]["cam2_psnr"].append(psnr2)

                ssim1 = compute_ssim(cam1_pred_t, cam1_gt)
                ssim2 = compute_ssim(cam2_pred_t, cam2_gt)
                results[d]["cam1_ssim"].append(ssim1)
                results[d]["cam2_ssim"].append(ssim2)
                results_per_task[ep_task_index][d]["cam1_ssim"].append(ssim1)
                results_per_task[ep_task_index][d]["cam2_ssim"].append(ssim2)

                if lpips_fn is not None:
                    with torch.no_grad():
                        # LPIPS 输入：[-1, 1] float，(1, 3, H, W)
                        t1_pred = video_out[0:1, :, 1].float()
                        t2_pred = video_out[1:2, :, 1].float()
                        t1_gt = (torch.from_numpy(cam1_gt).float() / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).cuda()
                        t2_gt = (torch.from_numpy(cam2_gt).float() / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).cuda()
                        lp1 = float(lpips_fn(t1_pred, t1_gt).item())
                        lp2 = float(lpips_fn(t2_pred, t2_gt).item())
                    results[d]["cam1_lpips"].append(lp1)
                    results[d]["cam2_lpips"].append(lp2)
                    results_per_task[ep_task_index][d]["cam1_lpips"].append(lp1)
                    results_per_task[ep_task_index][d]["cam2_lpips"].append(lp2)

                # ── 可选：保存预测 vs GT 对比图 ─────────────────────────
                if args.save_images:
                    tag = f"{ep_name}_t{start_t:04d}_d{d}"
                    Image.fromarray(np.hstack([cam1_pred_t, cam1_gt])).save(
                        f"{args.save_images}/{tag}_cam1_pred_vs_gt.png"
                    )
                    Image.fromarray(np.hstack([cam2_pred_t, cam2_gt])).save(
                        f"{args.save_images}/{tag}_cam2_pred_vs_gt.png"
                    )

            done += 1
            if done % 10 == 0 or done == total_windows:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_windows - done)
                print(f"  [{done}/{total_windows}] elapsed={elapsed/60:.1f}min  "
                      f"ETA={eta/60:.1f}min")

    # ── 辅助函数：从指标字典计算一行汇总 ─────────────────────────────────────
    def _summarize(r: defaultdict) -> dict:
        n = len(r["cam1_psnr"])
        if n == 0:
            return {}
        c1p = float(np.nanmean(r["cam1_psnr"]))
        c2p = float(np.nanmean(r["cam2_psnr"]))
        s = {
            "n": n,
            "cam1_psnr": round(c1p, 4),
            "cam2_psnr": round(c2p, 4),
            "avg_psnr":  round((c1p + c2p) / 2, 4),
        }
        if _HAS_SKIMAGE and r["cam1_ssim"]:
            c1s = float(np.nanmean(r["cam1_ssim"]))
            c2s = float(np.nanmean(r["cam2_ssim"]))
            s.update({
                "cam1_ssim": round(c1s, 4),
                "cam2_ssim": round(c2s, 4),
                "avg_ssim":  round((c1s + c2s) / 2, 4),
            })
        if _HAS_LPIPS and r["cam1_lpips"]:
            s.update({
                "cam1_lpips": round(float(np.nanmean(r["cam1_lpips"])), 4),
                "cam2_lpips": round(float(np.nanmean(r["cam2_lpips"])), 4),
            })
        return s

    def _print_table(title: str, results_dict: Dict[int, defaultdict]):
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")
        header = f"{'d':>4} | {'cam1_PSNR':>10} {'cam2_PSNR':>10} {'avg_PSNR':>10}"
        if _HAS_SKIMAGE:
            header += f" | {'cam1_SSIM':>10} {'cam2_SSIM':>10} {'avg_SSIM':>10}"
        if _HAS_LPIPS:
            header += f" | {'cam1_LPIPS':>10} {'cam2_LPIPS':>10}"
        header += f" | {'N':>5}"
        print(header)
        print("-" * len(header))
        for d in delays:
            r = results_dict[d]
            s = _summarize(r)
            if not s:
                continue
            row = (f"{d:>4} | {s['cam1_psnr']:>10.3f} {s['cam2_psnr']:>10.3f} "
                   f"{s['avg_psnr']:>10.3f}")
            if _HAS_SKIMAGE and "cam1_ssim" in s:
                row += (f" | {s['cam1_ssim']:>10.4f} {s['cam2_ssim']:>10.4f} "
                        f"{s['avg_ssim']:>10.4f}")
            if _HAS_LPIPS and "cam1_lpips" in s:
                row += f" | {s['cam1_lpips']:>10.4f} {s['cam2_lpips']:>10.4f}"
            row += f" | {s['n']:>5}"
            print(row)

    # ── 6. 汇总并打印结果 ─────────────────────────────────────────────────
    summary = {}

    # 6a. 整体汇总（所有 task 合并）
    _print_table("评估结果（所有 task 合并，按 delay 分组）", results)
    for d in delays:
        s = _summarize(results[d])
        if s:
            summary[f"d={d}"] = s

    # 6b. 按 task 细分（多任务评估时显示）
    per_task_summary: Dict[str, dict] = {}
    if len(all_task_indices_seen) > 1:
        for tidx in sorted(all_task_indices_seen):
            desc = task_descriptions.get(tidx, f"task {tidx}")
            _print_table(
                f"Task {tidx}  [{desc[:50]}{'…' if len(desc) > 50 else ''}]",
                results_per_task[tidx],
            )
            task_key = f"task{tidx}"
            per_task_summary[task_key] = {}
            for d in delays:
                s = _summarize(results_per_task[tidx][d])
                if s:
                    per_task_summary[task_key][f"d={d}"] = s

    print(f"\n{'='*60}")
    print(f"总耗时: {(time.time()-t_start)/60:.1f} min")

    # ── 7. 保存 JSON 结果 ─────────────────────────────────────────────────
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "ckpt": str(args.ckpt),
            "data_root": str(data_root),
            "experiment": experiment_name,
            "task_indices": task_indices,
            "val_ratio": 0.1,
            "split_seed": 0,
            "sample_seed": args.seed,
            "samples_per_episode": args.samples_per_episode,
            "num_steps": args.num_steps,
            "guidance": args.guidance,
            "t5_emb_path": str(args.t5_emb_path) if args.t5_emb_path else None,
            "tokenizer_backend": tokenizer_backend,
            "tokenizer_vae_pth": str(tokenizer_vae_pth) if tokenizer_vae_pth else None,
            "lightx2v_root": str(args.lightx2v_root) if tokenizer_backend == "lightvae" and args.lightx2v_root else None,
            "delays_evaluated": delays,
        }
        out = {"meta": meta, "results": summary}
        if per_task_summary:
            out["results_per_task"] = per_task_summary
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {out_path}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline eval of Libero-Spatial World Model (skip-dynamics, paired 5-frame layout)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt", type=pathlib.Path, required=True,
        help="Path to model_ema_bf16.pt (or model_ema_fp32.pt)",
    )
    p.add_argument(
        "--data-root", type=str, default=None,
        help=f"Override data root. Default: {_DEFAULT_DATA_ROOT}",
    )
    p.add_argument(
        "--experiment", type=str, default=None,
        help=(
            "Experiment name. Defaults to ac_libero_lerobot_256_pixels_2b_task0 "
            "when --task-indices is given, else ac_libero_lerobot_256_pixels_2b."
        ),
    )
    p.add_argument(
        "--task-indices", type=int, nargs="+", default=None, metavar="TASK_IDX",
        help=(
            "Evaluate only these task indices (e.g. --task-indices 0). "
            "Omit to evaluate all tasks."
        ),
    )
    p.add_argument(
        "--t5-emb-path", type=str, default=None,
        help=(
            "Path to pre-computed T5 embedding pkl file "
            "(generated by scripts/precompute_libero_t5.py). "
            "If not given, prompt strings are encoded online by Video2WorldInference."
        ),
    )
    p.add_argument(
        "--num-steps", type=int, default=35,
        help="Diffusion denoising steps (reduce to 10-15 for speed)",
    )
    p.add_argument(
        "--guidance", type=float, default=0.0,
        help="Classifier-free guidance scale (0.0 = no CFG, ~2x faster)",
    )
    p.add_argument(
        "--samples-per-episode", type=int, default=20,
        help="Number of start_t to sample per val episode",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for start_t sampling (split seed is always 0)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (e.g. outputs/eval_wm/task0.json)",
    )
    p.add_argument(
        "--save-images", type=str, default=None,
        help="Directory to save pred vs GT image pairs (optional)",
    )
    p.add_argument(
        "--tokenizer-backend", type=str, choices=["wan2pt1", "lightvae"], default="wan2pt1",
        help="Tokenizer backend for evaluation inference.",
    )
    p.add_argument(
        "--tokenizer-vae-pth", type=str, default="",
        help=(
            "Optional VAE checkpoint path override. "
            "Works for both backends: wan2pt1/lightvae."
        ),
    )
    p.add_argument(
        "--use-lightvae", action="store_true",
        help="Deprecated alias of --tokenizer-backend lightvae.",
    )
    p.add_argument(
        "--lightvae-pth", type=str, default="/home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth",
        help="LightVAE checkpoint path when --use-lightvae is enabled.",
    )
    p.add_argument(
        "--lightx2v-root", type=str, default="",
        help="Optional LightX2V repo root (used when lightx2v is not importable from PYTHONPATH).",
    )
    return p.parse_args()


if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    args = parse_args()
    evaluate(args)
