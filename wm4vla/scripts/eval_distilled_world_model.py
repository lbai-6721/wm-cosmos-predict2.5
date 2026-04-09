#!/usr/bin/env python3
"""eval_distilled_world_model.py

离线评估 DMD2 蒸馏后的 Action-Conditioned World Model（LIBERO task0）。

与 eval_world_model.py 的主要区别：
  - 使用 DMD2 蒸馏实验配置（registry_predict2p5.py）加载模型
  - 支持多个推理步数（--num-steps 1 2 4），每组独立统计指标
  - 直接调用 model.generate_samples_from_batch(num_steps=N) 做少步推理
  - 不需要 CFG（蒸馏模型 teacher_guidance=0）
  - 指标：PSNR、SSIM（cam1 + cam2）、可选 LPIPS

视频帧布局（strong paired batch，5 帧，state_t=2）：
  Batch 0:
    Frame 0    : cam1_t          ← conditioning latent 0（第三视角）
    Frames 1–4 : zeros           ← 待预测 cam1（latent 1）
  Batch 1:
    Frame 0    : cam2_t          ← conditioning latent 0（腕部）
    Frames 1–4 : zeros           ← 待预测 cam2（latent 1）

用法示例：

  # 测试 1、2、4 步，自动生成三组结果
  CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \\
      --ckpt /path/to/distilled/model_ema_bf16.pt \\
      --task-indices 0 \\
      --num-steps 1 2 4 \\
      --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \\
      --output outputs/eval_distill/task0_steps124.json

  # 只测 1 步（快速验证）
  CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \\
      --ckpt /path/to/distilled/model_ema_bf16.pt \\
      --task-indices 0 \\
      --num-steps 1 \\
      --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \\
      --output outputs/eval_distill/task0_step1.json

  # 保存预测 vs GT 对比图（仅针对第一个 num_steps）
  CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \\
      --ckpt /path/to/distilled/model_ema_bf16.pt \\
      --task-indices 0 \\
      --num-steps 4 \\
      --save-images outputs/eval_distill/images_step4
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
from wm4vla.configs.wm_conditioning import INFER_DELAY_MAX

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
# 蒸馏实验名（对应 experiments_dmd2_ac_predict2p5.py 中注册的名称）
_DISTILL_EXPERIMENT_NAME = "dmd2_trigflow_distill_wm_libero_lerobot_256_task0"
_DISTILL_CONFIG_FILE = "cosmos_predict2/_src/interactive/configs/registry_predict2p5.py"

_NUM_LATENT_COND  = 1    # state_t=2 → 1 conditioning latent frames
_NUM_VIDEO_FRAMES = 5    # 1 + (2-1)×4 = 5 pixel frames
_RESOLUTION       = "256,256"
_MAX_DELAY        = INFER_DELAY_MAX


# ══════════════════════════════════════════════════════════════════════════════
# 图像解码 / 编码工具（与 eval_world_model.py 相同）
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
    img = ((t.float() + 1.0) / 2.0).clamp(0, 1)
    img = (img * 255.0).to(torch.uint8)
    return img.permute(1, 2, 0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 指标计算（与 eval_world_model.py 相同）
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
# 蒸馏模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_distilled_model(ckpt_path: str, experiment_name: str):
    """
    用 DMD2 配置加载蒸馏后的 student 模型。

    推理时不需要 teacher 和 fake_score 网络，节省显存。

    Args:
        ckpt_path: 本地 .pt checkpoint 路径（已通过 convert_distcp_to_pt.py 转换）
        experiment_name: DMD2 实验名（如 dmd2_trigflow_distill_wm_libero_lerobot_256_task0）

    Returns:
        model: 已加载 student EMA 权重的 DMD2Model，eval 模式
    """
    from cosmos_predict2._src.interactive.utils.model_loader import load_model_from_checkpoint

    print(f"\n{'='*60}")
    print("加载蒸馏模型 …")
    print(f"  ckpt        : {ckpt_path}")
    print(f"  experiment  : {experiment_name}")
    print(f"{'='*60}\n")

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment_name,
        s3_checkpoint_dir=str(ckpt_path),
        config_file=_DISTILL_CONFIG_FILE,
        load_ema_to_reg=True,
        skip_teacher_init=True,   # 推理不需要加载 teacher checkpoint
    )
    model.eval()

    # 推理不需要 fake_score 和 teacher，释放显存
    if hasattr(model, "net_fake_score") and model.net_fake_score is not None:
        del model.net_fake_score
        model.net_fake_score = None
        print("[info] net_fake_score 已释放（推理不需要）")
    if hasattr(model, "net_teacher") and model.net_teacher is not None:
        del model.net_teacher
        model.net_teacher = None
        print("[info] net_teacher 已释放（推理不需要）")

    torch.cuda.empty_cache()
    print("[info] 蒸馏模型加载完成。")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Val 数据集采样（与 eval_world_model.py 完全相同）
# ══════════════════════════════════════════════════════════════════════════════

def load_task_descriptions(data_root: str) -> Dict[int, str]:
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
    p = pathlib.Path(t5_emb_path)
    if not p.exists():
        print(f"[warn] T5 embedding file not found: {p}. Will use zero embeddings.")
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
    复现训练时的 train/val episode 分割，返回 val 集采样表。
    （与 eval_world_model.py 完全相同的逻辑）
    """
    data_dir = pathlib.Path(data_root) / "data" / "chunk-000"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("episode_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No episode parquet files in {data_dir}")

    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * val_ratio))
    val_set = set(perm[:n_val].tolist())

    val_files = [all_files[i] for i in range(len(all_files)) if i in val_set]
    print(f"[data] Total episodes: {len(all_files)}, val episodes: {len(val_files)}")

    if task_indices is not None:
        task_indices_set = set(task_indices)
        filtered = []
        for fp in val_files:
            try:
                df_meta = pd.read_parquet(fp, columns=["task_index"])
                task_idx = int(df_meta["task_index"].iloc[0])
                if task_idx in task_indices_set:
                    filtered.append(fp)
            except Exception:
                filtered.append(fp)
        val_files = filtered
        print(f"[data] After task filter ({task_indices}): {len(val_files)} episodes")

    rng_sample = np.random.default_rng(sample_seed)
    episode_samples = []
    for fp in val_files:
        try:
            df_meta = pd.read_parquet(fp, columns=["task_index"])
            task_idx = int(df_meta["task_index"].iloc[0])
        except Exception:
            task_idx = -1

        try:
            df_len = pd.read_parquet(fp, columns=[_ACT_KEY])
            T = len(df_len)
        except Exception:
            continue

        max_start = T - max_delay - 1
        if max_start < 0:
            continue

        n = min(samples_per_episode, max_start + 1)
        start_ts = sorted(rng_sample.choice(max_start + 1, size=n, replace=False).tolist())
        episode_samples.append((fp, task_idx, start_ts))

    return episode_samples


# ══════════════════════════════════════════════════════════════════════════════
# 输入视频构建（与 eval_world_model.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

def build_input_video(cam1_t: np.ndarray, cam2_t: np.ndarray) -> torch.Tensor:
    """
    构建 WM 输入的 paired 5 帧视频张量（uint8）。

    帧布局（state_t=2）：
      Batch 0:
        Frame 0    : cam1_t
        Frames 1–4 : zeros（待预测 cam1）
      Batch 1:
        Frame 0    : cam2_t
        Frames 1–4 : zeros（待预测 cam2）

    Returns:
        [2, 3, 5, H, W] uint8 tensor（CPU）
    """
    H, W = cam1_t.shape[:2]
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    cam1_frames = np.stack([cam1_t, blank, blank, blank, blank], axis=0)
    cam2_frames = np.stack([cam2_t, blank, blank, blank, blank], axis=0)
    paired_frames = np.stack([cam1_frames, cam2_frames], axis=0)  # [2, 5, H, W, 3]

    video = torch.from_numpy(paired_frames).permute(0, 4, 1, 2, 3)  # [2, 3, 5, H, W]
    return video


# ══════════════════════════════════════════════════════════════════════════════
# 数据批次构建（蒸馏模型专用）
# ══════════════════════════════════════════════════════════════════════════════

def build_data_batch(
    vid_input: torch.Tensor,
    t5_emb: torch.Tensor,
    action: torch.Tensor,
    delay_scalar: torch.Tensor,
) -> dict:
    """
    构建蒸馏模型推理所需的 data_batch。

    Args:
        vid_input: [2, 3, 5, H, W] uint8 CPU tensor
        t5_emb:    [2, 512, 1024] float32 T5 embedding（或零张量）
        action:    [2, T, action_dim] float32 packed action prefix
        delay_scalar: [2, 1] float32 normalized delay scalar

    Returns:
        dict: 所有 tensor 已移至 CUDA + bfloat16（浮点部分）
    """
    B, C, T, H, W = vid_input.shape
    data_batch = {
        "dataset_name": "video_data",
            "video": vid_input.cuda(),            # must stay uint8; model normalizes internally
        "fps": torch.tensor([24.0] * B),      # converted to bf16 below
        "padding_mask": torch.zeros(B, 1, H, W),  # concat_padding_mask=True → must be bf16
        "num_conditional_frames": _NUM_LATENT_COND,
        "t5_text_embeddings": t5_emb,
        "action": action,
        "delay_scalar": delay_scalar,
    }
    # All floating-point tensors except video (uint8) must be bfloat16 to match
    # model weights. padding_mask is especially critical: it gets torch.cat-ed with
    # bfloat16 x in prepare_embedded_sequence; a float32 padding_mask would upcast
    # x to float32 and break the bfloat16 linear layers (concat_padding_mask=True).
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
    return data_batch


def build_action_inputs(action_rows: np.ndarray, delay: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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
# 主评估流程
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_indices: Optional[List[int]] = (
        [int(t) for t in args.task_indices] if args.task_indices else None
    )

    num_steps_list: List[int] = args.num_steps  # e.g. [1, 2, 4]

    # ── 1. 加载蒸馏模型 ──────────────────────────────────────────────────────
    experiment_name = args.experiment or _DISTILL_EXPERIMENT_NAME
    model = load_distilled_model(str(args.ckpt), experiment_name)

    # ── 2. 可选：LPIPS 损失网络 ────────────────────────────────────────────
    lpips_fn = None
    if _HAS_LPIPS:
        lpips_fn = _lpips_lib.LPIPS(net="alex").cuda().eval()
        print("[info] LPIPS (AlexNet) loaded.")

    # ── 3. 加载任务描述 & T5 嵌入 ────────────────────────────────────────────
    data_root = args.data_root or _DEFAULT_DATA_ROOT
    task_descriptions = load_task_descriptions(data_root)

    # T5 嵌入查找表：key = task 描述字符串，value = [1, 512, 1024]
    t5_cache: Dict[str, torch.Tensor] = {}
    if args.t5_emb_path:
        emb_dict = load_t5_embeddings(args.t5_emb_path)
        if emb_dict is not None:
            for k, v in emb_dict.items():
                t5_cache[str(k)] = torch.as_tensor(v, dtype=torch.float32)
            print(f"[info] 预计算 T5 嵌入已加载：{len(t5_cache)} 个任务")

    def _get_t5_emb(prompt: str) -> torch.Tensor:
        """查询预计算嵌入；缺失时返回零向量（无文本引导时安全）。"""
        if prompt in t5_cache:
            emb = t5_cache[prompt]
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)  # [512, 1024] → [1, 512, 1024]
            return emb.repeat(2, 1, 1)
        # 缓存未命中：使用零向量（action-conditioned 蒸馏模型不依赖文本引导）
        print(f"[warn] T5 cache miss for prompt: '{prompt[:50]}…'. Using zeros.")
        return torch.zeros(2, 512, 1024, dtype=torch.float32)

    # ── 4. 构建 val 数据集采样表 ────────────────────────────────────────────
    episode_samples = build_val_episode_samples(
        data_root=data_root,
        val_ratio=0.1,
        split_seed=0,           # 固定，与训练一致
        max_delay=_MAX_DELAY,
        samples_per_episode=args.samples_per_episode,
        sample_seed=args.seed,
        task_indices=task_indices,
    )

    # ── 5. 初始化指标容器 ─────────────────────────────────────────────────
    delays = [1, 2, 3, 4]
    # results[num_steps][d]
    results_by_steps: Dict[int, Dict[int, defaultdict]] = {
        ns: {d: defaultdict(list) for d in delays}
        for ns in num_steps_list
    }
    # 按任务细分
    results_per_task_by_steps: Dict[int, Dict[int, Dict[int, defaultdict]]] = {
        ns: {} for ns in num_steps_list
    }
    all_task_indices_seen: set = set()

    if args.save_images:
        pathlib.Path(args.save_images).mkdir(parents=True, exist_ok=True)

    # ── 6. 评估主循环 ──────────────────────────────────────────────────────
    total_windows = sum(len(s) for _, _, s in episode_samples)
    print(
        f"\n开始评估，共 {total_windows} 个 (episode, start_t) "
        f"× {len(delays)} delays × {len(num_steps_list)} num_steps = "
        f"{total_windows * len(delays) * len(num_steps_list)} 次推理 …\n"
    )

    done = 0
    t_start = time.time()

    for ep_idx, (file_path, ep_task_index, start_ts) in enumerate(episode_samples):
        ep_name = file_path.stem

        prompt = task_descriptions.get(ep_task_index, f"libero task {ep_task_index}")
        t5_emb = _get_t5_emb(prompt)   # [1, 512, 1024]

        for ns in num_steps_list:
            if ep_task_index not in results_per_task_by_steps[ns]:
                results_per_task_by_steps[ns][ep_task_index] = {
                    d: defaultdict(list) for d in delays
                }
        all_task_indices_seen.add(ep_task_index)

        df = pd.read_parquet(file_path)
        T_ep = len(df)

        for start_t in start_ts:
            cam1_t = _decode_image(df[_CAM1_KEY].iloc[start_t])
            cam2_t = _decode_image(df[_CAM2_KEY].iloc[start_t])

            vid_input = build_input_video(cam1_t, cam2_t)   # [2, 3, 5, H, W]

            for d in delays:
                pred_t = start_t + d
                if pred_t >= T_ep:
                    continue

                cam1_gt = _decode_image(df[_CAM1_KEY].iloc[pred_t])
                cam2_gt = _decode_image(df[_CAM2_KEY].iloc[pred_t])

                action_rows = np.stack(
                    [np.asarray(df[_ACT_KEY].iloc[start_t + offset], dtype=np.float32) for offset in range(d)],
                    axis=0,
                )
                action, delay_scalar = build_action_inputs(action_rows, delay=d, batch_size=2)

                data_batch = build_data_batch(vid_input, t5_emb, action, delay_scalar)

                # ── 对每个 num_steps 分别推理 ──────────────────────────
                for ns in num_steps_list:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    time_start = time.perf_counter()
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        time_inference_start = time.perf_counter()
                        latents = model.generate_samples_from_batch(
                            data_batch,
                            n_sample=1,
                            seed=args.seed,
                            num_steps=ns,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        time_inference_end = time.perf_counter()
                        print(f"Time taken for inference: {time_inference_end - time_inference_start} seconds")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        time_decode_start = time.perf_counter()
                        video_out = model.decode(latents)  # [-1, 1], [2, 3, 5, H, W]
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        time_decode_end = time.perf_counter()
                        print(f"Time taken for decode: {time_decode_end - time_decode_start} seconds")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_elapsed = time.perf_counter() - time_start
                    elapsed_ms = total_elapsed * 1000
                    print(f"Time taken for total: {total_elapsed} seconds")
                    # 提取预测帧（paired 5-frame）
                    cam1_pred = _tensor_to_uint8(video_out[0, :, 1])   # batch0 frame1
                    cam2_pred = _tensor_to_uint8(video_out[1, :, 1])   # batch1 frame1

                    # 指标
                    psnr1 = compute_psnr(cam1_pred, cam1_gt)
                    psnr2 = compute_psnr(cam2_pred, cam2_gt)
                    results_by_steps[ns][d]["cam1_psnr"].append(psnr1)
                    results_by_steps[ns][d]["cam2_psnr"].append(psnr2)
                    results_by_steps[ns][d]["inference_ms"].append(elapsed_ms)
                    results_per_task_by_steps[ns][ep_task_index][d]["cam1_psnr"].append(psnr1)
                    results_per_task_by_steps[ns][ep_task_index][d]["cam2_psnr"].append(psnr2)

                    ssim1 = compute_ssim(cam1_pred, cam1_gt)
                    ssim2 = compute_ssim(cam2_pred, cam2_gt)
                    results_by_steps[ns][d]["cam1_ssim"].append(ssim1)
                    results_by_steps[ns][d]["cam2_ssim"].append(ssim2)
                    results_per_task_by_steps[ns][ep_task_index][d]["cam1_ssim"].append(ssim1)
                    results_per_task_by_steps[ns][ep_task_index][d]["cam2_ssim"].append(ssim2)

                    if lpips_fn is not None:
                        with torch.no_grad():
                            t1_pred = video_out[0:1, :, 1].float()
                            t2_pred = video_out[1:2, :, 1].float()
                            t1_gt = (
                                torch.from_numpy(cam1_gt).float() / 127.5 - 1
                            ).permute(2, 0, 1).unsqueeze(0).cuda()
                            t2_gt = (
                                torch.from_numpy(cam2_gt).float() / 127.5 - 1
                            ).permute(2, 0, 1).unsqueeze(0).cuda()
                            lp1 = float(lpips_fn(t1_pred, t1_gt).item())
                            lp2 = float(lpips_fn(t2_pred, t2_gt).item())
                        results_by_steps[ns][d]["cam1_lpips"].append(lp1)
                        results_by_steps[ns][d]["cam2_lpips"].append(lp2)
                        results_per_task_by_steps[ns][ep_task_index][d]["cam1_lpips"].append(lp1)
                        results_per_task_by_steps[ns][ep_task_index][d]["cam2_lpips"].append(lp2)

                    # 可选：仅对第一个 num_steps 保存图片（避免重复）
                    if args.save_images and ns == num_steps_list[0]:
                        tag = f"{ep_name}_t{start_t:04d}_d{d}"
                        Image.fromarray(np.hstack([cam1_pred, cam1_gt])).save(
                            f"{args.save_images}/{tag}_cam1_pred_vs_gt.png"
                        )
                        Image.fromarray(np.hstack([cam2_pred, cam2_gt])).save(
                            f"{args.save_images}/{tag}_cam2_pred_vs_gt.png"
                        )

            done += 1
            if done % 10 == 0 or done == total_windows:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_windows - done) if done < total_windows else 0
                print(f"  [{done}/{total_windows}] elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    # ── 辅助函数 ─────────────────────────────────────────────────────────────
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
        if r["inference_ms"]:
            s["avg_inference_ms"] = round(float(np.nanmean(r["inference_ms"])), 1)
        return s

    def _print_table(title: str, results_dict: Dict[int, defaultdict]):
        print(f"\n{'='*65}")
        print(title)
        print(f"{'='*65}")
        header = f"{'d':>4} | {'cam1_PSNR':>10} {'cam2_PSNR':>10} {'avg_PSNR':>10}"
        if _HAS_SKIMAGE:
            header += f" | {'cam1_SSIM':>10} {'avg_SSIM':>10}"
        if _HAS_LPIPS:
            header += f" | {'cam1_LPIPS':>10}"
        header += f" | {'ms/step':>8} | {'N':>5}"
        print(header)
        print("-" * len(header))
        for d in delays:
            r = results_dict[d]
            s = _summarize(r)
            if not s:
                continue
            row = (
                f"{d:>4} | {s['cam1_psnr']:>10.3f} {s['cam2_psnr']:>10.3f} "
                f"{s['avg_psnr']:>10.3f}"
            )
            if _HAS_SKIMAGE and "cam1_ssim" in s:
                row += f" | {s['cam1_ssim']:>10.4f} {s['avg_ssim']:>10.4f}"
            if _HAS_LPIPS and "cam1_lpips" in s:
                row += f" | {s['cam1_lpips']:>10.4f}"
            ms = s.get("avg_inference_ms", float("nan"))
            row += f" | {ms:>8.1f} | {s['n']:>5}"
            print(row)

    # ── 7. 汇总并打印结果 ─────────────────────────────────────────────────
    summary: Dict[str, dict] = {}

    for ns in num_steps_list:
        _print_table(
            f"蒸馏模型 num_steps={ns}  —  评估结果（所有 task，按 delay 分组）",
            results_by_steps[ns],
        )
        step_key = f"steps={ns}"
        summary[step_key] = {}
        for d in delays:
            s = _summarize(results_by_steps[ns][d])
            if s:
                summary[step_key][f"d={d}"] = s

    # 多任务细分（若评估了多个任务）
    per_task_summary: Dict[str, Dict[str, dict]] = {}
    if len(all_task_indices_seen) > 1:
        for ns in num_steps_list:
            for tidx in sorted(all_task_indices_seen):
                desc = task_descriptions.get(tidx, f"task {tidx}")
                _print_table(
                    f"steps={ns}  Task {tidx} [{desc[:50]}{'…' if len(desc) > 50 else ''}]",
                    results_per_task_by_steps[ns].get(tidx, {d: defaultdict(list) for d in delays}),
                )
                task_key = f"task{tidx}"
                if task_key not in per_task_summary:
                    per_task_summary[task_key] = {}
                per_task_summary[task_key][f"steps={ns}"] = {}
                for d in delays:
                    s = _summarize(
                        results_per_task_by_steps[ns].get(tidx, {}).get(d, defaultdict(list))
                    )
                    if s:
                        per_task_summary[task_key][f"steps={ns}"][f"d={d}"] = s

    print(f"\n{'='*65}")
    print(f"总耗时: {(time.time()-t_start)/60:.1f} min")

    # ── 8. 保存 JSON 结果 ─────────────────────────────────────────────────
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
            "num_steps_evaluated": num_steps_list,
            "delays_evaluated": delays,
            "t5_emb_path": str(args.t5_emb_path) if args.t5_emb_path else None,
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
        description=(
            "Offline eval of DMD2-distilled Action-Conditioned World Model "
            "(LIBERO, paired 5-frame layout, skip-dynamics)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt", type=pathlib.Path, required=True,
        help="蒸馏后的 model_ema_bf16.pt（由 convert_distcp_to_pt.py 转换得到）",
    )
    p.add_argument(
        "--data-root", type=str, default=None,
        help=f"覆盖数据根目录。默认: {_DEFAULT_DATA_ROOT}",
    )
    p.add_argument(
        "--experiment", type=str, default=None,
        help=(
            f"DMD2 实验名。默认: {_DISTILL_EXPERIMENT_NAME}"
        ),
    )
    p.add_argument(
        "--task-indices", type=int, nargs="+", default=None, metavar="TASK_IDX",
        help="只评估指定任务（如 --task-indices 0）。不指定则评估全部。",
    )
    p.add_argument(
        "--t5-emb-path", type=str, default=None,
        help=(
            "预计算 T5 embedding pkl 文件路径（scripts/precompute_libero_t5.py 生成）。"
            "不提供时使用零向量（对 action-conditioned 蒸馏模型基本无影响）。"
        ),
    )
    p.add_argument(
        "--num-steps", type=int, nargs="+", default=[1, 2, 4],
        metavar="N",
        help=(
            "推理步数列表，可指定多个（如 --num-steps 1 2 4）或单个（--num-steps 1）。"
            "每组步数独立统计指标。默认: 1 2 4"
        ),
    )
    p.add_argument(
        "--samples-per-episode", type=int, default=20,
        help="每个 val episode 随机抽取的 start_t 数量",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（split seed 始终固定为 0）",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="JSON 结果保存路径（如 outputs/eval_distill/task0.json）",
    )
    p.add_argument(
        "--save-images", type=str, default=None,
        help=(
            "可选：保存 pred vs GT 对比图片的目录。"
            "多个 num_steps 时仅对第一个步数保存图片。"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    args = parse_args()
    evaluate(args)
