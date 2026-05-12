#!/usr/bin/env python3
"""visualize_wm.py

直接输出 World Model 生成的原始 5 帧单视角视频（mp4）。

视频帧结构（state_t=2，paired batch）：
  Batch 0:
    Frame 0    : cam1_t            ← 条件帧（第一视角）
    Frames 1-4 : cam1 预测 ×4      ← WM 预测的第一视角（取 frame 1）
  Batch 1:
    Frame 0    : cam2_t            ← 条件帧（腕部视角）
    Frames 1-4 : cam2 预测 ×4      ← WM 预测的腕部视角（取 frame 1）

输出：每次推理生成两个 mp4（cam1 / cam2），分别对应强配对 batch 的两个视角。
可选追加 GT 对比（--with-gt 模式下，在每帧右侧加对应 GT）。

用法：
    python scripts/visualize_wm.py \\
        --ckpt /path/to/model_ema_bf16.pt \\
        --output outputs/wm_vis \\
        [--n-episodes 3] \\
        [--delay 1] \\
        [--num-steps 10] \\
        [--with-gt]

CUDA_VISIBLE_DEVICES=7 python scripts/visualize_wm.py \
    --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/reconstruct_new/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_libero_10_lerobot_256_skip_dynamics_dual_cam_task0/checkpoints/iter_000008000/model_ema_bf16.pt \
    --output outputs/wm_vis_new \
    --task-indices 0 \
    --n-episodes 2 \
    --delay 1 \
    --num-steps 35
"""

import argparse
import io
import pathlib
import random
import sys

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

# ── 常量 ──────────────────────────────────────────────────────────────────
_CAM1_KEY = "observation.images.image"
_CAM2_KEY = "observation.images.wrist_image"
_ACT_KEY  = "action"
_DEFAULT_DATA_ROOT = (
    "/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot"
    "/lerobot--libero_10_image@v2.0"
)
_EXPERIMENT_NAME_FULL = "ac_libero_lerobot_256_pixels_2b"
_EXPERIMENT_NAME_TASK01 = "ac_libero_lerobot_256_pixels_2b_task01"
_EXPERIMENT_NAME_TASK0 = "ac_libero_lerobot_256_pixels_2b_task0"
_CONFIG_FILE      = "wm4vla/configs/action_conditioned/config.py"
_NUM_LATENT_COND  = 1    # state_t=2 → 1 conditioning latent frame
_NUM_VIDEO_FRAMES = 5    # 1 + (2-1)×4 = 5 pixel frames
_RESOLUTION       = "256,256"
_MAX_DELAY        = INFER_DELAY_MAX


# ── 工具函数 ──────────────────────────────────────────────────────────────

def decode_image(cell) -> np.ndarray:
    """LeRobot v2.0 parquet 图像列 → (H, W, 3) uint8"""
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, (bytes, bytearray)):
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    raise TypeError(f"Unexpected image cell: {type(cell)}")


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [-1, 1] → (H, W, 3) uint8"""
    img = ((t.float() + 1.0) / 2.0).clamp(0, 1)
    return (img * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def video_tensor_to_frames(video: torch.Tensor, batch_idx: int) -> list[np.ndarray]:
    """[B, 3, T, H, W] float [-1, 1] → 指定 batch 的 T 帧 uint8 列表"""
    T = video.shape[2]
    return [tensor_to_uint8(video[batch_idx, :, t]) for t in range(T)]


def save_mp4(frames: list[np.ndarray], path: str, fps: int = 4):
    """保存 mp4，优先用 imageio，退而求其次用 cv2。"""
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        return
    except Exception:
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        return
    except Exception:
        pass
    raise RuntimeError(
        "无法保存 mp4，请安装：pip install imageio[ffmpeg]  或  pip install opencv-python"
    )


def build_input_video(cam1_t: np.ndarray, cam2_t: np.ndarray) -> torch.Tensor:
    """构建成对 5 帧 WM 输入，布局与训练/评估脚本一致。[2, 3, 5, H, W] uint8"""
    H, W = cam1_t.shape[:2]
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    cam1_frames = np.stack([cam1_t, blank, blank, blank, blank], axis=0)
    cam2_frames = np.stack([cam2_t, blank, blank, blank, blank], axis=0)
    frames = np.stack([cam1_frames, cam2_frames], axis=0)
    return torch.from_numpy(frames).permute(0, 4, 1, 2, 3)


def build_action_inputs(action_rows: np.ndarray, delay: int) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_masked_action_sequence(
        actions=action_rows,
        delay=delay,
        chunk_len=_MAX_DELAY,
    )
    delay_scalar = normalize_delay_scalar(delay=delay, max_delay=_MAX_DELAY)
    action = torch.from_numpy(packed).float().unsqueeze(0).repeat(2, 1, 1)
    delay_tensor = torch.from_numpy(delay_scalar).float().unsqueeze(0).repeat(2, 1)
    return action, delay_tensor


# ── Val split ──────────────────────────────────────────────────────────────

def sample_val_episodes(
    data_root: str,
    n: int,
    seed: int,
    task_indices: list[int] | None = None,
) -> list[pathlib.Path]:
    data_dir = pathlib.Path(data_root) / "data" / "chunk-000"
    all_files = sorted(data_dir.glob("episode_*.parquet"))
    rng_split = np.random.default_rng(0)        # split_seed=0，与训练一致
    perm = rng_split.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * 0.1))
    val_set = set(perm[:n_val].tolist())
    val_files = [all_files[i] for i in range(len(all_files)) if i in val_set]
    if task_indices is not None:
        wanted = set(task_indices)
        filtered = []
        for fp in val_files:
            try:
                task_idx = int(pd.read_parquet(fp, columns=["task_index"]).iloc[0]["task_index"])
            except Exception:
                continue
            if task_idx in wanted:
                filtered.append(fp)
        val_files = filtered
    rng2 = random.Random(seed)
    chosen = rng2.sample(val_files, min(n, len(val_files)))
    task_msg = task_indices if task_indices is not None else "all"
    print(f"[data] val episodes(task={task_msg}): {len(val_files)}, selected: {len(chosen)}")
    return chosen


# ── 主流程 ────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_indices = [int(t) for t in args.task_indices] if args.task_indices else None
    if args.experiment:
        experiment_name = args.experiment
    elif task_indices is None:
        experiment_name = _EXPERIMENT_NAME_FULL
    elif sorted(task_indices) == [0, 1]:
        experiment_name = _EXPERIMENT_NAME_TASK01
    elif sorted(task_indices) == [0]:
        experiment_name = _EXPERIMENT_NAME_TASK0
    else:
        experiment_name = _EXPERIMENT_NAME_FULL

    print("加载 World Model …")
    print(f"  ckpt        : {args.ckpt}")
    print(f"  experiment  : {experiment_name}")
    print(f"  task_indices: {task_indices if task_indices is not None else 'all'}")
    from wm4vla.inference.video2world import Video2WorldInference, resolve_tokenizer_overrides

    tokenizer_backend, _, experiment_opts = resolve_tokenizer_overrides(
        tokenizer_backend=args.tokenizer_backend,
        tokenizer_vae_pth=args.tokenizer_vae_pth,
        lightx2v_root=args.lightx2v_root,
        use_lightvae=args.use_lightvae,
        lightvae_pth=args.lightvae_pth,
        experiment_opts=args.experiment_opts,
    )
    if tokenizer_backend == "lightvae":
        print(f"[info] Tokenizer backend=lightvae, overrides: {experiment_opts}")
    elif experiment_opts:
        print(f"[info] Tokenizer backend=wan2pt1, overrides: {experiment_opts}")
    if tokenizer_backend != "lightvae" and args.lightx2v_root:
        print("[warn] --lightx2v-root is ignored when tokenizer backend is wan2pt1")

    wm = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=str(args.ckpt),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=_CONFIG_FILE,
        experiment_opts=experiment_opts,
    )
    wm.model.eval()
    print("模型加载完成。")

    data_root = args.data_root or _DEFAULT_DATA_ROOT
    val_eps = sample_val_episodes(data_root, args.n_episodes, args.seed, task_indices=task_indices)
    delays = [args.delay] if args.delay is not None else list(DEFAULT_EVAL_DELAYS)

    for ep_path in val_eps:
        ep_name = ep_path.stem
        df = pd.read_parquet(ep_path)
        T = len(df)
        task_index = int(df["task_index"].iloc[0])
        prompt = f"libero spatial task {task_index}"

        valid_starts = list(range(T - _MAX_DELAY))
        if not valid_starts:
            print(f"[skip] {ep_name}: too short (T={T})")
            continue
        start_t = random.Random(args.seed + hash(ep_name)).choice(valid_starts)

        cam1_t = decode_image(df[_CAM1_KEY].iloc[start_t])
        cam2_t = decode_image(df[_CAM2_KEY].iloc[start_t])
        vid_input = build_input_video(cam1_t, cam2_t)

        print(f"\n[{ep_name}] task={task_index}  start_t={start_t}  T={T}")

        for d in delays:
            pred_t = start_t + d
            if pred_t >= T:
                continue

            if d == 0 and BYPASS_WM_WHEN_DELAY_ZERO:
                video_out = None
                print(f"  d={d}: bypass WM，直接使用真实观测")
            else:
                action_rows = np.stack(
                    [np.asarray(df[_ACT_KEY].iloc[start_t + offset], dtype=np.float32) for offset in range(d)],
                    axis=0,
                )
                action, delay_scalar = build_action_inputs(action_rows, delay=d)

                print(f"  d={d}: 推理 ({args.num_steps} steps) …", end="", flush=True)
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
                print(" 完成")

            view_specs = [
                ("cam1", 0, decode_image(df[_CAM1_KEY].iloc[start_t]), decode_image(df[_CAM1_KEY].iloc[pred_t])),
                ("cam2", 1, decode_image(df[_CAM2_KEY].iloc[start_t]), decode_image(df[_CAM2_KEY].iloc[pred_t])),
            ]

            for view_name, batch_idx, cond_gt, pred_gt in view_specs:
                if video_out is None:
                    raw_frames = [cond_gt.copy() for _ in range(_NUM_VIDEO_FRAMES)]
                    raw_frames[1:] = [pred_gt.copy() for _ in range(_NUM_VIDEO_FRAMES - 1)]
                else:
                    raw_frames = video_tensor_to_frames(video_out, batch_idx)
                out_frames = raw_frames

                if args.with_gt:
                    per_frame_gt = {0: cond_gt}
                    per_frame_gt.update({i: pred_gt for i in range(1, 5)})
                    out_frames = [
                        np.hstack([raw_frames[i], per_frame_gt[i]])
                        for i in range(len(raw_frames))
                    ]

                vid_path = str(out_dir / f"{ep_name}_t{start_t:04d}_d{d}_{view_name}.mp4")
                slow = []
                for f in out_frames:
                    slow.extend([f] * 4)
                try:
                    save_mp4(slow, vid_path, fps=8)
                    print(f"    → {vid_path}")
                except RuntimeError as e:
                    print(f"    [warn] {e}")
                    for fi, frame in enumerate(out_frames):
                        Image.fromarray(frame).save(
                            out_dir / f"{ep_name}_t{start_t:04d}_d{d}_{view_name}_frame{fi:02d}.png"
                        )
                    print(f"    → 已改存 PNG 帧 至 {out_dir}")

    print(f"\n完成！输出目录: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="保存 WM 原始 5 帧成对视角输出视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=pathlib.Path, required=True,
                   help="Path to model_ema_bf16.pt")
    p.add_argument("--output", type=str, default="outputs/wm_vis",
                   help="输出目录")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--experiment", type=str, default=None,
        help=(
            "Experiment name. 默认会根据 --task-indices 自动选择："
            "all->ac_libero_lerobot_256_pixels_2b, "
            "[0,1]->ac_libero_lerobot_256_pixels_2b_task01, "
            "[0]->ac_libero_lerobot_256_pixels_2b_task0"
        ),
    )
    p.add_argument(
        "--task-indices", type=int, nargs="+", default=[0], metavar="TASK_IDX",
        help="可视化这些 task（默认 [0]，即与你的 task0 训练一致）",
    )
    p.add_argument("--n-episodes", type=int, default=3,
                   help="可视化的 val episode 数量")
    p.add_argument("--delay", type=int, default=None,
                   help="指定单个 d（不指定则跑 d=0..max_delay；d=0 直接 bypass WM）")
    p.add_argument("--num-steps", type=int, default=35,
                   help="扩散去噪步数（10 可快速预览）")
    p.add_argument("--guidance", type=float, default=0.0,
                   help="CFG guidance 强度（0=纯条件推理，~2× 加速；>0 启用 CFG）")
    p.add_argument(
        "--tokenizer-backend", type=str, choices=["wan2pt1", "lightvae"], default="wan2pt1",
        help="Tokenizer backend for visualization inference.",
    )
    p.add_argument(
        "--tokenizer-vae-pth", type=str, default="",
        help="Optional VAE checkpoint path override for either tokenizer backend.",
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
    p.add_argument(
        "--experiment-opts", nargs="*", default=[],
        help="Additional Hydra overrides passed to the WM config (optional).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-gt", action="store_true",
                   help="在视频右侧拼接对应 GT 帧（frame0 条件帧，frame1-4 预测帧）")
    return p.parse_args()


if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    args = parse_args()
    main(args)
