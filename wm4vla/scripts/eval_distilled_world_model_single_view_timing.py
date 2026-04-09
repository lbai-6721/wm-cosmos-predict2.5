#!/usr/bin/env python3
"""Measure single-view generation latency for distilled WM4VLA model.

This script benchmarks distilled model inference time with single-view input
(`cam1` or `cam2`) while reusing val split sampling and prompt/action
construction logic from `eval_distilled_world_model.py`.


CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model_single_view_timing.py \
  --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5-new/output/wm-distill-output/distill_v3_light/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints/iter_000011000/model_ema_bf16.pt \
  --task-indices 0 \
  --camera cam1 \
  --num-steps 1 \
  --max-windows 20 \
  --warmup-calls 2 \
  --t5-emb-path /home/kyji/public/dataset/lerobot/lerobot--libero_10_image@v2.0/meta/t5_embeddings.pkl \
  --save-images outputs/eval_distill/single_view_cam1_timing_images \
  --output outputs/eval_distill/single_view_cam1_timing.json
"""

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wm4vla.scripts import eval_distilled_world_model as base_eval


def build_single_view_input(view_t: np.ndarray) -> torch.Tensor:
    """Build single-view 5-frame input with shape [1, 3, 5, H, W]."""
    height, width = view_t.shape[:2]
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    frames = np.stack([view_t, blank, blank, blank, blank], axis=0)
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    img = ((t.float() + 1.0) / 2.0).clamp(0, 1)
    img = (img * 255.0).to(torch.uint8)
    return img.permute(1, 2, 0).cpu().numpy()


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarize_times(times: List[float]) -> dict:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_sec": round(float(arr.mean()), 6),
        "std_sec": round(float(arr.std()), 6),
        "min_sec": round(float(arr.min()), 6),
        "p50_sec": round(float(np.percentile(arr, 50)), 6),
        "p95_sec": round(float(np.percentile(arr, 95)), 6),
        "max_sec": round(float(arr.max()), 6),
    }


def evaluate_single_view_timing(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_indices: Optional[List[int]] = (
        [int(task) for task in args.task_indices] if args.task_indices else None
    )

    experiment_name = args.experiment or base_eval._DISTILL_EXPERIMENT_NAME
    model = base_eval.load_distilled_model(str(args.ckpt), experiment_name)

    data_root = args.data_root or base_eval._DEFAULT_DATA_ROOT
    task_descriptions = base_eval.load_task_descriptions(data_root)

    t5_cache: Dict[str, torch.Tensor] = {}
    if args.t5_emb_path:
        emb_dict = base_eval.load_t5_embeddings(args.t5_emb_path)
        if emb_dict is not None:
            for key, value in emb_dict.items():
                t5_cache[str(key)] = torch.as_tensor(value, dtype=torch.float32)
            print(f"[info] 预计算 T5 嵌入已加载：{len(t5_cache)} 个任务")

    def _get_t5_emb(prompt: str) -> torch.Tensor:
        if prompt in t5_cache:
            emb = t5_cache[prompt]
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            return emb
        return torch.zeros(1, 512, 1024, dtype=torch.float32)

    episode_samples = base_eval.build_val_episode_samples(
        data_root=data_root,
        val_ratio=0.1,
        split_seed=0,
        max_delay=base_eval._MAX_DELAY,
        samples_per_episode=args.samples_per_episode,
        sample_seed=args.seed,
        task_indices=task_indices,
    )

    if args.max_windows is not None:
        capped = []
        remaining = args.max_windows
        for file_path, ep_task_index, start_ts in episode_samples:
            if remaining <= 0:
                break
            kept = start_ts[:remaining]
            if kept:
                capped.append((file_path, ep_task_index, kept))
                remaining -= len(kept)
        episode_samples = capped

    delays = args.delays
    all_times: List[float] = []
    results_by_delay = {delay: [] for delay in delays}
    measured_calls = 0
    warmup_calls = 0

    if args.save_images:
        pathlib.Path(args.save_images).mkdir(parents=True, exist_ok=True)

    camera_key = base_eval._CAM1_KEY if args.camera == "cam1" else base_eval._CAM2_KEY
    print(
        f"开始蒸馏模型单视角计时，共 {sum(len(s) for _, _, s in episode_samples)} 个窗口，"
        f"{len(delays)} 个 delays，warmup={args.warmup_calls}\n"
    )

    for file_path, ep_task_index, start_ts in episode_samples:
        ep_name = file_path.stem
        prompt = task_descriptions.get(ep_task_index, f"libero task {ep_task_index}")
        t5_emb = _get_t5_emb(prompt)
        df = base_eval.pd.read_parquet(file_path)
        total_frames = len(df)

        for start_t in start_ts:
            view_t = base_eval._decode_image(df[camera_key].iloc[start_t])
            vid_input = build_single_view_input(view_t)

            for delay in delays:
                pred_t = start_t + delay
                if pred_t >= total_frames:
                    continue

                action_rows = np.stack(
                    [np.asarray(df[base_eval._ACT_KEY].iloc[start_t + offset], dtype=np.float32) for offset in range(delay)],
                    axis=0,
                )
                action, delay_scalar = base_eval.build_action_inputs(action_rows, delay=delay, batch_size=1)

                data_batch = base_eval.build_data_batch(vid_input, t5_emb, action, delay_scalar)

                _sync_cuda()
                start_time = time.perf_counter()
                with torch.no_grad():
                    latents = model.generate_samples_from_batch(
                        data_batch,
                        n_sample=1,
                        seed=args.seed,
                        num_steps=args.num_steps,
                    )
                    video_out = model.decode(latents)
                _sync_cuda()
                elapsed = time.perf_counter() - start_time

                if warmup_calls < args.warmup_calls:
                    warmup_calls += 1
                    print(
                        f"[warmup {warmup_calls}/{args.warmup_calls}] "
                        f"{ep_name} t={start_t} d={delay} time={elapsed:.4f}s"
                    )
                    continue

                measured_calls += 1
                all_times.append(elapsed)
                results_by_delay[delay].append(elapsed)
                print(
                    f"[{measured_calls}] {args.camera} {ep_name} "
                    f"t={start_t} d={delay} time={elapsed:.4f}s"
                )

                if args.save_images:
                    pred = _tensor_to_uint8(video_out[0, :, 1])
                    gt = base_eval._decode_image(df[camera_key].iloc[pred_t])
                    tag = f"{ep_name}_t{start_t:04d}_d{delay}_{args.camera}"
                    Image.fromarray(np.hstack([pred, gt])).save(
                        f"{args.save_images}/{tag}_pred_vs_gt.png"
                    )

    if not all_times:
        raise RuntimeError("没有测到任何有效样本，请检查 --max-windows / --delays / 数据集长度。")

    summary = {
        "meta": {
            "ckpt": str(args.ckpt),
            "data_root": str(data_root),
            "experiment": experiment_name,
            "camera": args.camera,
            "task_indices": task_indices,
            "samples_per_episode": args.samples_per_episode,
            "max_windows": args.max_windows,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "warmup_calls": args.warmup_calls,
            "delays": delays,
            "t5_emb_path": str(args.t5_emb_path) if args.t5_emb_path else None,
        },
        "overall": _summarize_times(all_times),
        "per_delay": {
            f"d={delay}": _summarize_times(delay_times)
            for delay, delay_times in results_by_delay.items()
            if delay_times
        },
    }

    print(f"\n{'=' * 60}")
    print("蒸馏模型单视角生成耗时汇总")
    print(f"{'=' * 60}")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    for delay, stats in summary["per_delay"].items():
        print(f"{delay}: {json.dumps(stats, ensure_ascii=False)}")

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {out_path}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure single-view generation latency of distilled WM4VLA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=pathlib.Path, required=True, help="Path to distilled model_ema_*.pt")
    parser.add_argument(
        "--camera",
        type=str,
        choices=("cam1", "cam2"),
        default="cam1",
        help="Which single view to generate",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Override data root. Default: {base_eval._DEFAULT_DATA_ROOT}",
    )
    parser.add_argument("--experiment", type=str, default=None, help="Distill experiment name override")
    parser.add_argument("--task-indices", type=int, nargs="+", default=None, metavar="TASK_IDX")
    parser.add_argument("--t5-emb-path", type=str, default=None, help="Path to pre-computed T5 embeddings")
    parser.add_argument("--num-steps", type=int, default=1, help="Distilled sampling steps")
    parser.add_argument(
        "--samples-per-episode",
        type=int,
        default=20,
        help="Number of start_t to sample per val episode",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on total sampled windows before expanding delays",
    )
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Which delays to benchmark",
    )
    parser.add_argument("--warmup-calls", type=int, default=1, help="Warmup calls excluded from timing stats")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--output", type=str, default=None, help="Path to save timing summary JSON")
    parser.add_argument(
        "--save-images",
        type=str,
        default=None,
        help="Optional directory to save pred-vs-gt image pairs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_single_view_timing(parse_args())
