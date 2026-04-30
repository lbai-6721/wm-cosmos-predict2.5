#!/usr/bin/env python3
"""Offline evaluation for MetaWorld DMD2-distilled wm4vla world models.

This script mirrors wm4vla/scripts/eval_distilled_world_model.py, but keeps the
MetaWorld-specific single-view path separate:

  * one camera: observation.image
  * action layout: [B, 8, 5] = 4D action + valid mask
  * LightVAE tokenizer by default, matching train_distill_metaworld.sh

Example:

  CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_metaworld.py \
      --ckpt /path/to/model_ema_bf16.pt \
      --benchmark mt50 \
      --num-steps 1 2 4 \
      --t5-emb-path ${METAWORLD_T5_EMB_PATH} \
      --lightvae-pth ${LIGHTVAE_PTH} \
      --lightx2v-root ${LIGHTX2V_ROOT} \
      --output outputs/eval_distill/metaworld_mt50_steps124.json
"""

from __future__ import annotations

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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wm4vla.conditioning import normalize_delay_scalar, pack_masked_action_sequence
from wm4vla.configs.wm_conditioning import ACTION_CHUNK_LEN

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


_IMAGE_KEY = "observation.image"
_ACT_KEY = "action"
_EPISODE_KEY = "episode_index"
_FRAME_KEY = "frame_index"
_TASK_KEY = "task_index"

_DEFAULT_DATA_ROOT = os.environ.get(
    "METAWORLD_DATA_ROOT",
    "/mnt/cpfs/yangboxue/vla/wm4vla/data/dataset/lerobot/metaworld_mt50",
)
_DEFAULT_LIGHTVAE_PTH = os.environ.get(
    "LIGHTVAE_PTH",
    "/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V/save_results/"
    "wan21_lightvae_distill_metaworld/exports/lightvae_step_0013000.safetensors",
)
_DEFAULT_LIGHTX2V_ROOT = os.environ.get(
    "LIGHTX2V_ROOT",
    "/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V",
)

_DISTILL_CONFIG_FILE = "cosmos_predict2/_src/interactive/configs/registry_predict2p5.py"
_EXPERIMENT_MT50 = "dmd2_trigflow_distill_wm_metaworld_mt50_256"
_EXPERIMENT_TASK0 = "dmd2_trigflow_distill_wm_metaworld_mt50_256_task0"

_NUM_LATENT_COND = 1
_NUM_VIDEO_FRAMES = 5
_MAX_DELAY = ACTION_CHUNK_LEN

_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


@dataclass(frozen=True)
class EpisodeSample:
    file_path: pathlib.Path
    episode_index: int
    task_index: int
    start_ts: List[int]


def _normalize_benchmark(benchmark: str) -> str:
    if benchmark in ("all", "mt50"):
        return "mt50"
    if benchmark == "task0":
        return "task0"
    raise ValueError(f"Unsupported benchmark: {benchmark!r}")


def _infer_experiment_name(experiment: Optional[str], benchmark: str) -> str:
    if experiment:
        return experiment
    if benchmark == "task0":
        return _EXPERIMENT_TASK0
    return _EXPERIMENT_MT50


def _resolve_task_indices(benchmark: str, task_indices: Optional[Sequence[int]]) -> Optional[List[int]]:
    if task_indices is not None:
        return [int(t) for t in task_indices]
    if benchmark == "task0":
        return [0]
    return None


def _resolve_t5_emb_path(requested_path: Optional[str], data_root: str) -> Optional[str]:
    if requested_path:
        return requested_path
    env_path = os.environ.get("METAWORLD_T5_EMB_PATH")
    if env_path:
        return env_path
    candidate = pathlib.Path(data_root) / "meta" / "t5_embeddings.pkl"
    if candidate.exists():
        return str(candidate)
    return None


def _decode_image(cell, image_size: int) -> np.ndarray:
    if not isinstance(cell, dict):
        raise TypeError(f"Expected dict image cell, got {type(cell)}")

    raw = cell.get("bytes")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"Unexpected image cell format: {list(cell.keys())}")

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), _BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    image = ((t.float() + 1.0) / 2.0).clamp(0, 1)
    image = (image * 255.0).to(torch.uint8)
    return image.permute(1, 2, 0).cpu().numpy()


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return float("nan")
    return float(_ssim_fn(pred, gt, channel_axis=-1, data_range=255))


def read_dataset_fps(data_root: str) -> float:
    info_file = pathlib.Path(data_root) / "meta" / "info.json"
    if not info_file.exists():
        return 80.0
    with open(info_file) as f:
        info = json.load(f)
    return float(info.get("fps", 80.0))


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


def load_distilled_model(
    ckpt_path: str,
    experiment_name: str,
    lightvae_pth: str,
    lightx2v_root: Optional[str],
    use_batched_vae: bool,
    extra_experiment_opts: Sequence[str],
):
    from cosmos_predict2._src.interactive.utils.model_loader import load_model_from_checkpoint

    experiment_opts = [
        "tokenizer=wan2pt1_lightvae_tokenizer",
        f"+model.config.tokenizer.vae_pth={lightvae_pth}",
        f"model.config.tokenizer.use_batched_vae={str(use_batched_vae).lower()}",
    ]
    if lightx2v_root:
        experiment_opts.append(f"+model.config.tokenizer.lightx2v_root={lightx2v_root}")
    experiment_opts.extend(extra_experiment_opts)

    print(f"\n{'=' * 60}")
    print("Loading distilled MetaWorld model")
    print(f"  ckpt       : {ckpt_path}")
    print(f"  experiment : {experiment_name}")
    print(f"  tokenizer  : wan2pt1_lightvae_tokenizer")
    print(f"  lightvae   : {lightvae_pth}")
    if lightx2v_root:
        print(f"  lightx2v   : {lightx2v_root}")
    if extra_experiment_opts:
        print(f"  extra opts : {list(extra_experiment_opts)}")
    print(f"{'=' * 60}\n")

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment_name,
        s3_checkpoint_dir=str(ckpt_path),
        config_file=_DISTILL_CONFIG_FILE,
        load_ema_to_reg=True,
        skip_teacher_init=True,
        experiment_opts=experiment_opts,
    )
    model.eval()

    if hasattr(model, "net_fake_score") and model.net_fake_score is not None:
        del model.net_fake_score
        model.net_fake_score = None
        print("[info] Released net_fake_score for inference.")
    if hasattr(model, "net_teacher") and model.net_teacher is not None:
        del model.net_teacher
        model.net_teacher = None
        print("[info] Released net_teacher for inference.")

    torch.cuda.empty_cache()
    return model


def build_val_episode_samples(
    data_root: str,
    val_ratio: float,
    split_seed: int,
    samples_per_episode: int,
    sample_seed: int,
    task_indices: Optional[Sequence[int]],
) -> List[EpisodeSample]:
    data_dir = pathlib.Path(data_root) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"MetaWorld data directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No MetaWorld parquet files found under {data_dir}/chunk-*")

    episode_infos: List[tuple[pathlib.Path, int, int, int]] = []
    for file_path in all_files:
        df_meta = pd.read_parquet(file_path, columns=[_EPISODE_KEY, _FRAME_KEY, _TASK_KEY])
        for ep_index, ep_df in df_meta.groupby(_EPISODE_KEY, sort=False):
            task_index = int(ep_df[_TASK_KEY].iloc[0])
            episode_infos.append((file_path, int(ep_index), task_index, len(ep_df)))

    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(len(episode_infos))
    n_val = max(1, int(len(episode_infos) * val_ratio))
    val_set = set(perm[:n_val].tolist())
    task_filter = set(int(t) for t in task_indices) if task_indices is not None else None

    rng_sample = np.random.default_rng(sample_seed)
    samples: List[EpisodeSample] = []
    for i, (file_path, ep_index, task_index, total_frames) in enumerate(episode_infos):
        if i not in val_set:
            continue
        if task_filter is not None and task_index not in task_filter:
            continue

        max_start = total_frames - _MAX_DELAY - 1
        if max_start < 0:
            continue

        n = min(samples_per_episode, max_start + 1)
        start_ts = sorted(rng_sample.choice(max_start + 1, size=n, replace=False).tolist())
        samples.append(
            EpisodeSample(
                file_path=file_path,
                episode_index=ep_index,
                task_index=task_index,
                start_ts=start_ts,
            )
        )

    print(f"[data] Total episodes: {len(episode_infos)}, val episodes: {n_val}")
    if task_filter is not None:
        print(f"[data] After task filter ({sorted(task_filter)}): {len(samples)} episodes")
    else:
        print(f"[data] Sampled val episodes: {len(samples)}")
    return samples


def build_input_video(image_t: np.ndarray) -> torch.Tensor:
    h, w = image_t.shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    frames = np.stack([image_t, blank, blank, blank, blank], axis=0)
    return torch.from_numpy(frames[None]).permute(0, 4, 1, 2, 3)


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


def build_data_batch(
    video: torch.Tensor,
    t5_emb: torch.Tensor,
    action: torch.Tensor,
    delay_scalar: torch.Tensor,
    fps: float,
) -> dict:
    b, _, _, h, w = video.shape
    data_batch = {
        "dataset_name": "video_data",
        "video": video.cuda(),
        "fps": torch.full((b,), float(fps), dtype=torch.float32),
        "padding_mask": torch.zeros(b, 1, h, w),
        "image_size": torch.full((b, 4), float(h), dtype=torch.float32),
        "num_frames": _NUM_VIDEO_FRAMES,
        "num_conditional_frames": _NUM_LATENT_COND,
        "t5_text_embeddings": t5_emb,
        "action": action,
        "delay_scalar": delay_scalar,
    }
    for key, value in list(data_batch.items()):
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            data_batch[key] = value.cuda().to(dtype=torch.bfloat16)
    return data_batch


def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    benchmark = _normalize_benchmark(args.benchmark)
    data_root = args.data_root or _DEFAULT_DATA_ROOT
    task_indices = _resolve_task_indices(benchmark, args.task_indices)
    experiment_name = _infer_experiment_name(args.experiment, benchmark)
    t5_emb_path = _resolve_t5_emb_path(args.t5_emb_path, data_root)
    fps = read_dataset_fps(data_root)

    os.environ["METAWORLD_DATA_ROOT"] = str(data_root)
    if t5_emb_path:
        os.environ["METAWORLD_T5_EMB_PATH"] = str(t5_emb_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distilled world-model evaluation.")

    model = load_distilled_model(
        ckpt_path=str(args.ckpt),
        experiment_name=experiment_name,
        lightvae_pth=str(args.lightvae_pth),
        lightx2v_root=str(args.lightx2v_root) if args.lightx2v_root else None,
        use_batched_vae=bool(args.use_batched_vae),
        extra_experiment_opts=args.experiment_opts or [],
    )

    lpips_fn = None
    if _HAS_LPIPS:
        lpips_fn = _lpips_lib.LPIPS(net="alex").cuda().eval()
        print("[info] LPIPS (AlexNet) loaded.")

    task_descriptions = load_task_descriptions(data_root)
    t5_cache: Dict[str, torch.Tensor] = {}
    if t5_emb_path:
        emb_dict = load_t5_embeddings(t5_emb_path)
        if emb_dict is not None:
            for key, value in emb_dict.items():
                t5_cache[str(key)] = torch.as_tensor(value, dtype=torch.float32)
            print(f"[info] Loaded precomputed T5 embeddings: {len(t5_cache)} tasks")

    def _get_t5_emb(prompt: str) -> torch.Tensor:
        if prompt in t5_cache:
            emb = t5_cache[prompt]
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            return emb
        print(f"[warn] T5 cache miss for prompt: {prompt[:50]!r}. Using zeros.")
        return torch.zeros(1, 512, 1024, dtype=torch.float32)

    episode_samples = build_val_episode_samples(
        data_root=data_root,
        val_ratio=0.1,
        split_seed=0,
        samples_per_episode=args.samples_per_episode,
        sample_seed=args.seed,
        task_indices=task_indices,
    )

    delays = list(range(1, _MAX_DELAY + 1))
    num_steps_list: List[int] = args.num_steps
    results_by_steps: Dict[int, Dict[int, defaultdict]] = {
        ns: {delay: defaultdict(list) for delay in delays}
        for ns in num_steps_list
    }
    results_per_task_by_steps: Dict[int, Dict[int, Dict[int, defaultdict]]] = {
        ns: {} for ns in num_steps_list
    }
    all_task_indices_seen: set[int] = set()

    if args.save_images:
        pathlib.Path(args.save_images).mkdir(parents=True, exist_ok=True)

    total_windows = sum(len(sample.start_ts) for sample in episode_samples)
    print(
        f"\nStarting MetaWorld eval: {total_windows} windows x {len(delays)} delays "
        f"x {len(num_steps_list)} num_steps = "
        f"{total_windows * len(delays) * len(num_steps_list)} forwards\n"
    )

    done = 0
    t_start = time.time()

    for sample in episode_samples:
        prompt = task_descriptions.get(sample.task_index, f"metaworld task {sample.task_index}")
        t5_emb = _get_t5_emb(prompt)
        all_task_indices_seen.add(sample.task_index)
        for ns in num_steps_list:
            results_per_task_by_steps[ns].setdefault(
                sample.task_index,
                {delay: defaultdict(list) for delay in delays},
            )

        df = pd.read_parquet(sample.file_path)
        ep_df = (
            df[df[_EPISODE_KEY] == sample.episode_index]
            .sort_values(_FRAME_KEY)
            .reset_index(drop=True)
        )
        ep_name = f"{sample.file_path.stem}_ep{sample.episode_index:04d}"
        t_ep = len(ep_df)

        for start_t in sample.start_ts:
            image_t = _decode_image(ep_df[_IMAGE_KEY].iloc[start_t], image_size=args.image_size)
            video_input = build_input_video(image_t)

            for delay in delays:
                pred_t = start_t + delay
                if pred_t >= t_ep:
                    continue

                image_gt = _decode_image(ep_df[_IMAGE_KEY].iloc[pred_t], image_size=args.image_size)
                action_rows = np.stack(
                    [
                        np.asarray(ep_df[_ACT_KEY].iloc[start_t + offset], dtype=np.float32)
                        for offset in range(delay)
                    ],
                    axis=0,
                )
                action, delay_scalar = build_action_inputs(action_rows, delay=delay, batch_size=1)
                data_batch = build_data_batch(video_input, t5_emb, action, delay_scalar, fps=fps)

                for ns in num_steps_list:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    time_start = time.perf_counter()
                    with torch.no_grad():
                        latents = model.generate_samples_from_batch(
                            data_batch,
                            n_sample=1,
                            seed=args.seed,
                            num_steps=ns,
                        )
                        video_out = model.decode(latents)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - time_start) * 1000

                    image_pred = _tensor_to_uint8(video_out[0, :, 1])
                    psnr = compute_psnr(image_pred, image_gt)
                    ssim = compute_ssim(image_pred, image_gt)

                    results_by_steps[ns][delay]["psnr"].append(psnr)
                    results_by_steps[ns][delay]["ssim"].append(ssim)
                    results_by_steps[ns][delay]["inference_ms"].append(elapsed_ms)
                    task_bucket = results_per_task_by_steps[ns][sample.task_index][delay]
                    task_bucket["psnr"].append(psnr)
                    task_bucket["ssim"].append(ssim)

                    if lpips_fn is not None:
                        with torch.no_grad():
                            pred_tensor = video_out[0:1, :, 1].float()
                            gt_tensor = (
                                torch.from_numpy(image_gt).float() / 127.5 - 1
                            ).permute(2, 0, 1).unsqueeze(0).cuda()
                            lpips_value = float(lpips_fn(pred_tensor, gt_tensor).item())
                        results_by_steps[ns][delay]["lpips"].append(lpips_value)
                        task_bucket["lpips"].append(lpips_value)

                    if args.save_images and ns == num_steps_list[0]:
                        tag = f"{ep_name}_t{start_t:04d}_d{delay}"
                        Image.fromarray(np.hstack([image_pred, image_gt])).save(
                            f"{args.save_images}/{tag}_pred_vs_gt.png"
                        )

            done += 1
            if done % 10 == 0 or done == total_windows:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_windows - done) if done < total_windows else 0
                print(f"  [{done}/{total_windows}] elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min")

    def _summarize(bucket: defaultdict) -> dict:
        n = len(bucket["psnr"])
        if n == 0:
            return {}
        summary = {
            "n": n,
            "psnr": round(float(np.nanmean(bucket["psnr"])), 4),
        }
        if _HAS_SKIMAGE and bucket["ssim"]:
            summary["ssim"] = round(float(np.nanmean(bucket["ssim"])), 4)
        if _HAS_LPIPS and bucket["lpips"]:
            summary["lpips"] = round(float(np.nanmean(bucket["lpips"])), 4)
        if bucket["inference_ms"]:
            summary["avg_inference_ms"] = round(float(np.nanmean(bucket["inference_ms"])), 1)
        return summary

    def _print_table(title: str, result_dict: Dict[int, defaultdict]):
        print(f"\n{'=' * 65}")
        print(title)
        print(f"{'=' * 65}")
        header = f"{'d':>4} | {'PSNR':>10}"
        if _HAS_SKIMAGE:
            header += f" | {'SSIM':>10}"
        if _HAS_LPIPS:
            header += f" | {'LPIPS':>10}"
        header += f" | {'ms/step':>8} | {'N':>5}"
        print(header)
        print("-" * len(header))
        for delay in delays:
            summary = _summarize(result_dict[delay])
            if not summary:
                continue
            row = f"{delay:>4} | {summary['psnr']:>10.3f}"
            if _HAS_SKIMAGE and "ssim" in summary:
                row += f" | {summary['ssim']:>10.4f}"
            if _HAS_LPIPS and "lpips" in summary:
                row += f" | {summary['lpips']:>10.4f}"
            row += f" | {summary.get('avg_inference_ms', float('nan')):>8.1f} | {summary['n']:>5}"
            print(row)

    summary: Dict[str, dict] = {}
    for ns in num_steps_list:
        _print_table(
            f"MetaWorld distilled model num_steps={ns} - all tasks by delay",
            results_by_steps[ns],
        )
        step_key = f"steps={ns}"
        summary[step_key] = {}
        for delay in delays:
            s = _summarize(results_by_steps[ns][delay])
            if s:
                summary[step_key][f"d={delay}"] = s

    per_task_summary: Dict[str, Dict[str, dict]] = {}
    if len(all_task_indices_seen) > 1:
        for ns in num_steps_list:
            for task_index in sorted(all_task_indices_seen):
                desc = task_descriptions.get(task_index, f"task {task_index}")
                _print_table(
                    f"steps={ns} Task {task_index} [{desc[:50]}]",
                    results_per_task_by_steps[ns].get(
                        task_index,
                        {delay: defaultdict(list) for delay in delays},
                    ),
                )
                task_key = f"task{task_index}"
                per_task_summary.setdefault(task_key, {})
                per_task_summary[task_key][f"steps={ns}"] = {}
                for delay in delays:
                    s = _summarize(
                        results_per_task_by_steps[ns]
                        .get(task_index, {})
                        .get(delay, defaultdict(list))
                    )
                    if s:
                        per_task_summary[task_key][f"steps={ns}"][f"d={delay}"] = s

    print(f"\n{'=' * 65}")
    print(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "ckpt": str(args.ckpt),
            "data_root": str(data_root),
            "benchmark": benchmark,
            "experiment": experiment_name,
            "task_indices": task_indices,
            "val_ratio": 0.1,
            "split_seed": 0,
            "sample_seed": args.seed,
            "samples_per_episode": args.samples_per_episode,
            "num_steps_evaluated": num_steps_list,
            "delays_evaluated": delays,
            "t5_emb_path": str(t5_emb_path) if t5_emb_path else None,
            "tokenizer": "wan2pt1_lightvae_tokenizer",
            "lightvae_pth": str(args.lightvae_pth),
            "lightx2v_root": str(args.lightx2v_root) if args.lightx2v_root else None,
            "use_batched_vae": bool(args.use_batched_vae),
            "fps": fps,
            "image_size": args.image_size,
        }
        out = {"meta": meta, "results": summary}
        if per_task_summary:
            out["results_per_task"] = per_task_summary
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {out_path}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline eval for MetaWorld DMD2-distilled wm4vla world models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=pathlib.Path,
        required=True,
        help="Distilled model_ema_bf16.pt converted by convert_distcp_to_pt.py.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"MetaWorld data root. Defaults to METAWORLD_DATA_ROOT or {_DEFAULT_DATA_ROOT}.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mt50", "all", "task0"],
        default="mt50",
        help="MetaWorld benchmark selector. task0 also defaults --task-indices to [0].",
    )
    parser.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="TASK_IDX",
        help="Optional task filter. Overrides the task0 default when provided.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=f"Override DMD2 experiment name. Defaults to {_EXPERIMENT_MT50} or {_EXPERIMENT_TASK0}.",
    )
    parser.add_argument(
        "--t5-emb-path",
        type=str,
        default=None,
        help="Precomputed T5 embedding pkl. Defaults to METAWORLD_T5_EMB_PATH or data_root/meta/t5_embeddings.pkl.",
    )
    parser.add_argument(
        "--lightvae-pth",
        type=str,
        default=_DEFAULT_LIGHTVAE_PTH,
        help="LightVAE checkpoint used by wan2pt1_lightvae_tokenizer.",
    )
    parser.add_argument(
        "--lightx2v-root",
        type=str,
        default=_DEFAULT_LIGHTX2V_ROOT,
        help="LightX2V repo root used to import the LightVAE WanVAE implementation.",
    )
    parser.add_argument(
        "--use-batched-vae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LightVAE batched encode/decode path.",
    )
    parser.add_argument(
        "--experiment-opts",
        type=str,
        nargs="*",
        default=[],
        help="Additional Hydra overrides appended after the LightVAE overrides.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        metavar="N",
        help="Student sampling steps to evaluate.",
    )
    parser.add_argument(
        "--samples-per-episode",
        type=int,
        default=20,
        help="Number of start_t samples per val episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed. Val split seed is fixed to 0 to match training.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size used for evaluation resize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--save-images",
        type=str,
        default=None,
        help="Optional directory for pred-vs-GT comparison images. Only the first num_steps is saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
