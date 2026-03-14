#!/usr/bin/env python3
"""precompute_libero_t5.py

为 LIBERO 数据集的任务描述预计算 T5-11B 文本嵌入，保存为 pickle 文件。

格式：
    {task_description_str: tensor(1, 512, 1024, bfloat16)}  → t5_embeddings.pkl

使用方法（在项目根目录）：

  # 使用本地已下载的模型目录（推荐，用 huggingface-cli download --local-dir 下载的）：
  CUDA_VISIBLE_DEVICES=7 python scripts/precompute_libero_t5.py \\
      --data-root /path/to/lerobot--libero_10_image@v2.0 \\
      --t5-model /home/kyji/public/models/google-t5-11b

  # 在线下载（首次约 42 GB）：
  CUDA_VISIBLE_DEVICES=7 python scripts/precompute_libero_t5.py \\
      --data-root /path/to/lerobot--libero_10_image@v2.0
"""

import argparse
import json
import pathlib
import pickle
import sys

import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute T5-11B embeddings for LIBERO task descriptions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="lerobot/lerobot--libero_10_image@v2.0",
        help="LIBERO LeRobot dataset root (contains meta/tasks.jsonl)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pkl path. Default: <data_root>/meta/t5_embeddings.pkl",
    )
    p.add_argument(
        "--t5-model",
        type=str,
        default=None,
        help=(
            "T5-11B 模型路径（支持两种方式）：\n"
            "  本地目录：直接传绝对路径，如 /home/kyji/public/models/google-t5-11b\n"
            "            （用 huggingface-cli download --local-dir 下载的）\n"
            "  在线下载：留空，从 HuggingFace 自动下载到默认缓存 (~/.cache/huggingface)"
        ),
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for T5 encoding (default 512, matching Cosmos inference)",
    )
    return p.parse_args()


def main():
    _root = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    args = parse_args()

    data_root = pathlib.Path(args.data_root)
    tasks_file = data_root / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {tasks_file}")

    out_path = (
        pathlib.Path(args.output)
        if args.output
        else (data_root / "meta" / "t5_embeddings.pkl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 读取任务描述 ──────────────────────────────────────────────────────────
    tasks = [json.loads(line) for line in tasks_file.read_text().strip().splitlines()]
    tasks.sort(key=lambda x: x["task_index"])
    print(f"Found {len(tasks)} tasks:")
    for t in tasks:
        print(f"  [{t['task_index']}] {t['task']}")

    # ── 加载 T5-11B ───────────────────────────────────────────────────────────
    from cosmos_predict2._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder

    t5_model_arg = args.t5_model
    local_path = pathlib.Path(t5_model_arg) if t5_model_arg else None

    if local_path is not None and local_path.is_dir():
        # 本地目录：直接作为 model_name，跳过网络下载
        print(f"\n加载 T5-11B（本地）: {local_path}")
        encoder = CosmosT5TextEncoder(
            model_name=str(local_path),
            device="cuda",
            local_files_only=True,
        )
    else:
        # 在线下载（HuggingFace 默认缓存）
        print("\n加载 T5-11B（在线）: google-t5/t5-11b")
        print("  (首次运行会从 HuggingFace 下载约 42 GB，请耐心等待)")
        encoder = CosmosT5TextEncoder(
            model_name="google-t5/t5-11b",
            device="cuda",
        )

    # ── 逐条编码，存为 {task_str: tensor(1, 512, 1024)} ──────────────────────
    t5_embeddings = {}
    for t in tqdm(tasks, desc="Encoding"):
        task_str = t["task"]
        emb = encoder.encode_prompts(task_str, max_length=args.max_length)
        # emb: (1, 512, 1024) float32 on CUDA
        t5_embeddings[task_str] = emb.to(dtype=torch.bfloat16).cpu()

    # 验证 shape
    sample_emb = next(iter(t5_embeddings.values()))
    print(f"\n嵌入 shape: {sample_emb.shape}  dtype: {sample_emb.dtype}")
    assert sample_emb.shape == (1, 512, 1024), f"Unexpected shape: {sample_emb.shape}"

    # ── 保存 pkl ──────────────────────────────────────────────────────────────
    with open(out_path, "wb") as f:
        pickle.dump(t5_embeddings, f)
    print(f"\n已保存 {len(t5_embeddings)} 条嵌入 → {out_path}")

    # ── 同时保存一份 task_index → task_str 映射，方便 dataset 查表 ────────────
    index_map = {t["task_index"]: t["task"] for t in tasks}
    map_path = out_path.parent / "task_index_to_str.json"
    map_path.write_text(json.dumps(index_map, ensure_ascii=False, indent=2))
    print(f"已保存 task_index 映射 → {map_path}")


if __name__ == "__main__":
    main()
