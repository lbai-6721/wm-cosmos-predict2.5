# Kinetix Pixels 数据集说明

本文件说明 `src/generate_pixels_dataset.py` 当前生成的 **trajectory chunk** 数据格式和使用方式。

## 1. 目标

生成按时间步组织的轨迹数据，适配 world model / VLA 后训练（如需要 `observation_img + action + symbolic/proprio` 的场景）。

- 采样逻辑复用 `src/generate_data.py`（专家策略采动作）
- 每个 step 从 `env_state` 渲染 `observation_img`
- 不切 clip、不做滑窗重复存储（减小磁盘占用）
- 按 chunk 写成压缩 `npz`

## 2. 当前存储格式（重要）

当前脚本已改为：
- **trajectory chunk (`.npz`)**
- **不是** WebDataset tar
- **不是** clip 数据集

保留字段：
- 图像观测 `observation_img`
- symbolic 观测 `obs`
- `action/done/solved/return_/length`

## 3. 默认配置

- 分辨率：`128 x 128`
- 每个 chunk 的时间长度：`batch_size`（默认 `16`）
- 并行环境数：`num_envs`（默认 `16`，建议根据显存调）

## 4. 输出目录结构

脚本输出到 `<run_path>/data_pixels/`：

```text
<run_path>/data_pixels/
  meta.json
  worker_0.log
  worker_1.log
  ...
  trajectories/
    worlds_l_grasp_easy/
      chunk_000000.npz
      chunk_000001.npz
      ...
    worlds_l_catapult/
      chunk_000000.npz
      ...
```

## 5. 单个 chunk 的字段与形状

每个 `chunk_XXXXXX.npz` 包含：

- `observation_img`: `uint8`, shape `[T, E, H, W, 3]`
- `obs`: `float32`, shape `[T, E, obs_dim]`
- `action`: `float32`, shape `[T, E, 6]`
- `done`: `bool`, shape `[T, E]`
- `solved`: `float32`, shape `[T, E]`
- `return_`: `float32`, shape `[T, E]`
- `length`: `float32`, shape `[T, E]`

其中：
- `T = batch_size`（默认 16）
- `E = num_envs`（默认 16）
- `H = W = image_size`（默认 128）

## 6. 运行方式

### 6.1 单进程

```bash
uv run src/generate_pixels_dataset.py \
  --config.run-path <run_path>
```

常用参数：

```bash
--config.image-size 128
--config.num-envs 16
--config.batch-size 16
```

### 6.2 8 卡并行（推荐）

```bash
bash scripts/run_generate_pixels_parallel.sh <run_path> 8 16 16
```

参数含义：
- 第 2 个参数：GPU/worker 数（如 8）
- 第 3 个参数：每个 worker 的 `num_envs`
- 第 4 个参数：每个 worker 的 `batch_size`

## 7. 关键前置条件

`<run_path>` 需要是 expert 训练输出目录，至少包含：

- `seed_*/<step>/stats/*.json`
- `seed_*/<step>/policies/*.pkl`

否则脚本无法选专家策略并采集数据。
