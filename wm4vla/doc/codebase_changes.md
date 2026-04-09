# wm-cosmos-predict2.5 改造说明（wm4vla）

本文档记录在 `wm-cosmos-predict2.5` 中，为复现 `cosmos-predict2.5/wm4vla` 训练流程所做的代码改动。

---

## 1. 改造目标

- 在当前仓库中完整接入 `wm4vla` 的 skip-dynamics 训练/评估路径
- 保持 `cosmos_predict2` 原有入口不变（`scripts.train`、Hydra 配置、模型结构）
- 新增 Kinetix 与 LIBERO（LeRobot）数据路径与实验配置

---

## 2. 新增目录与模块

新增 `wm4vla/` 包：

- `wm4vla/datasets/`
  - `dataset_kinetix.py`
  - `dataset_libero.py`
  - `dataset_lerobot_libero.py`
- `wm4vla/configs/`
  - `data_registry.py`（数据注册）
  - `experiments.py`（实验注册）
- `wm4vla/scripts/`
  - `precompute_libero_t5.py`
  - `eval_world_model.py`
  - `visualize_wm.py`
- `wm4vla/doc/`
  - `train_wm_pixels.md`
  - 本文件 `codebase_changes.md`

---

## 3. 对 `cosmos_predict2` 的接线改动

### 3.1 数据注册接线

文件：`cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`

- 在 `register_training_and_val_data()` 末尾新增：
  - `from wm4vla.configs.data_registry import register_wm4vla_data`
  - `register_wm4vla_data()`

效果：Hydra 可识别以下 dataloader：
- `kinetix_5frame_128_train/val`
- `lerobot_libero_dual_cam_256_train/val`
- `lerobot_libero_dual_cam_256_task0_train/val`
- `lerobot_libero_dual_cam_256_task01_train/val`

### 3.2 实验注册接线

文件：`cosmos_predict2/experiments/base/action.py`

- 追加注册：
  - `from wm4vla.configs.experiments import register_wm4vla_experiments`
  - `register_wm4vla_experiments()`

新增可用实验：
- `ac_kinetix_pixels_2b`
- `ac_libero_pixels_2b`
- `ac_libero_lerobot_256_pixels_2b`
- `ac_libero_lerobot_256_pixels_2b_task0`
- `ac_libero_lerobot_256_pixels_2b_task01`

### 3.3 conditioner 改为纯条件训练

文件：`cosmos_predict2/_src/predict2/action/configs/action_conditioned/conditioner.py`

- `ActionConditionedConditionerConfig` 中：
  - `text.dropout_rate = 0.0`
  - `use_video_condition.dropout_rate = 0.0`

效果：训练时关闭 CFG dropout，保持纯条件训练。

### 3.4 guidance=0 推理加速

文件：`cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py`

- 在 `velocity_fn` 中新增分支：
  - `if guidance == 0: return cond_v`

效果：推理跳过 uncond 分支前向，速度提升（约接近 2x）。

---

## 4. 兼容与转发改动

### 4.1 dataset re-export

新增：

- `cosmos_predict2/_src/predict2/action/datasets/dataset_kinetix.py`
- `cosmos_predict2/_src/predict2/action/datasets/dataset_lerobot_libero.py`

它们仅做 re-export 到 `wm4vla.datasets.*`，保证旧导入路径可用。

### 4.2 顶层脚本转发

新增：

- `scripts/precompute_libero_t5.py`
- `scripts/eval_world_model.py`
- `scripts/visualize_wm.py`

三者均为 forwarding shim，转发到 `wm4vla/scripts/` 对应实现，保持命令习惯一致。

---

## 5. 关键逻辑变更总结

- **数据组织方式**：从原始 bridge/action-chunk 任务，扩展到 skip-dynamics 任务定义。
- **帧布局**：
  - Kinetix：9 帧（`state_t=3`）
  - LIBERO LeRobot：17 帧（`state_t=5`）
- **条件输入**：
  - action 统一拼接归一化 delay：`[action ; d/(max_delay-1)]`
  - 文本条件由 `t5_text_embeddings` 提供（Kinetix 为零向量占位）
- **推理策略**：默认建议 `guidance=0` 纯条件推理，提升吞吐。

---

## 6. 与原仓库的差异注意点

- 当前仓库默认 LeRobot 根路径使用：
  - `/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/lerobot/...`
- 原仓库有些路径为 `cosmos-predict2.5`，这里已替换为 `wm-cosmos-predict2.5` 对应路径。
- 若你的数据不在默认目录，请通过环境变量覆盖：
  - `LEROBOT_LIBERO_DATA_ROOT`
  - `LEROBOT_LIBERO_T5_EMB_PATH`
  - `KINETIX_DATA_PIXELS_DIR`

---

## 7. 验证建议

1. 先跑冒烟训练：
   - `experiment=ac_kinetix_pixels_2b`
   - `experiment=ac_libero_lerobot_256_pixels_2b_task0`
2. 再跑评估脚本验证输出格式：
   - `python scripts/eval_world_model.py ...`
3. 最后跑可视化检查帧索引是否正确：
   - `python scripts/visualize_wm.py ...`

---

## 8. DMD2 蒸馏支持

基于 teacher 模型进行 DMD2（Distribution Matching Distillation 2）蒸馏，目标是将推理步数从 50 步缩减到 1-4 步，加速 VLA 策略闭环控制的异步推理。

### 8.1 `make_experiment` 新增 `tokenizer` 参数

文件：`cosmos_predict2/_src/interactive/configs/registry_experiment/experiments_dmd2_predict2p5.py`

- 在 `make_experiment` 函数签名中新增 `tokenizer: str = "wan2pt1_tokenizer"` 参数
- 默认值 `"wan2pt1_tokenizer"` 保持向后兼容，存量 Bridge 实验不受影响
- wm4vla 蒸馏实验传入 `"wan2pt1_tokenizer"`，与当前 teacher 训练实际使用的 VAE 保持一致

### 8.2 新增两个蒸馏实验配置

文件：`cosmos_predict2/_src/interactive/configs/registry_experiment/experiments_dmd2_ac_predict2p5.py`

新注册实验：

| 实验名 | 分辨率 | 帧数 | state_t | action_dim | 条件帧数 |
|--------|--------|------|---------|------------|---------|
| `dmd2_trigflow_distill_wm_libero_lerobot_256_task0` | 256×256 | 5 帧 | 2 | 8 | 1（每个 paired 样本各自的 `view_t`） |
| `dmd2_trigflow_distill_wm_kinetix_128_9frame` | 128×128 | 9 帧 | 3 | 7 | 2（blank + obs_t） |

**与原 Bridge 实验相比，关键差异配置及原因：**

| 配置项 | 设置值 | 原因 |
|--------|--------|------|
| `tokenizer` | `wan2pt1_tokenizer` | wm4vla teacher 实际使用 wan2pt1（`DEFAULT_CHECKPOINT.experiment` 将 wan2pt2 覆盖为 wan2pt1） |
| `model.config.tokenizer.vae_pth` | 本地 HuggingFace 缓存路径 | 跳过 S3 下载，避免 `credentials/s3_training.secret` 依赖 |
| `multiply_noise_by_video_len` | `False` | teacher 训练 `adjust_video_noise=False`，噪声乘数为 1.0，必须匹配 |
| `use_clean_cond_timesteps` | `False` | teacher 训练 `conditional_frame_timestep=-1.0`，不对条件帧调整 timestep |
| `teacher_guidance` | `0` | teacher 纯条件训练（无 CFG dropout），跳过 uncond 前向，节省显存 |
| `text_encoder_config` | `None` | 使用数据 batch 中预计算的 T5 embedding，无需在线编码器 |
| `fsdp_shard_size` | `4` | 4-GPU 单节点，不能超过 GPU 数量 |
| `net.use_crossattn_projection` | `False` | 与 teacher 训练架构一致 |
| `num_action_per_chunk` | `8` | 当前 LIBERO teacher 使用 8 槽 masked action prefix |

当前 LIBERO 蒸馏还必须同步新的条件接口：

- `action: [B, 8, 8]`，每个 slot 为 `[raw_action(7), valid_mask]`
- `delay_scalar: [B, 1]` 独立输入
- 旧版 `action: [B, 1, 8] = [a_{t+d}; normalized_delay]` 只用于兼容历史实验，不再是主线 teacher 语义

网络类型均使用 `cosmos_v1_2B_action_conditioned_{student/teacher/fake_score}`（对应 `ActionConditionedMinimalV1LVGDiT`），与 wm4vla teacher 架构匹配。

### 8.3 新增蒸馏说明文档

新增 `wm4vla/doc/distillation.md`，内容包括：

- DMD2 算法概述（Generator step / Critic step 交替更新）
- 环境变量配置
- Teacher checkpoint 格式说明及 DCP→`.pt` 转换命令
- 各配置差异的详细原因说明（对照表）
- 两个蒸馏实验的完整启动命令
- 蒸馏输出 checkpoint 转换命令
- 调参建议（batch_size、student_update_freq、训练步数）
