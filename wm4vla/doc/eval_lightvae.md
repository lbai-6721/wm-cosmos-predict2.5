# wm-cosmos LightVAE 对齐说明

本文档记录 `wm-cosmos-predict2.5` 相对于 `cosmos-predict2.5/wm4vla/doc/eval_lightvae.md` 已完成的对齐改动，以及在 wm-cosmos 侧的使用方式。

## 已完成的代码对齐

- `cosmos_predict2/_src/predict2/tokenizers/wan2pt1_lightvae.py`
  - 新增 LightVAE tokenizer 接口（含 lazy import、dtype 对齐、batch decode 兼容、decode timing）。
  - 新增 `use_batched_vae`（默认 `True`）：
    - 在 5D 输入下优先走 batched encode/decode 快路径；
    - 当 `parallel=True` 时自动 warning 并回退到 legacy per-sample 路径（稳态兼容）。
- `cosmos_predict2/_src/interactive/configs/registry_defaults/tokenizer.py`
  - 注册 `tokenizer/wan2pt1_lightvae_tokenizer`，修复 distill 分支 `MissingConfigException`。
  - 默认注入 `use_batched_vae=True`，支持 Hydra 覆盖。
- `wm4vla/scripts/eval_world_model.py`
  - 增加命令行切换参数：
    - `--tokenizer-backend {wan2pt1,lightvae}`
    - `--tokenizer-vae-pth`
    - `--lightx2v-root`
    - 兼容别名 `--use-lightvae` / `--lightvae-pth`
  - 支持通过 `experiment_opts` 把 tokenizer override 透传给 `Video2WorldInference`。
- `cosmos_predict2/_src/predict2/inference/video2world.py`
  - decode 阶段增加计时日志，并在计时前后调用 `torch.cuda.synchronize()`。
- `wm4vla/configs/experiments.py`
  - 支持双路径可切换：
    - `WM4VLA_WAN21_VAE_PATH`
    - `WM4VLA_LIGHTVAE_PATH`
    - `WM4VLA_VAE_BACKEND={wan2pt1,lightvae}`

## 使用方式

### 1) wm-cosmos 离线评估脚本

使用 Wan2.1（默认推荐）：

```bash
python wm4vla/scripts/eval_world_model.py \
  --ckpt /home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/benchmark/pi_libero_one_for_all_16000/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_pi_libero_256_all/checkpoints/iter_000002500/model_ema_bf16.pt \
  --tokenizer-backend wan2pt1 \
  --tokenizer-vae-pth /home/kyji/public/models/lightx2v/vae/Wan2.1_VAE.pth \
  --output outputs/eval_wm/task0_wan21.json
```

使用 LightVAE：

```bash
python wm4vla/scripts/eval_world_model.py \
  --ckpt /home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/benchmark/pi_libero_one_for_all_16000/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_pi_libero_256_all/checkpoints/iter_000002500/model_ema_bf16.pt \
  --tokenizer-backend lightvae \
  --tokenizer-vae-pth /home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth \
  --lightx2v-root /home/kyji/storage_net/tmp/lbai/LightX2V \
  --output outputs/eval_wm/task0_lightvae_20260419.json
```

评估单个delay
```
# 使用wan vae

CKPT=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/benchmark/pi_libero_10_delay8_only_30000/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_pi_libero_256_10/checkpoints/iter_000002500/model_ema_bf16.pt \
OUTPUT=outputs/eval_distill/libero_10_delay8_wan_20260421.json \
SAVE_IMAGES=outputs/eval_distill/libero_10_delay8_wan_imgs_20260421 \
bash wm4vla/scripts/eval_distilled_delay8_pi_libero.sh

# 使用lightvae

CKPT=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/benchmark/pi_libero_10_delay8_only_30000/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_pi_libero_256_10/checkpoints/iter_000002500/model_ema_bf16.pt \
  BENCHMARK=10 \
  TOKENIZER_BACKEND=lightvae \
  LIGHTVAE_PTH=/home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth \
  LIGHTX2V_ROOT=/home/kyji/storage_net/tmp/lbai/LightX2V \
  OUTPUT=outputs/eval_distill/libero_10_delay8_lightvae_20260421.json \
  SAVE_IMAGES=outputs/eval_distill/libero_10_delay8_lightvae_imgs_20260421 \
  bash wm4vla/scripts/eval_distilled_delay8_pi_libero.sh

```


如需关闭 5D 批量路径（调试用）：

```bash
python wm4vla/scripts/eval_world_model.py \
  ... \
  --tokenizer-backend lightvae \
  --tokenizer-vae-pth /home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth \
  --experiment-opts model.config.tokenizer.use_batched_vae=false
```

### 2) WM4VLA 在线评估（vlash 入口）

已支持 teacher / distilled 共用参数：

- `--wm_tokenizer_backend {wan2pt1,lightvae}`
- `--wm_tokenizer_vae_pth`
- `--wm_lightx2v_root`

注意：若 `wm_distilled=true` 且使用 LightVAE，必须保证当前 `wm-cosmos-predict2.5` 已包含本文件列出的 tokenizer 注册改动。

## 常见错误

### `MissingConfigException: Could not find 'tokenizer/wan2pt1_lightvae_tokenizer'`

- 原因：`wm-cosmos-predict2.5` 侧未注册 lightvae tokenizer。
- 处理：确认以下文件已存在并生效：
  - `cosmos_predict2/_src/predict2/tokenizers/wan2pt1_lightvae.py`
  - `cosmos_predict2/_src/interactive/configs/registry_defaults/tokenizer.py`

### `size mismatch for WanVAE_...`

- 原因：backend 与权重不匹配。
- 正确配对：
  - `wan2pt1` ↔ `Wan2.1_VAE.pth`
  - `lightvae` ↔ `lightvaew2_1.pth`
