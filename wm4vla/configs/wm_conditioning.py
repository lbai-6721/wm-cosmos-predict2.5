"""Unified wm4vla conditioning defaults for LeRobot LIBERO training."""

DEFAULT_MAX_DELAY = 8

# Training excludes delay=0 because the real observation is used directly.
TRAIN_DELAY_MIN = 1
TRAIN_DELAY_MAX = DEFAULT_MAX_DELAY

# Inference may request delay=0, which should bypass the world model.
INFER_DELAY_MIN = 0
INFER_DELAY_MAX = DEFAULT_MAX_DELAY

ACTION_CHUNK_LEN = DEFAULT_MAX_DELAY
BYPASS_WM_WHEN_DELAY_ZERO = True
ACTION_MASK_DIM = 1
DELAY_SCALAR_DIM = 1

DEFAULT_EVAL_DELAYS = tuple(range(INFER_DELAY_MIN, INFER_DELAY_MAX + 1))

# LeRobot LIBERO uses 7-dim actions. Per-slot feature = raw action + valid mask.
LEROBOT_LIBERO_RAW_ACTION_DIM = 7
LEROBOT_LIBERO_ACTION_SLOT_DIM = LEROBOT_LIBERO_RAW_ACTION_DIM + ACTION_MASK_DIM

# Backward-compatible aliases for the current single supported dataset.
LIBERO_RAW_ACTION_DIM = LEROBOT_LIBERO_RAW_ACTION_DIM
LIBERO_ACTION_SLOT_DIM = LEROBOT_LIBERO_ACTION_SLOT_DIM
