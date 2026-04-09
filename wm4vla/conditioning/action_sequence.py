"""Helpers for packing fixed-length masked action prefixes."""

from __future__ import annotations

import numpy as np


def normalize_delay_scalar(delay: int, max_delay: int) -> np.ndarray:
    """Normalize a training delay to one scalar in [0, 1]."""
    if max_delay <= 1:
        raise ValueError(f"max_delay must be > 1, got {max_delay}")
    if delay < 1 or delay > max_delay:
        raise ValueError(f"delay must be in [1, {max_delay}], got {delay}")
    return np.asarray([(delay - 1) / (max_delay - 1)], dtype=np.float32)


def pack_masked_action_sequence(
    actions: np.ndarray,
    delay: int,
    chunk_len: int,
) -> np.ndarray:
    """Pack a variable-length action prefix into a fixed-length tensor.

    Output layout per slot:
      [raw_action..., action_mask]
    """
    if delay < 0 or delay > chunk_len:
        raise ValueError(f"delay must be in [0, {chunk_len}], got {delay}")

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [N, action_dim], got {actions.shape}")
    if actions.shape[0] < delay:
        raise ValueError(f"expected at least {delay} actions, got {actions.shape[0]}")

    action_dim = int(actions.shape[1])
    packed = np.zeros((chunk_len, action_dim + 1), dtype=np.float32)

    if delay > 0:
        packed[:delay, :action_dim] = actions[:delay]
        packed[:delay, action_dim] = 1.0

    return packed
