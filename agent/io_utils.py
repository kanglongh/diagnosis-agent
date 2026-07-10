"""信号 IO 工具 (从 vibrolab 粘贴, 自包含)"""
import numpy as np


def make_windows(signal: np.ndarray, window: int = 2048, step: int = 2048) -> np.ndarray:
    """滑窗切分: (N,) → (n_win, window)."""
    n = len(signal)
    if n < window:
        return np.empty((0, window), dtype=signal.dtype)
    n_win = (n - window) // step + 1
    out = np.zeros((n_win, window), dtype=signal.dtype)
    for i in range(n_win):
        out[i] = signal[i * step: i * step + window]
    return out
