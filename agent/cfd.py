"""
CFD 120 维特征提取 (从 vibrolab 粘贴, 唯一来源)

五模块: time[0:12] + freq[12:27] + band[27:91] + peaks[91:103] + cepstral[103:120]

本文件自包含, 不依赖 vibrolab 包. 所有数值与 vibrolab.features.extract_cfd 一致.
"""
from __future__ import annotations
import numpy as np

FS_DEFAULT = 12000
N_DIMS = 120

MODULES = {
    'time':     slice(0, 12),
    'freq':     slice(12, 27),
    'band':     slice(27, 91),
    'peaks':    slice(91, 103),
    'cepstral': slice(103, 120),
}


def _extract_time(w: np.ndarray) -> np.ndarray:
    absw = np.abs(w)
    mean = w.mean()
    std = w.std() + 1e-12
    peak = absw.max()
    rms = np.sqrt(np.mean(w ** 2)) + 1e-12
    return np.array([
        mean, std, rms, peak,
        peak - w.min(),                                    # PeakToPeak
        absw.mean(),                                       # MeanAbs
        np.mean(np.sqrt(absw)) ** 2,                       # SRA
        np.mean((w - mean) ** 3) / std ** 3,               # Skewness
        np.mean((w - mean) ** 4) / std ** 4,               # Kurtosis
        peak / rms,                                        # CrestFactor
        peak / (absw.mean() + 1e-12),                      # ImpulseFactor
        rms / (absw.mean() + 1e-12),                       # ShapeFactor
    ], dtype=np.float32)


def _extract_peaks(mag: np.ndarray, freqs: np.ndarray, n_peaks=6) -> np.ndarray:
    """主峰 6 个 x [freq, amp] = 12 维."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(mag)
    if len(idx) == 0:
        return np.zeros(2 * n_peaks, dtype=np.float32)
    top = idx[np.argsort(mag[idx])[::-1][:n_peaks]]
    top_sorted = top[np.argsort(freqs[top])]
    out = np.zeros(2 * n_peaks, dtype=np.float32)
    for i, k in enumerate(top_sorted):
        out[2 * i]     = freqs[k]
        out[2 * i + 1] = mag[k]
    return out


def _extract_freq(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """15 维频域统计."""
    total = mag.sum() + 1e-12
    p = mag / total
    centroid = (p * freqs).sum()
    bw = np.sqrt((p * (freqs - centroid) ** 2).sum())
    cs = np.cumsum(mag)
    rolloff85 = freqs[np.searchsorted(cs, 0.85 * cs[-1])] if cs[-1] > 0 else 0
    rolloff95 = freqs[np.searchsorted(cs, 0.95 * cs[-1])] if cs[-1] > 0 else 0
    x_ax = np.log(freqs + 1)
    y_ax = np.log(mag + 1e-12)
    slope = np.polyfit(x_ax, y_ax, 1)[0] if len(x_ax) > 1 else 0
    flatness = np.exp(np.mean(np.log(mag + 1e-12))) / (mag.mean() + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum()
    return np.array([
        centroid, bw, rolloff85, rolloff95, slope, flatness, entropy,
        mag.mean(), mag.std(), mag.max(), mag.min(),
        np.percentile(mag, 25), np.percentile(mag, 50),
        np.percentile(mag, 75), np.percentile(mag, 90),
    ], dtype=np.float32)


def _extract_band(mag: np.ndarray, freqs: np.ndarray, fs: int) -> np.ndarray:
    """32 带 x [能量占比, log 能量] = 64 维."""
    n_bands = 32
    max_f = fs / 2
    band_edges = np.linspace(0, max_f, n_bands + 1)
    total_e = (mag ** 2).sum() + 1e-12
    ratios = np.zeros(n_bands, dtype=np.float32)
    logenergies = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        m = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        be = (mag[m] ** 2).sum()
        ratios[i] = be / total_e
        logenergies[i] = np.log(be + 1e-12)
    return np.concatenate([ratios, logenergies])


def _extract_cepstral(w: np.ndarray) -> np.ndarray:
    """17 维倒谱系数."""
    from scipy.fft import rfft
    spec = np.abs(rfft(w)) + 1e-12
    ceps = np.fft.irfft(np.log(spec)).real
    return ceps[:17].astype(np.float32)


def extract_cfd(windows: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """(N, W) 振动信号 → (N, 120) CFD 特征矩阵."""
    from scipy.fft import rfft, rfftfreq
    if windows.ndim != 2:
        raise ValueError(f'Expected 2D, got {windows.shape}')
    n = windows.shape[0]
    out = np.zeros((n, N_DIMS), dtype=np.float32)
    freqs = rfftfreq(windows.shape[1], 1.0 / fs)
    for i in range(n):
        w = windows[i]
        mag = np.abs(rfft(w))
        out[i, MODULES['time']]     = _extract_time(w)
        out[i, MODULES['freq']]     = _extract_freq(mag, freqs)
        out[i, MODULES['band']]     = _extract_band(mag, freqs, fs)
        out[i, MODULES['peaks']]    = _extract_peaks(mag, freqs)
        out[i, MODULES['cepstral']] = _extract_cepstral(w)
    return out
