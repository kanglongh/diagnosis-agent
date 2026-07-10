"""
域适应方法 (4 种)

    none         — 不施加域适应
    mean_align   — 目标域均值对齐到源域
    zscore_align — 逐维 Z-score 对齐到源域分布
    coral        — CORAL 二阶协方差对齐
"""
import numpy as np
from scipy.linalg import sqrtm

DA_METHODS = {
    'none':         '不施加域适应',
    'mean_align':   '目标域均值对齐到源域 (一阶矩对齐)',
    'zscore_align': '逐维 Z-score 归一化到源域分布 (一阶+二阶矩对齐)',
    'coral':        'CORAL 协方差对齐 (二阶统计量对齐)',
}


def apply_da(X_src: np.ndarray, X_tgt: np.ndarray, method: str) -> tuple:
    """对已标准化的 (X_src, X_tgt) 施加域适应.

    Returns (X_src', X_tgt').
    """
    if method == 'none':
        return X_src, X_tgt

    elif method == 'mean_align':
        return X_src, X_tgt - X_tgt.mean(axis=0) + X_src.mean(axis=0)

    elif method == 'zscore_align':
        src_mean, src_std = X_src.mean(axis=0), X_src.std(axis=0) + 1e-8
        tgt_mean, tgt_std = X_tgt.mean(axis=0), X_tgt.std(axis=0) + 1e-8
        Xt_aligned = (X_tgt - tgt_mean) / tgt_std * src_std + src_mean
        return X_src, Xt_aligned

    elif method == 'coral':
        Cs = np.cov(X_src, rowvar=False) + np.eye(X_src.shape[1]) * 1e-3
        Ct = np.cov(X_tgt, rowvar=False) + np.eye(X_tgt.shape[1]) * 1e-3
        Ct_inv = np.linalg.inv(Ct)
        Cs_sqrt = np.real(sqrtm(Cs))
        Ct_inv_sqrt = np.real(sqrtm(Ct_inv))
        Xt_coral = (X_tgt - X_tgt.mean(axis=0)) @ Ct_inv_sqrt @ Cs_sqrt + X_src.mean(axis=0)
        return X_src, Xt_coral

    else:
        raise ValueError(f'未知 DA 方法: {method}. 可选: {list(DA_METHODS.keys())}')
