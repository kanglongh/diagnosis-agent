"""
特征选择算法 (6 种, 原理各不相同)

策略:
    cohens_d     — 保留跨域最稳定的维 (d 值最低 n 维)
    fisher       — 保留类间可分性最强的维 (ANOVA F 值最高 n 维)
    mutual_info  — 保留与标签非线性依赖最强的维 (互信息最高 n 维)
    l1_lasso     — L1 正则化自动稀疏选维 (保留非零权重)
    rfe          — 递归特征消除 (SVM 做基分类器, 贪心剔除)
    random       — 随机选维 (基线, 必须显著优于它)
"""
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

SELECTORS = {
    'cohens_d':    '按跨域漂移量 Cohen\'s d 取最低 n 维 (保留跨域稳定特征)',
    'fisher':      '按 Fisher score (ANOVA F) 取最高 n 维 (保留类间可分的特征)',
    'mutual_info': '按互信息取最高 n 维 (保留与标签非线性依赖最强的特征)',
    'weighted':    '★ 综合权重: cohens_d + fisher + mutual_info + 方差, 按总分取最高 n 维',
    'l1_lasso':    'L1 正则化 (Lasso) 自动选维, n 控制稀疏度 (越小越稀疏)',
    'rfe':         '递归特征消除 (RFE), 用 SVM 做基分类器, 贪心剔除',
    'random':      '随机选 n 维 (基线, 用于验证其他方法不是碰运气)',
}


def compute_weights(X_src, y_src, X_tgt, save_detailed=False):
    """计算每维的任务目标权重 (0-1, 越高越重要).

    权重 = normalized(cohens_d 稳定性) × 0.3
         + normalized(fisher 可分性)    × 0.3
         + normalized(mutual_info)      × 0.2
         + normalized(方差信息量)       × 0.2

    返回 (weights, detail_dict) 或 weights.
    """
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.preprocessing import MinMaxScaler

    n_dims = X_src.shape[1]

    # 1. Cohen's d 稳定性 (d越低→权重越高)
    d = np.abs(X_src.mean(0) - X_tgt.mean(0)) / \
        (np.sqrt((X_src.var(0) + X_tgt.var(0)) / 2) + 1e-8)
    cohen_w = 1.0 / (1.0 + d)  # d=0 → 1.0, d=∞ → 0

    # 2. Fisher score (越高越好)
    F, _ = f_classif(X_src, y_src)
    F = np.nan_to_num(F, nan=0.0)
    fisher_w = MinMaxScaler().fit_transform(F.reshape(-1, 1)).ravel()

    # 3. Mutual info (越高越好)
    mi = mutual_info_classif(X_src, y_src, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    mi_w = MinMaxScaler().fit_transform(mi.reshape(-1, 1)).ravel()

    # 4. 方差信息量 (越高越好)
    var = X_src.var(0)
    var[var < 1e-8] = 0
    var_w = MinMaxScaler().fit_transform(var.reshape(-1, 1)).ravel()

    # 综合
    weights = 0.3 * cohen_w + 0.3 * fisher_w + 0.2 * mi_w + 0.2 * var_w

    if save_detailed:
        detail = {
            'cohens_d': d.tolist(), 'cohens_weight': cohen_w.tolist(),
            'fisher': F.tolist(), 'fisher_weight': fisher_w.tolist(),
            'mutual_info': mi.tolist(), 'mi_weight': mi_w.tolist(),
            'variance': var.tolist(), 'var_weight': var_w.tolist(),
        }
        return weights, detail
    return weights


def select_dims(X_src, y_src, X_tgt, selector: str, n_dims: int) -> np.ndarray:
    """返回选中的维度索引 (绝对索引, 0 ~ n_total-1).

    Parameters
    ----------
    X_src : (n_src, n_total) 源域特征
    y_src : (n_src,) 源域标签
    X_tgt : (n_tgt, n_total) 目标域特征
    selector : str  特征选择算法名
    n_dims : int  目标维数

    Returns
    -------
    dims : ndarray 选中的维度索引
    """
    n_total = X_src.shape[1]
    n_dims = min(max(n_dims, 1), n_total)

    if selector == 'cohens_d':
        mu_s = X_src.mean(axis=0)
        mu_t = X_tgt.mean(axis=0)
        var_s = X_src.var(axis=0)
        var_t = X_tgt.var(axis=0)
        d = np.abs(mu_s - mu_t) / (np.sqrt((var_s + var_t) / 2) + 1e-8)
        return np.argsort(d)[:n_dims]

    elif selector == 'fisher':
        F, _ = f_classif(X_src, y_src)
        return np.argsort(F)[::-1][:n_dims]

    elif selector == 'mutual_info':
        mi = mutual_info_classif(X_src, y_src, random_state=42)
        return np.argsort(mi)[::-1][:n_dims]

    elif selector == 'l1_lasso':
        C = 1.0 / max(n_dims, 1)
        lr = LogisticRegression(penalty='l1', solver='saga', C=C,
                                max_iter=3000, random_state=42)
        lr.fit(X_src, y_src)
        scores = np.abs(lr.coef_).mean(axis=0)
        nonzero = np.where(scores > 1e-6)[0]
        if len(nonzero) == 0:
            return np.argsort(scores)[::-1][:n_dims]
        return nonzero

    elif selector == 'rfe':
        n_features_to_select = max(n_dims, 2)
        estimator = SVC(kernel='linear', random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
        rfe.fit(X_src, y_src)
        return np.argsort(rfe.ranking_)[:n_dims]

    elif selector == 'weighted':
        weights = compute_weights(X_src, y_src, X_tgt)
        return np.argsort(weights)[::-1][:n_dims]

    elif selector == 'random':
        rng = np.random.default_rng(42)
        return rng.choice(n_total, size=n_dims, replace=False)

    else:
        raise ValueError(f'未知 selector: {selector}. 可选: {list(SELECTORS.keys())}')
