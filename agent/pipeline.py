"""
诊断管道 · analyze / evaluate / ablation / gap

analyze_task   → 权重分布 + 模块级诊断 + 文献对标
evaluate        → 特征选择 + 分类器评估 (5-fold CV)
module_ablation → 逐个关闭模块, 看每模块的边际贡献
compare_gap     → CFD vs 文献特征池的缺口分析
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from .selectors import select_dims, SELECTORS, compute_weights
from .da_methods import apply_da
from .tools import CLASSIFIERS, CLF_NAMES
from .cfd import MODULES


def analyze_task(task) -> dict:
    Xs, ys = task['X_src'], task['y_src']
    name = task.get('name', 'unknown')

    # 1. 每维 Fisher + MI 权重
    F, _ = f_classif(Xs, ys)
    F = np.nan_to_num(F, nan=0.0)
    fisher_w = MinMaxScaler().fit_transform(F.reshape(-1, 1)).ravel()
    mi = mutual_info_classif(Xs, ys, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    mi_w = MinMaxScaler().fit_transform(mi.reshape(-1, 1)).ravel()
    var = Xs.var(0)
    var[var < 1e-8] = 0
    var_w = MinMaxScaler().fit_transform(var.reshape(-1, 1)).ravel()
    weights = 0.4 * fisher_w + 0.3 * mi_w + 0.3 * var_w

    ranked = np.argsort(weights)[::-1]

    # 2. 模块级诊断
    module_report = {}
    for mod, sl in MODULES.items():
        seg_w = weights[sl]
        module_report[mod] = {
            'n_dims': int(seg_w.shape[0]),
            'mean_weight': round(float(seg_w.mean()), 3),
            'top3_weight': round(float(np.sort(seg_w)[-3:].mean()), 3) if len(seg_w) >= 3 else 0,
            'zero_weight': int((seg_w < 0.01).sum()),
        }

    # 3. 识别冗余模块 (均值权重 < 0.1 且零权维 > 50%)
    redundant = [mod for mod, r in module_report.items()
                 if r['mean_weight'] < 0.1 and r['zero_weight'] > r['n_dims'] * 0.5]

    # 4. 高权重 + 高方差但低互信息 → 可能是噪声放大
    noisy = [{'dim': int(i), 'fisher': round(float(fisher_w[i]), 3),
              'mi': round(float(mi_w[i]), 3), 'var': round(float(var_w[i]), 3)}
             for i in np.where((fisher_w > 0.6) & (mi_w < 0.2) & (var_w > 0.5))[0][:5]]

    return {
        'task': name,
        'n_samples': int(Xs.shape[0]), 'n_dims': int(Xs.shape[1]),
        'n_classes': int(len(np.unique(ys))),
        'top10_dims': [{'dim': int(i), 'weight': round(float(weights[i]), 3)} for i in ranked[:10]],
        'modules': module_report,
        'redundant_modules': redundant,
        'noisy_dims': noisy,
    }


def evaluate(task, selector, n_dims, da='none', classifiers=None, base_dims=None) -> dict:
    """特征选择 + 评估 (同域 5-fold CV, 跨域 train/test)."""
    Xs, ys = task['X_src'], task['y_src']
    is_cross = task.get('mode') == 'cross'
    Xt = task.get('X_tgt', Xs) if is_cross else Xs
    yt = task.get('y_tgt', ys) if is_cross else ys
    ALL = np.arange(Xs.shape[1])

    if base_dims is not None:
        base_dims = np.array(base_dims, dtype=int)
        X_sub = Xs[:, base_dims]
    else:
        X_sub = Xs[:, ALL]

    sc = StandardScaler().fit(X_sub)
    Xs_s = sc.transform(X_sub)
    sub_dims = select_dims(Xs_s, ys, Xs_s, selector, n_dims)
    actual_dims = (base_dims[sub_dims] if base_dims is not None else sub_dims).tolist()
    n_actual = int(len(sub_dims))

    if n_actual == 0:
        c_list = classifiers if classifiers else CLF_NAMES
        return {'selector': selector, 'n_dims': n_dims, 'actual_n_dims': 0,
                'da': da, 'results': {c: 0.0 for c in c_list}, 'surviving_dims': []}

    clf_list = classifiers if classifiers else CLF_NAMES
    results = {}
    if is_cross:
        # 跨域: StandardScaler on src, DA on tgt, train on src, test on tgt
        Xt_sub = Xt[:, base_dims] if base_dims is not None else Xt[:, ALL]
        Xt_s = sc.transform(Xt_sub)
        Xt_sd = apply_da(Xs_s, Xt_s, da)[1] if da != 'none' else Xt_s
        from sklearn.metrics import accuracy_score
        for clf_name in clf_list:
            clf = CLASSIFIERS[clf_name]()
            clf.fit(Xs_s[:, sub_dims], ys)
            results[clf_name] = round(float(accuracy_score(yt, clf.predict(Xt_sd[:, sub_dims]))), 4)
    else:
        for clf_name in clf_list:
            clf = CLASSIFIERS[clf_name]()
            scores = cross_val_score(clf, Xs_s[:, sub_dims], ys,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=42))
            results[clf_name] = round(float(scores.mean()), 4)

    return {'selector': selector, 'n_dims': n_dims, 'actual_n_dims': n_actual,
            'da': da, 'results': results, 'surviving_dims': actual_dims}


def module_ablation(task, module_names) -> dict:
    """关闭指定模块(多个), 看剩余维度的诊断精度.
    同域: 5-fold CV. 跨域: train on source, test on target.
    module_names: 如 'band' 或 ['band', 'cepstral']
    """
    Xs, ys = task['X_src'], task['y_src']
    is_cross = task.get('mode') == 'cross'
    Xt = task.get('X_tgt', Xs) if is_cross else Xs
    yt = task.get('y_tgt', ys) if is_cross else ys
    ALL = np.arange(Xs.shape[1])
    if isinstance(module_names, str):
        module_names = [module_names]

    removed_ranges = []
    for mn in module_names:
        if mn not in MODULES:
            return {'error': f'未知模块: {mn}. 可选: {list(MODULES.keys())}'}
        sl = MODULES[mn]
        removed_ranges.append((sl.start, sl.stop))

    keep = np.array([i for i in ALL
                     if not any(start <= i < stop for start, stop in removed_ranges)])

    sc = StandardScaler().fit(Xs[:, keep]); Xs_s = sc.transform(Xs[:, keep])

    results = {}
    if is_cross:
        Xt_s = sc.transform(Xt[:, keep])
        from sklearn.metrics import accuracy_score
        for clf_name in ['RF', 'KNN-5', 'SVM-RBF']:
            clf = CLASSIFIERS[clf_name](); clf.fit(Xs_s, ys)
            results[clf_name] = round(float(accuracy_score(yt, clf.predict(Xt_s))), 4)
    else:
        for clf_name in ['RF', 'KNN-5', 'SVM-RBF']:
            clf = CLASSIFIERS[clf_name]()
            scores = cross_val_score(clf, Xs_s, ys,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=42))
            results[clf_name] = round(float(scores.mean()), 4)

    mode_label = '跨域' if is_cross else '同域'
    return {
        'modules_removed': module_names,
        'dims_removed': Xs.shape[1] - int(len(keep)),
        'dims_kept': int(len(keep)),
        'mode': mode_label,
        'results': results,
        'surviving_dims': keep.tolist(),  # ← Agent 可直接传 base_dims
        'interpretation': '精度下降>0.03 → 关键模块. 不变或微涨 → 冗余/噪声. surviving_dims 可直接用作 test_subset 的 base_dims.'
    }

