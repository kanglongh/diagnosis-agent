"""
CWRU 跨域任务生成 (从 vibrolab 粘贴)

用法: python prepare_cwru.py [--cwru-dir ../vibrolab/data] [--out data/cwru_tasks.npz]
"""
from __future__ import annotations
import argparse, os, sys, glob, numpy as np, scipy.io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.cfd import extract_cfd
from agent.io_utils import make_windows

FS = 12000
WINDOW = 2048

LABEL_MAP = {'N': 0, 'B007': 1, 'B014': 2, 'B021': 3,
             'IR007': 4, 'IR014': 5, 'IR021': 6,
             'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9}


def load_hp(data_root, hp):
    """加载指定 HP 的全部 .mat, 返回 (features, labels)."""
    # DE 端 12k + Normal 基线
    X_all, y_all = [], []
    files = (glob.glob(os.path.join(data_root, '12k_DE', f'*_{hp}.mat'))
           + glob.glob(os.path.join(data_root, 'Normal', f'Normal_{hp}.mat')))

    for fp in files:
        fname = os.path.basename(fp).split('.')[0]
        # 解析标签: B007_3 → B007, Normal_3 → N
        lbl = fname.rsplit('_', 1)[0]
        if lbl == 'Normal': lbl = 'N'
        if lbl not in LABEL_MAP:
            continue
        m = scipy.io.loadmat(fp)
        for key in m:
            if key.endswith('_DE_time'):
                sig = np.asarray(m[key]).squeeze()
                windows = make_windows(sig, WINDOW, WINDOW)
                if len(windows) == 0:
                    continue
                feats = extract_cfd(windows, fs=FS)
                X_all.append(feats)
                y_all.extend([LABEL_MAP[lbl]] * len(feats))
                break
    return np.vstack(X_all).astype(np.float32), np.array(y_all)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cwru-dir', default=os.path.join(os.path.dirname(__file__), '..', 'vibrolab', 'data'))
    p.add_argument('--out', default='data/cwru_tasks.npz')
    args = p.parse_args()

    save = {'task_names': []}
    for src_hp, tgt_hp in [(3, 0)]:
        name = f'CWRU_3HP_0HP'
        print(f'{name}: {src_hp}HP→{tgt_hp}HP (最难方向)')
        Xs, ys = load_hp(args.cwru_dir, src_hp)
        Xt, yt = load_hp(args.cwru_dir, tgt_hp)
        print(f'  src={Xs.shape[0]}窗, tgt={Xt.shape[0]}窗, {len(np.unique(ys))}类')
        save[f'{name}_X_src'] = Xs
        save[f'{name}_y_src'] = ys
        save[f'{name}_X_tgt'] = Xt
        save[f'{name}_y_tgt'] = yt
        save['task_names'].append(name)

    np.savez_compressed(args.out, **save)
    print(f'保存: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
