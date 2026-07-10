"""PU 跨域任务 (Y6 vibration_1 正确通道)"""
import numpy as np, scipy.io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.cfd import extract_cfd
from agent.io_utils import make_windows

FS = 64000
BEARINGS = ['K001','K002','KA01','KA05','KI01','KI05']
LABEL = {'K001':0,'K002':0,'KA01':1,'KA05':1,'KI01':2,'KI05':2}

def load_cond(cond):
    Xl, yl = [], []
    for lbl, bk in enumerate(BEARINGS):
        for i in range(1,21):
            fp = f'data/PU/{bk}/{cond}_{bk}_{i}.mat'
            if not os.path.exists(fp): continue
            m = scipy.io.loadmat(fp)
            key = [k for k in m.keys() if not k.startswith('__')][0]
            sig = np.array(m[key][0,0]['Y'][0,6]['Data']).flatten()
            w = make_windows(sig, 10240, 5120)
            Xl.append(extract_cfd(w, fs=FS)); yl.extend([LABEL[bk]]*len(w))
    return np.vstack(Xl).astype(np.float32), np.array(yl)

save = {'task_names': []}
for src_cond, tgt_cond in [('N09_M07_F10', 'N15_M07_F10')]:
    name = f'PU_N09_N15'
    print(f'{name}')
    Xs, ys = load_cond(src_cond); Xt, yt = load_cond(tgt_cond)
    print(f'  src={Xs.shape[0]}窗, tgt={Xt.shape[0]}窗')
    save[f'{name}_X_src'] = Xs; save[f'{name}_y_src'] = ys
    save[f'{name}_X_tgt'] = Xt; save[f'{name}_y_tgt'] = yt
    save['task_names'].append(name)

np.savez_compressed('data/pu_tasks.npz', **save)
print(f'保存: data/pu_tasks.npz ({os.path.getsize("data/pu_tasks.npz")/1e6:.1f} MB)')
