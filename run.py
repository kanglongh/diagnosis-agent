"""
diagnosis-agent · 跨域对比 (SCA vs 无SCA)

用法: python run.py [--dataset pu|cwru] [--sca on|off] [--model deepseek-v4-flash]
"""
from __future__ import annotations
import argparse, os, sys, numpy as np, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_env = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env):
    with open(_env, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from agent.core import run_agent
from agent.pipeline import evaluate
from agent.cfd import MODULES


def prefilter(X, X_tgt=None):
    dead = np.zeros(X.shape[1], dtype=bool)
    dead |= np.isnan(X).any(axis=0) | np.isinf(X).any(axis=0) | (np.ptp(X, axis=0) < 1e-10)
    if X_tgt is not None:
        dead |= np.isnan(X_tgt).any(axis=0) | np.isinf(X_tgt).any(axis=0) | (np.ptp(X_tgt, axis=0) < 1e-10)
    alive = np.where(~dead)[0]
    return alive


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='pu', choices=['pu', 'cwru'])
    p.add_argument('--data', default='data/pu_tasks.npz')
    p.add_argument('--model', default='deepseek-v4-flash')
    p.add_argument('--sca', default='on', choices=['on', 'off'])
    args = p.parse_args()

    if args.dataset == 'cwru':
        args.data = 'data/cwru_tasks.npz'

    # 日志
    os.makedirs('logs', exist_ok=True)
    ts = datetime.datetime.now().strftime('%m%d_%H%M')
    log_file = f'logs/{args.dataset}_sca{args.sca}_{ts}.log'
    fh = open(log_file, 'w', encoding='utf-8')
    import io
    _real_stdout = sys.stdout
    class Tee(io.StringIO):
        def write(self, s):
            _real_stdout.write(s); fh.write(s); fh.flush()
            return len(s) if hasattr(s, '__len__') else 0
    sys.stdout = Tee()

    print(f'{"="*60}\ndiagnosis-agent · {args.dataset.upper()} 跨域')
    print(f'SCA: {args.sca.upper()} | 模型: {args.model} | 日志: {log_file}')
    print(f'{"="*60}')

    z = np.load(args.data, allow_pickle=True)

    for tn in z['task_names']:
        Xs = z[f'{tn}_X_src']; ys = z[f'{tn}_y_src']
        Xt = z[f'{tn}_X_tgt']; yt = z[f'{tn}_y_tgt']
        alive = prefilter(Xs, Xt)
        Xs, Xt = Xs[:, alive], Xt[:, alive]
        n, nc = Xs.shape[1], len(np.unique(ys))
        print(f'\n{tn}: {n}维 {nc}类 src={Xs.shape[0]}窗 tgt={Xt.shape[0]}窗')

        tsk = {'X_src': Xs, 'y_src': ys, 'X_tgt': Xt, 'y_tgt': yt, 'name': tn, 'mode': 'cross'}
        full_bl = max(evaluate(tsk, 'cohens_d', n)['results'].values())
        bot40 = max(evaluate(tsk, 'cohens_d', 40)['results'].values())
        print(f'full{n}d: {full_bl:.4f}  bot40: {bot40:.4f}')

        tsk['sca_enabled'] = (args.sca == 'on')
        res = run_agent(tsk, model=args.model, verbose=True)

        bp = res['best_pipeline']
        if bp:
            print(f'\n  Agent: {bp["selector"]}+{bp["actual_n_dims"]}d+{bp.get("da","none")}x{bp["classifier"]}={bp["accuracy"]:.4f}')
            print(f'  vs full: Δ={bp["accuracy"]-full_bl:+.4f}  vs bot40: Δ={bp["accuracy"]-bot40:+.4f}')
            print(f'  尝试次数: {res["n_calls"]}')

    print(f'\n日志保存: {log_file}')
    fh.close()


if __name__ == '__main__':
    main()
