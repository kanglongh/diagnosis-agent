"""SCA 守卫 · 每次最多一条追问"""
from __future__ import annotations


class SCAGuard:
    _ALL_MODULES = {'time', 'freq', 'band', 'peaks', 'cepstral'}

    def __init__(self):
        self._modules_tested = set()
        self._module_results = {}
        self._best_acc = 0.0
        self._stagnation = 0
        self._n_tests = 0
        self._ablation_done_injected = False
        self._selectors_tried = set()
        self._n_dims_ranges = set()
        self._base_dims_done = False
        self._test_log = []

    def state_dict(self): return {
        'modules_tested': sorted(self._modules_tested), 'module_results': self._module_results,
        'best_acc': self._best_acc, 'stagnation': self._stagnation, 'n_tests': self._n_tests,
        'selectors_tried': sorted(self._selectors_tried), 'n_dims_ranges': sorted(self._n_dims_ranges),
        'base_dims_done': self._base_dims_done, 'test_log': self._test_log,
    }

    def load_state(self, d: dict):
        if not d: return
        self._modules_tested = set(d.get('modules_tested', []))
        self._module_results = d.get('module_results', {})
        self._best_acc = d.get('best_acc', 0.0)
        self._stagnation = d.get('stagnation', 0)
        self._n_tests = d.get('n_tests', 0)
        self._selectors_tried = set(d.get('selectors_tried', []))
        self._n_dims_ranges = set(d.get('n_dims_ranges', []))
        self._base_dims_done = d.get('base_dims_done', False)
        self._test_log = d.get('test_log', [])
        if self._modules_tested: self._ablation_done_injected = len(self._modules_tested) >= 5

    def pre_test(self, selector: str, n_dims: int, hypothesis: str) -> list:
        """每次最多返回1条追问, 按优先级."""
        # 消融不足 (最高优先)
        missing = sorted(self._ALL_MODULES - self._modules_tested)
        if missing and len(self._modules_tested) < 5:
            return [{'level': 'challenge', 'tag': '缺消融',
                     'message': '未测:' + ','.join(missing) + '. 先消融?'}]
        # 重复
        prev = [t for t in self._test_log if t['selector'] == selector and abs(t['n_dims'] - n_dims) <= 5]
        if prev:
            return [{'level': 'challenge', 'tag': '重复',
                     'message': f'{selector}+{prev[-1]["n_dims"]}d={prev[-1]["accuracy"]:.3f}已试. 区别?'}]
        # 停滞
        if self._stagnation >= 3:
            return [{'level': 'challenge', 'tag': '天花板?',
                     'message': f'{self._stagnation}轮未破{self._best_acc:.4f}. 极限?'}]
        # 路径依赖
        untried = [s for s in ['fisher', 'weighted', 'l1_lasso'] if s not in self._selectors_tried]
        if untried and self._n_tests >= 2:
            return [{'level': 'challenge', 'tag': '路径?',
                     'message': '未试:' + ','.join(untried) + '. 凭什么更差?'}]
        # 区间盲区
        if len(self._n_dims_ranges) < 2 and self._n_tests >= 3:
            return [{'level': 'challenge', 'tag': '区间?',
                     'message': '只在low搜. 如果最优是40维?'}]
        return []

    def post_test(self, hypothesis: str, accuracy: float, old_best: float = 0.0) -> list:
        if not hypothesis or len(hypothesis) < 10: return []
        was_new = accuracy > old_best + 0.001
        if was_new:
            return [{'level': 'encourage', 'tag': '新高',
                     'message': f'{accuracy:.4f} 假设成立. 继续.'}]
        elif any(kw in hypothesis for kw in ['消融', '关', '噪音', '涨', '反涨']):
            return [{'level': 'notice', 'tag': '推论?',
                     'message': '前提对, 推论没兑现. 修正推论, 别否定前提.'}]
        else:
            return [{'level': 'reflect', 'tag': '假设?',
                     'message': f'预测未兑现({accuracy:.4f}). 哪里错了?'}]

    def track_test(self, selector, n_dims, accuracy, hypothesis, base_dims=False, classifier='?'):
        self._n_tests += 1; self._selectors_tried.add(selector)
        if n_dims <= 25: self._n_dims_ranges.add('low')
        elif n_dims <= 50: self._n_dims_ranges.add('mid')
        else: self._n_dims_ranges.add('high')
        if base_dims: self._base_dims_done = True
        self._test_log.append({'selector': selector, 'n_dims': n_dims,
                               'accuracy': accuracy, 'classifier': classifier,
                               'hypothesis': hypothesis[:80]})
        if accuracy > self._best_acc + 0.001: self._best_acc, self._stagnation = accuracy, 0
        else: self._stagnation += 1

    def observe_ablation(self, module_key, acc_after):
        for m in module_key.split(','):
            if m.strip() in self._ALL_MODULES: self._modules_tested.add(m.strip())
        self._module_results[module_key] = acc_after

    def ablation_summary(self) -> str:
        if len(self._modules_tested) < 5 or self._ablation_done_injected: return ''
        self._ablation_done_injected = True
        lines = [f'[消融] 完成. 最优={self._best_acc:.4f}']
        over = []
        for mod, acc in sorted(self._module_results.items()):
            if acc > self._best_acc + 0.001: over.append(f'{mod}={acc:.4f}')
        if over: lines.append('超最优: ' + '; '.join(over) + ' → 追这个方向.')
        return '\n'.join(lines)

    def auto_diagnose(self) -> str:
        best_abl = max(self._module_results.values()) if self._module_results else 0.0
        if best_abl > self._best_acc + 0.001:
            m = [k for k, v in self._module_results.items() if v == best_abl][0]
            return f'[系统] 消融{m}={best_abl:.4f} > 最优{self._best_acc:.4f}. 追这个.'
        gaps = self._coverage_gaps()
        if gaps: return f'[系统] 最优{self._best_acc:.4f}. 缺口: {"; ".join(gaps)}'
        return f'[系统] 最优{self._best_acc:.4f}. 覆盖率完整.'

    def _coverage_gaps(self) -> list:
        gaps = []
        if sorted(self._ALL_MODULES - self._modules_tested): gaps.append('消融')
        untried = [s for s in ['fisher', 'weighted', 'l1_lasso'] if s not in self._selectors_tried]
        if untried: gaps.append('selector:' + ','.join(untried))
        if len(self._n_dims_ranges) < 2: gaps.append('n_dims区间')
        return gaps

    @property
    def modules_covered(self): return len(self._modules_tested)
    @property
    def best_accuracy(self): return self._best_acc
