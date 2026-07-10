"""
诊断 Agent · SCA 进度感知反馈 (不拦截, 打标签 + 强制反思)
"""
from __future__ import annotations
import json, os, numpy as np
from .pipeline import analyze_task, evaluate, module_ablation
from .tools import CLF_NAMES
from .sca import SCAGuard


def run_agent(task: dict, model='qwen-plus', verbose=True, max_rounds=20) -> dict:
    task_name = task.get('name', 'unknown')
    n_dims_max = task['X_src'].shape[1]
    is_cross = task.get('mode') == 'cross'
    selectors = ['weighted', 'fisher', 'mutual_info', 'l1_lasso', 'cohens_d', 'random'] if is_cross else \
                ['weighted', 'fisher', 'mutual_info', 'l1_lasso', 'random']

    api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('LLM_BASE_URL', 'https://api.deepseek.com/v1')
    if not api_key:
        raise RuntimeError('set API key')
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    sca_on = task.get('sca_enabled', True)
    guard = SCAGuard()

    skip_note = ''
    flow_text = f"""## 实验流程
1.{skip_note} 观察: analyze_task 看权重分布
2.{skip_note} 假设: module_ablation → 哪些模块关掉后精度变化? 形成假设
3. 验证: test_subset 验证你的假设 —— 不是盲扫 selector
4. 修正: 假设被证实还是推翻? 修正, 再验证
5. 收敛: 连续修正后趋于稳定 → 判断是否触顶"""

    task_type = '跨域' if is_cross else '同域'
    SYSTEM_PROMPT = f"""你是故障诊断科学家. 在 {n_dims_max} 维 CFD 特征上进行{task_type}诊断.

## 工作方法: 假设→验证→修正
你不是在调参. 你是在做科学实验.
每一轮: 观察数据 → 形成假设 → 设计实验验证 → 看结果 → 修正假设.
假设必须明确: "我认为X, 因为Y. 如果X成立, 实验应该看到Z."

## 工具
- analyze_task → 权重分布 + 模块报告
- module_ablation(module_names) → 关模块看精度, 返回 surviving_dims
- test_subset(selector, n_dims, classifiers?, base_dims?, hypothesis) → 选维 + 验证假设

{flow_text}

## 系统反馈标签
每次 test_subset 后系统会打标签:
  [notice]   — 提醒: 消融不足/缺假设
  [challenge] — 追问: 停滞, 是否触顶
  [reflect]  — 对比: 假设 vs 实际, 为什么对/错
  [encourage] — 新高, 方向对了
不要忽略这些标签. 每次反思都是你的学习机会.

分类器: {CLF_NAMES}"""

    # 第一轮用户消息
    user_msg = f'诊断{task_name}. 从 analyze_task 开始.'
    if False:  # prior_state logic removed
        done_mods = ', '.join(sorted(guard._modules_tested))
        gaps = guard._coverage_gaps()
        gap_str = '; '.join(gaps) if gaps else '全部覆盖, 可触发天花板诊断'

        # 消融结果
        ablation_lines = ['消融结果 (已完成, 不要重做):']
        for mod, acc in sorted(guard._module_results.items()):
            ablation_lines.append(f'  关{mod}: RF={acc:.4f}')
        ablation_lines.append(f'  最优 so far: {guard.best_accuracy:.4f}')

        # 探索轨迹
        test_lines = [f'已探索轨迹 ({guard._n_tests} 次):']
        for i, t in enumerate(guard._test_log, 1):
            test_lines.append(f'  [{i}] {t["selector"]}+{t["n_dims"]}d = {t["accuracy"]:.4f} | {t["hypothesis"][:60]}')
        test_lines.append(f'缺口: {gap_str}')

        user_msg = '历史记录如下. 不要重复已试过的组合.\n\n' + '\n'.join(ablation_lines + test_lines)
    else:
        user_msg += ' 从 analyze_task 开始.'

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_msg}
    ]

    TOOLS = [
        {'type': 'function', 'function': {
            'name': 'analyze_task', 'description': '权重分布 + 模块诊断.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []}
        }},
        {'type': 'function', 'function': {
            'name': 'module_ablation', 'description': '关模块看精度. 一次测单个或多个: [\"band\"] 或 [\"band\",\"cepstral\"] 测耦合.',
            'parameters': {'type': 'object', 'properties': {
                'module_names': {
                    'type': 'array',
                    'items': {'type': 'string', 'enum': ['time', 'freq', 'band', 'peaks', 'cepstral']},
                    'description': '要关闭的模块列表. 一次可以关多个, 测耦合效应.'
                },
            }, 'required': ['module_names']}
        }},
        {'type': 'function', 'function': {
            'name': 'test_subset', 'description': f'选维. clf={CLF_NAMES}. hypothesis建议写.',
            'parameters': {'type': 'object', 'properties': {
                'selector': {'type': 'string', 'enum': selectors},
                'n_dims': {'type': 'integer', 'minimum': 5, 'maximum': n_dims_max},
                'da': {'type': 'string', 'enum': ['none', 'mean_align', 'zscore_align', 'coral']} if is_cross else {'type': 'string', 'enum': ['none']},
                'classifiers': {'type': 'array', 'items': {'type': 'string'}},
                'base_dims': {'type': 'array', 'items': {'type': 'integer'}},
                'hypothesis': {'type': 'string', 'description': '预计结果+理由.'},
            }, 'required': ['selector', 'n_dims']}
        }}
    ]

    tool_log = []
    for round_i in range(max_rounds):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              tools=TOOLS, tool_choice='auto', temperature=0.2,
                                              max_tokens=4096, extra_body={})
        msg = resp.choices[0].message

        if msg.content and not msg.tool_calls:
            if sca_on and guard._stagnation >= 3:
                ok = guard.submit_diagnosis(msg.content)
                if verbose:
                    mark = '✓' if ok else '?'
                    print(f'\n  💭 [诊断 {mark}] {msg.content}')
            elif msg.content and verbose:
                print(f'\n  💭 {msg.content}')
            if round_i < 2:
                messages.append({'role': 'user', 'content': '请调用工具.'})
                continue
            break

        if msg.content and verbose:
            print(f'\n  💭 {msg.content}')

        if not msg.tool_calls:
            # 消融发现超优结果但 Agent 没追 → 硬拦, 必须回答
            best_abl = max(guard._module_results.values()) if guard._module_results else 0.0
            has_unpursued = best_abl > guard.best_accuracy + 0.001
            if has_unpursued:
                best_mod = [m for m, a in guard._module_results.items() if a == best_abl][0]
                q = (f'消融发现 关{best_mod}={best_abl:.4f} > 你当前最优{guard.best_accuracy:.4f}. '
                     f'先回答: 为什么你没追这个方向? 关{best_mod}后的精度比你的最优还好, '
                     f'这意味着什么? 回答完才能继续调工具.')
                messages.append({'role': 'user', 'content': q})
                continue
            if sca_on and guard._stagnation >= 2:
                messages.append({'role': 'user', 'content': 'SCA追问未回应. 请调用工具继续探索, 或在 💭 中诊断天花板.'})
                continue
            if round_i < 2:
                messages.append({'role': 'user', 'content': '请调用工具.'})
                continue
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                continue
            fn = tc.function.name
            if fn not in ('analyze_task', 'module_ablation', 'test_subset'):
                continue

            # 执行
            if fn == 'analyze_task':
                r = analyze_task(task)
            elif fn == 'module_ablation':
                mn = args['module_names']
                mod_key = ','.join(sorted(mn)) if isinstance(mn, list) else str(mn)
                if sca_on and mod_key in guard._module_results:
                    prev = guard._module_results[mod_key]
                    if verbose:
                        print(f'\n  🏷 [challenge] 重复消融: {mod_key} 已测过 (RF={prev:.4f})')
                r = module_ablation(task, mn)
                guard.observe_ablation(mod_key, r['results'].get('RF', 0))
                if verbose:
                    mode_tag = r.get('mode', '')
                    print(f'\n  🔬[{mode_tag}] 关 {r["modules_removed"]}: -{r["dims_removed"]}维, '
                          f'RF={r["results"]["RF"]:.4f}')
            elif fn == 'test_subset':
                hypothesis = args.pop('hypothesis', '')

                if sca_on:
                    for tag in guard.pre_test(args.get('selector', '?'), args.get('n_dims', 0), hypothesis):
                        if verbose: print(f'  [{tag["level"]}] {tag["tag"]}: {tag["message"]}')
                        messages.append({'role': 'user', 'content': f'[{tag["level"].upper()}] {tag["tag"]}: {tag["message"]}'})

                if verbose:
                    n = len([t for t in tool_log if t['name'] == 'test_subset']) + 1
                    clfs = args.get('classifiers', [])
                    bd = ''
                    if args.get('base_dims'):
                        n_base = len(args['base_dims'])
                        # 推断base来自哪个消融
                        tag = ''
                        if n_base == 108: tag = ' (去time)'
                        elif n_base == 105: tag = ' (去freq)'
                        elif n_base == 56: tag = ' (去band)'
                        elif n_base == 108: tag = ' (去peaks)'
                        elif n_base == 103: tag = ' (去cepstral)'
                        bd = f', base={n_base}d{tag}'
                    print(f'  [{n}] {args["selector"]:12s} n={args["n_dims"]:3d} '
                          f'[{",".join(clfs) if clfs else "RF"}]{bd}', end=' ')

                r = evaluate(task, **{k: v for k, v in args.items()
                                         if k in ('selector', 'n_dims', 'da', 'classifiers', 'base_dims')})
                args['hypothesis'] = hypothesis

                best = 0.0; best_clf = '?'
                if 'results' in r and r['results']:
                    best_clf = max(r['results'].items(), key=lambda x: x[1])[0]
                    best = r['results'][best_clf]

                    # 追踪覆盖率 (先记旧最优, track后再对比)
                    old_best = guard.best_accuracy
                    has_base = bool(args.get('base_dims'))
                    if has_base and len(args.get('base_dims', [])) <= args.get('n_dims', 120):
                        guard._control_var_done = True
                    guard.track_test(args['selector'], args['n_dims'], best, hypothesis, has_base, best_clf)
                elif verbose:
                    print('→ 评估失败')

                if sca_on:
                    for tag in guard.post_test(hypothesis, best, old_best):
                        if verbose: print(f'  [{tag["level"]}] {tag["tag"]}: {tag["message"]}')
                        messages.append({'role': 'user', 'content': f'[{tag["level"].upper()}] {tag["tag"]}: {tag["message"]}'})

                if verbose:
                    print(f'→ {r.get("actual_n_dims","?")}d, {best_clf}={best:.4f}')

            tool_log.append({'name': fn, 'arguments': args, 'result': r})
            assist_msg = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': tc.id, 'type': 'function',
                'function': {'name': fn, 'arguments': tc.function.arguments}
            }]}
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                assist_msg['reasoning_content'] = msg.reasoning_content
            messages.append(assist_msg)
            messages.append({'role': 'tool', 'tool_call_id': tc.id,
                             'content': json.dumps(r, ensure_ascii=False)})

            # 停滞3轮 + 未诊断 → 强制注入天花板追问
            if guard._stagnation >= 3:
                pass  # SCA注释: auto_diagnose

    best_acc, best_pipeline = 0.0, None
    for t in tool_log:
        if t['name'] == 'test_subset' and 'results' in t['result']:
            for clf, acc in t['result']['results'].items():
                if acc > best_acc:
                    best_acc = acc
                    rr = t['result']
                    best_pipeline = {'selector': rr['selector'], 'n_dims': rr['n_dims'],
                                     'actual_n_dims': rr.get('actual_n_dims', 0),
                                     'classifier': clf, 'accuracy': acc}

    n_try = len([t for t in tool_log if t['name'] == 'test_subset' and 'results' in t['result']])
    obs = f'{best_pipeline["selector"]}+{best_pipeline["actual_n_dims"]}d+{best_pipeline["classifier"]}={best_pipeline["accuracy"]:.4f}' if best_pipeline else ''
    return {
        'task': task_name, 'best_pipeline': best_pipeline,
        'n_calls': n_try, 'tool_log': tool_log,
        'memory_entry': {
            'best_pipeline': best_pipeline, 'observations': obs,
            'sca_state': guard.state_dict(),
        }
    }
