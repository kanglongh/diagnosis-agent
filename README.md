# diagnosis-agent

> LLM Agent 作为跨域故障诊断的搜索优化器——不做分类器，做决策层探索。
>
> 脱胎于 [vibrolab](https://github.com/kanglongh/vibrolab)，借由 AI 辅助开发，展现 SCA 安全架构的设计理念。

---

## 诚实声明

这个项目是我硕士毕业课题的衍生。vibrolab 证明了 Cohen's d + Bot-40 在 CWRU 跨域诊断上有效（3KB 模型烧进 ESP32），但我一直好奇：**Bot-40 是人工一刀切——能不能让 LLM 自己找到每个任务的最优解？**

因为个人精力和时间有限，这个项目的代码由 AI 辅助完成。它不是生产级代码，不是学术 SOTA，只是一个**架构设计的草图**——用来展现两个想法：

1. **LLM 在故障诊断中的正确角色不是分类器，是决策层搜索优化器**
2. **SCA（安全约束架构）可以从硬件安全移植到 LLM 探索行为的约束**

---

## 架构

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  LLM Agent  │────▶│  SCA 守卫     │────▶│  诊断管道      │
│  (决策层)    │     │  (约束层)     │     │  特征选择+分类  │
└─────────────┘     └──────────────┘     └───────────────┘
       ▲                    │                     │
       │                    ▼                     ▼
       │            追问/拦截/反思            evaluate()
       │                                         │
       └──────────── 结果反馈 ──────────────────┘
```

**SCA 守卫**不信任 LLM 的自律——不是"建议"Agent 做什么，而是拦住不经思考的探索行为：
- 消融未完成就选维 → 追问"你都不知道哪个模块是噪音"
- 连续多轮无进展 → 追问"是天花板还是没搜完"
- 消融发现超优结果但 Agent 没追 → 硬拦"为什么不追"

SCA 的设计理念来自 [embodied-ai-robot](https://github.com/kanglongh/embodied-ai-robot) 的硬件安全层——硬件钳舵机，软件钳探索。同一基因，不同战场。

---

## 核心实验结果

数据集: PU 轴承 (Paderborn University), N09→N15 跨域, 3 类

| 方法 | 精度 | 说明 |
|---|---|---|
| full120 全维 SVM | 0.75 | 基线 |
| **bot40** (Cohen's d) | 0.64 | **人工预设最优策略** |
| **Agent 搜索 (有 SCA)** | **0.8331** | **超越人工预设 +19pp** · [日志](logs/pu_scaon_0710_1401.log) |
| Agent 搜索 (无 SCA) | 0.8399 | 精度略高，但重复探索更多 · [日志](logs/pu_scaoff_0710_1405.log) |

SCA 不提升最终精度（天花板在 CFD 特征表达力），但**减少无意义的重复探索**、**推动 Agent 尝试被忽略的算法**。

---

## 目录

```
├── agent/
│   ├── core.py         # Agent 循环 + SCA 开关
│   ├── sca.py          # SCA 守卫 (消融不足/重复/停滞追问)
│   ├── pipeline.py     # 诊断管道 (评估/消融/权重分析)
│   ├── selectors.py    # 7 种特征选择算法
│   ├── da_methods.py   # 4 种域适应
│   ├── tools.py        # 分类器工厂
│   ├── cfd.py          # CFD 120 维提取 (从 vibrolab 粘贴)
│   └── io_utils.py     # 滑窗切分
├── run.py              # 入口 (--dataset pu/cwru --sca on/off)
├── prepare_pu.py       # PU 数据 → CFD 特征
├── prepare_cwru.py     # CWRU 数据 → CFD 特征
├── logs/               # SCA ON/OFF 对照实验日志
├── docs/
│   └── AGENT_FLOW.md   # 架构流程图
└── data/               # 数据目录 (需自行下载)
```

---

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填 LLM API key

# 下载 PU 数据集到 data/PU/ (链接见 data/README.md)
python prepare_pu.py
python run.py --dataset pu --sca on --model deepseek-v4-flash
```

---

## 项目起源

这个项目想回答一个问题：**能不能把过去几年的成果串起来？**

- vibrolab 证明了传统方法在跨域诊断上不输深度学习，且能部署到 40 元的 ESP32 开发板。同一套 CFD 特征体系迁移到 PU 轴承数据集上依然有效
- embodied-ai-robot 证明了 LLM 可以驱动真实物理设备，但需要 SCA 硬件安全层兜底

如果把 vibrolab 的诊断管道当作工具、把 SCA 移植到 LLM 探索行为约束、让 Agent 在 CFD 特征空间里自主搜索最优组合——这就是 diagnosis-agent。

它不是产品，不是论文，是我对自己过去思考的一次综合演练。

三个仓库做的是同一件事——**"用可接受的成本，让机器自己判断自己有没有故障"**——只是用了不同的技术路线：vibrolab 从信号处理入手做到边缘部署，robot 从 LLM 驱动硬件验证了安全约束的必要性，本项目把前两者的方法合并成一套可复现的 Agent 框架。

---

## 已知局限

- 代码由 AI 辅助生成，非生产级质量
- SCA 的追问效果受限于底层 LLM 的推理能力
- 仅在 PU 和 CWRU 上测试，未在其他数据集验证
- 同域探索深度不足（CWRU 同域近乎饱和，Agent 无优化空间）
- 一个待验证的猜想：LLM 的自主探索能力受限于提示词工程，更强的基座模型是否能突破当前的天花板？

---

## 相关项目

- [vibrolab](https://github.com/kanglongh/vibrolab) — CWRU 轴承跨工况故障诊断，ESP32-S3 边缘部署
- [embodied-ai-robot](https://github.com/kanglongh/embodied-ai-robot) — LLM 驱动的桌面具身机器人，MQTT + ESP32 三层架构

---

## 作者

康龙辉 · 燕山大学机械工程学院 2027 届硕士研究生
📧 [k3132755765@163.com](mailto:k3132755765@163.com) · GitHub [@kanglongh](https://github.com/kanglongh)

## License

MIT. 见 [LICENSE](LICENSE).
