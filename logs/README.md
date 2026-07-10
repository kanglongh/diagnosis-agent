# SCA ON vs SCA OFF · 对照实验日志

数据集: PU N09→N15 跨域, 3类, 120维 CFD
模型: DeepSeek V4 Flash
基线: full120=0.7533, bot40 (人工预设最优策略)=0.6424

Agent 工具箱: 5 种特征选择器 (weighted/fisher/mutual_info/l1_lasso/cohens_d)
7 种分类器 (SVM-RBF/SVM-Linear/LR/KNN-3/5/7/RF)

---

## 最终结果

| 配置 | 最优精度 | 轮次 | 最优组合 |
|---|---|---|---|
| SCA OFF | **0.8399** | 18 | weighted+100d+RF |
| SCA ON | **0.8331** | 18 | weighted+25d+SVM-RBF |

两个配置都大幅超越人工预设 (bot40=0.6424)，也超越全维基线 (0.7533)。
SCA 对最终精度无显著影响——天花板由 CFD 特征表达力决定，不是搜索策略。

---

## 探索过程对比

### 第一阶段：消融（两版本一致）

两个版本都独立发现了同一个关键事实：

> **关 time 模块后 SVM-RBF 从 0.45 跳到 0.65 —— time 是跨域噪声。**

这不是 SCA 提示的——Agent 自己从消融结果中推断出来的。说明 Agent 在没有任何约束的情况下，也能形成合理的科学假设。

**两者的消融策略略有不同：**
- SCA OFF: 逐个关 5 个模块（peaks→cepstral→time→freq→band）
- SCA ON: 同样逐个关，但顺序不同（peaks→time→freq→band→cepstral）

---

### 第二阶段：假设验证（两版本一致）

两个版本都验证了"去 time 提升精度"的假设：
- SCA OFF: 第 2 轮，base=108d (去time)，SVM-Linear=0.7898
- SCA ON: 第 2 轮，base=108d (去time)，SVM-Linear=0.7898

**完全相同的结果。** 在没有 SCA 干扰的情况下，Agent 独立完成了假设→验证的科学循环。

---

### 第三阶段：探索优化（关键分歧）

#### SCA OFF 的策略：系统化的维度精炼

Agent 始终保持 base=108d (去time)，在除去噪声模块后的子空间里系统性地探索：
- weighted 从 60d→90d→95d→98d→100d→105d→108d 逐档测试
- 尝试了 fisher、mutual_info、l1_lasso 作为补充
- 分类器从全量逐渐收敛到 KNN-5/KNN-7/RF 三选一
- **重复无效探索 4-5 次**（第 4-5 轮 weighted+108d 连试两次，第 7/10 轮本质上同一结果）

**最终：weighted+100d+RF=0.8399**

#### SCA ON 的策略：被 SCA 推动的多样化

SCA 的追问迫使 Agent 尝试了原本不会尝试的方向：
- 第 3 轮 SCA：`[challenge] 路径?: 未试:fisher,l1_lasso. 凭什么更差?`
- 第 5 轮 Agent 换 l1_lasso → **精度跳上 0.8202（新高）** ← SCA 直接生效
- 第 6 轮 Agent 换 fisher → 0.6085（无效）→ SCA 标记
- 第 8 轮 l1_lasso 再次尝试 → 0.8319（新高）

**但 SCA 也导致了探索碎片化：**
- Agent 被 SCA 推着不断换算法，丢掉了 SCA OFF 版本的"去 time 后在子空间精炼"这条主线
- 第 7-17 轮陷入了 l1_lasso 的反复微调，SCA 的重复检测标记了8次
- 最终未用 base_dims，在 full120 上靠 l1_lasso 自动稀疏到了 25d

**最终：l1_lasso+25d+SVM-RBF=0.8331**

---

## SCA 的影响量化

| 指标 | SCA OFF | SCA ON |
|---|---|---|
| 无效重复 | 4-5 次（无人指出） | 8 次（每次都被标记） |
| 被 SCA 推动的算法切换 | — | fisher(0.6085)、l1_lasso(0.8202) |
| 探索策略 | 主线清晰（去time→精炼） | 碎片化（频繁切换算法） |
| 最终精度的维度 | 100d（保守） | 25d（激进稀疏） |
| 最终精度 | 0.8399 | 0.8331 |

**SCA 推了一把好棋**（第 3→第 5 轮：从未试 l1_lasso 到 0.8202 新高），
**但也打乱了系统性探索**（Agent 在 SCA OFF 里能保持"去 time + 精炼"的主线，SCA ON 里主线被打断）。

---

## 日志格式说明

```
[1] weighted n=120 [SVM-RBF,SVM-Linear,LR,...] → 120d, SVM-Linear=0.7525
 ↑      ↑        ↑                                 ↑       ↑
轮次  选择器   目标维数                           实际维数 最佳分类器=精度

🔬[跨域] 关 ['time']: -12维, RF=0.5390
        消融模块       移除数  跨域精度

💭 Agent 的自主推理输出
[challenge] SCA 追问（消融不足/重复/路径依赖/天花板）
[encourage] SCA 肯定（新高/假设成立）
[reflect]   SCA 反思（假设落空/推论需修正）
[notice]    SCA 提醒
```

---

## 结论

1. **SCA 在引导探索方向上有效**——推 Agent 尝试了被忽略的 l1_lasso，直接创造了新高
2. **SCA 的追问频率需要调优**——每轮都问"路径依赖"导致 Agent 过度切换策略
3. **SCA 不是银弹**——它对最终精度无影响（天花板在特征池），但确实改变了探索行为
4. **Agent 的核心能力来自自身**——假设形成、消融推理、验证循环是 Agent 自主完成的，SCA 只是辅助
