# DreamerV3 World Model Lab：从 CarRacing 到 Crafter 的训练、失效诊断与探索消融

> **实验研究报告 / 2026-11-19**  
> **状态**：Crafter 训练至 30k environment steps，完成 Actor/Reward/Ensemble/Intrinsic Reward 的阶段性研究；将无 Intrinsic Reward 的 30k Control 作为目前最优**候选**。  
> **实现基础**：从 [R2-Dreamer 官方实现](../README.md) 的 PyTorch DreamerV3 基线扩展。本文的工程改动、分析和本地实验与上游原作者的研究贡献应明确区分。不是 Crafter 官方排行榜结果。

## 摘要（Executive Summary）

这不是一次“运行 DreamerV3 就成功”的复现。项目先从 **CarRacing-v3** 的单机视觉强化学习流程验证出发；虽然重建/World Model Loss 明显下降，但驾驶评估没有改善。为更直接地研究长时序决策、稀疏成就奖励及探索行为，转向 **Crafter**。在 Crafter 中，早期 Actor 曾将大多数动作压到单一选择（尤其是 \`do\`，后续实验又出现 \`move_down\` 和 \`place_plant\` 偏好）。通过诊断 Mode/Sample 两种决策、调整策略熵系数、加入 5% ε-greedy 行为数据采集、延长在线训练，动作分布的塌缩显著缓解。

进一步检查发现：动作多样性提高 ≠ 奖励预测准确，尤其是 **RSSM Prior 对稀疏正奖励以及 \`do\` 操作的提前预测**依然较弱。对 Reward Head 增加正奖励加权 NLL 后，25k Actor 的游戏表现略好，但在**相同 5,209 个 transition**上的 Prior 正奖励召回并没有实质改善。最后，尝试三个轻量 Latent Prediction Head 形成 **Ensemble Disagreement Intrinsic Reward**。三个头能在留出数据上学习下一状态特征（MSE 1.0025 → 0.2783），但直接把固定 Ensemble 的分歧奖励送入 Actor-Critic 想象回报，在 25k→30k 消融中**使真实游戏奖励下降、动作重新集中、Critic 价值明显偏高**。因此该探索机制目前记为**负消融结果**，不声称探索能力得到提升。

**关键教训**：World Model Loss 下降不代表控制成功；高动作熵不代表有目的的探索；模型预测分歧存在不代表内在奖励有用；训练步数增加也不保证策略改善。

---

## 1. 研究目标与实验边界

- **主要目标**：在实际视觉环境上复现并理解 DreamerV3 风格的 Encoder → RSSM → Reward/Continue/Decoder → imagined Actor-Critic 训练链条，验证探索、奖励预测与策略行为的关系。
- **扩展目标**：可重复的断点续训、Replay Buffer、TensorBoard 指标、Crafter 原生 HUD 录像、Mode/Sample 行为诊断、Reward 加权消融，以及 Latent Disagreement 探索扩展。
- **实验设备**：Windows、Conda/PyTorch、NVIDIA RTX 5060 8 GB；显存约束促使我们选用有限 batch 与轻量 Ensemble。
- **范围限制**：目前训练预算最多 30k Crafter 环境步，远小于常见的大预算 Crafter 评估。**Episode reward 和每局不同 achievements 不是官方 Crafter Score**；不据此声称已经学会高级合成/真正通关。
- **研究性质**：一个主要 seed 的工程实验与小预算消融，结论是本实验配置下的证据，不是稳定的多 seed 统计结论。

## 2. 从 CarRacing 到 Crafter：为什么改变环境？

### 2.1 最初的 CarRacing / 自建 World Model 试验

在早期 \`world-model-lab\` 实验中，我们先为 CarRacing-v3 收集了 5 个 episode 的 \`.npz\` 数据，检查 \`observations\`、\`actions\`、\`rewards\`、\`dones\`，搭建过 CNN Encoder/Decoder、Gaussian RSSM、重建/奖励损失与 action-conditioned rollout 的初始练习。该阶段的意义在于理解连续动作和时序数据对齐，而不是证明驾驶策略已经收敛。

在随后 DreamerV3 CarRacing 小规模 smoke test 中：

| 项目 | 已记录情况 |
|---|---|
| 环境 | CarRacing-v3，RGB 96×96，连续动作：steering、gas、brake |
| 模型参数 | 约 **8.65M** |
| 实验预算 | 约 **1,000** 训练步的短程验证；预采集约 1,000 步 |
| World Model Loss | **406.42 → 135.12** |
| Image Loss | **401.57 → 129.68** |
| Evaluation return | 曾观察到约 **-78.98 → -84.88**（未改善） |
| 工程问题 | 一次 1k 后评估因 Replay/Checkpoint 恢复项不一致而失败，需要区分模型权重与完整训练状态的加载 |

**判断**：Loss 下降显示部分世界模型拟合进展，但无法直接推导驾驶策略成功；小规模 CarRacing 回报仍为负。后来转向 Crafter，主要为了快速分析离散动作、稀疏成就奖励、资源采集以及探索缺陷。上述 CarRacing 试验属于**前期另一份实验/代码库**，并非本仓库所有代码都直接运行了 CarRacing；因此此处是研究历程摘要，不将其误写为当前 \`r2dreamer\` 分支的可复现实验指标。

### 2.2 为什么采用 Crafter？

Crafter 提供 **17 个离散动作**：移动、交互 \`do\`、休息、放置及制造工具等。奖励与成就稀疏，且需要探索、采集与资源决策。它很适合判断一个模型究竟在“行动”还是在“学会行动”。

该环境使早期问题变得显眼：某些模型虽然稳定输出动作，却会过分频繁地使用某一个 action；Mode 策略有时会严重偏向同一操作，而 Sample 稍好。这并非传感器或游戏完全没有运行，而是**策略行为质量**的问题。

## 3. DreamerV3 训练链条与工程实现

### 3.1 核心模型

1. **Encoder**：把游戏图像编码成观察嵌入 \`e_t\`。
2. **RSSM**：包含确定性状态 \`h_t\` 与离散随机状态 \`z_t\`。Prior 不看下一帧预测 \`z_{t+1}\`；Posterior 借助真实 \`o_{t+1}\` 更新状态。训练中结合 KL 的 dynamics/representation 目标。
3. **Heads**：Reward Head 对奖励分布建模，Continue Head 预测 episode continuation，Dreamer 路线的 Decoder 负责重建图像。
4. **Actor-Critic**：从 posterior 状态出发，通过 RSSM 的 imagined rollout、预测奖励、折扣与 λ-return 更新策略和值函数；真实环境采样通过 Replay Buffer 反馈世界模型。
5. **评价**：\`mode\` = Actor 输出分布的最大概率动作；\`sample\` = 按 Actor 概率采样，不等于训练额外使用的 ε-greedy。

在 12M 级配置下，RSSM deterministic hidden 为 2048，stochastic 32×16=512，特征向量约 2560 维；Crafter Action 为 17 维 one-hot（实际以训练 config 和代码为准）。

### 3.2 关键工程修改与代码对应

| 功能 | 文件 | 解决什么 |
|---|---|---|
| 主训练与 Hydra 配置 | \`train.py\`, \`configs/\` | 实验设置、环境与网络装配 |
| 训练循环 | \`trainer.py\` | 在线收集、Mode/Sample 评估、ε-greedy、视频、日志、训练步数 |
| Dreamer & RSSM | \`dreamer.py\`, \`rssm.py\` | 世界模型 loss、reward weighting、imagination Actor-Critic |
| Replay | \`buffer.py\` | transition 对齐、序列采样、在线采样及 latent state |
| 完整断点续训 | \`checkpointing.py\` | agent、optimizer、replay、trainer counter、随机状态的保存/恢复 |
| Crafter HUD | \`hud_recorder.py\`, \`evaluate_hud.py\` | 不仅展示模型输入，还能查看原始画面、库存、实际成就与动作 |
| Actor 行为诊断 | \`diagnose_actor.py\` | 每局 Reward/Length/不同成就、动作直方图、经验熵、Mode/Sample |
| Reward 精度诊断 | \`diagnose_reward.py\` | 真实奖励与 Prior/Posterior 预测、按动作分析 |
| 固定数据公平比较 | \`compare_reward_shared.py\` | 在相同 transition 上比较两个 checkpoint |
| Auxiliary Ensemble | \`disagreement_ensemble.py\`, \`train_disagreement.py\` | 下一 posterior 特征投影预测、头间分歧、留出验证 |
| Intrinsic 奖励 | \`dreamer.py\`, \`train.py\` | 在 imagined reward 上加开关可控、截断的 exploration bonus |
| 防回归 smoke test | \`test_intrinsic_math.py\` | 检查分歧非负、形状、奖励时间对齐、裁剪上限 |

\`install_*.py\` 是开发过程中为本地已有代码添加功能的历史补丁脚本，不代表需要对最新版仓库反复执行安装。

### 3.3 实际解决过的工程故障

- **Checkpoint 与 Replay**：部分早期 checkpoint 只适合评估或载入权重，并不能无损地恢复 Replay/Optimizer；后来添加专门的完整断点续训路径。不要对不可信 \`.pt\` 使用 \`torch.load(weights_only=False)\`。
- **动作时间对齐**：必须区分“产生刚收到图像的上一动作”与“为下一次 env.step 选出的动作”，尤其 ε-greedy 改写 action 后，agent state 的 \`prev_action\` 也必须同步。
- **Crafter 成就统计**：\`log_achievement\` 是 episode 内累计计数，应取最终/最大值，不可逐帧求和。
- **HUD 与可视化**：图像张量与玩家看到的原生 Crafter HUD 不同；为分析 \`do\` / \`place_plant\` 行为增加原生 render + inventory/achievements 的记录。
- **Windows 运行**：保留训练产物在 \`logdir/\`，忽略 checkpoint/video/replay 等大文件，GitHub 主要保存可复现脚本与实验说明。

## 4. 关键失效：Actor 一开始偏向单一 Action

### 4.1 问题的演变，不是一次性解决

早期 Crafter 预热/短预算训练曾出现 **\`do\` 占约 98.9%** 的动作极端集中，经验动作熵约 0.087，阶段性的平均评估 Reward 曾只有 0.2/0.4 左右。这个结果是早期低预算/不同配置的诊断观察，不能与后面的 30-episode 固定评估直接混为一表。

随后，5k 的初始 Actor 较容易偏向 **\`place_plant\`** 等单个选择。在提高策略 entropy 系数（\`model.act_entropy\` 从 0.0003 到 **0.003**）后，动作分布改善，但某轮 **Mode \`move_down\` 约 54.9%**，**\`do\` 23.7%**，**\`place_plant\` 15.2%**，因此不是从此完全解决。

继续加入 **训练采集阶段 ε=0.05** 的随机动作（评估阶段不使用 ε 替代 policy action），并从 5k 扩展到 20k，动作分布更广、Reward/成就指标也逐步改善。

**诊断方法**：

- Mode 检测 greedy policy 对少数动作的固着倾向；
- Sample 测试真实 stochastic policy 的分布与成绩；
- 经验熵 \`-sum_a p(a) log p(a)\` 测试**整局/全局动作统计分布**，它不等于 \`train/action_entropy\` 所记录的每状态 policy entropy；
- 真实游戏 HUD、Inventory 和 achievements 结合动作频率，检查重复操作是否产生资源/成就；
- 不能只凭动作类别占比证明“原地卡住”，需要位置、轨迹或视频证据。

### 4.2 早期 Actor 诊断对照

| 阶段 / 配置 | Episode/策略 | Mode Reward | Sample Reward | Mode 经验熵 | Sample 经验熵 | 主要诊断 |
|---|---|---:|---:|---:|---:|---|
| 5k Baseline, entropy=0.0003 | 10/策略 | 1.50 | 1.90 | 约 1.026 | 1.130 | 存在显著动作偏好 |
| 5k Entropy×10, entropy=0.003 | 10/策略 | 2.00 | 2.20 | 1.165 | 1.548 | Mode \`move_down\` 54.9%，并未完全解除偏好 |
| 5k Entropy×10 + ε=0.05 | 10/策略 | 1.90* | 2.60 | — | 1.891 | Sample Reward 与成就（3.5）提高 |
| 20k Entropy×10 + ε=0.05 | 10/策略 | 2.30 | 2.60 | 2.600 | 2.686 | 经验熵更高，动作塌缩缓解 |

\`*\` 注：5k ε 组在不同时间记录/测试版本中曾见 Mode 1.9/2.3；该表沿用同一批早期汇总中 1.9 的口径，不把不同评估批次拼成“确定的单一结果”。20k Mode 2.3 与 Sample 2.6 为后续 10 局诊断。它们与下面 25k/30k 的 30 局评估不是严格配对的统计检验。

**阶段结论**：Entropy regularization + ε-greedy online collection 使策略从极端少数动作选择转向更多样化，但多样性仍不能保证动作有效或高奖励。

## 5. Reward Model：为什么 Actor 多样了还不够？

### 5.1 20k 的 Prior / Posterior 诊断

在一次 20k checkpoint、约 1,586 transitions 的 10-episode Sample 评估中，有 35 个正奖励、39 个负奖励，剩余约 1,512 个为零奖励。稀疏事件导致平均误差可能看似不高，但真正决定能否计划的**正奖励召回**仍较低。

| 项目 | Prior | Posterior |
|---|---:|---:|
| MAE | 0.0240 | 0.0209 |
| RMSE | 0.1056 | 0.0959 |
| 正奖励 Recall（原诊断阈值 0.5） | 45.7% | 57.1% |
| 零奖励 FPR | 0.73% | 0.66% |

特别是 \`do\` 的 Prior 正奖励 Recall 曾只有 **22.2%（2/9）**。但注意，奖励中有 **+0.1** 这样的小正奖励，用预测阈值 0.5 来判定其是否“正确识别”并不合理。因此报告固定数据对照时明确使用阈值 **0.05**；阈值不同的 Recall **不能**直接跨表对比。

### 5.2 Reward-positive weighted NLL（25k）

在 20k checkpoint 上分叉，保持其余配置相同、各增加 5k steps：

- **Control**：\`+model.reward_pos_weight=1.0\`；
- **Weighted**：\`+model.reward_pos_weight=3.0\`；
- Loss 用正奖励样本加权的 NLL，并按权重和归一化；原始 reward 标签不更改，\`reward_pos_weight=1.0\` 等价于不加权。

| 25k 模型，30 局/策略 | Control | Weighted |
|---|---:|---:|
| Mode mean reward | 1.933 | **2.233** |
| Sample mean reward | 1.967 | **2.333** |
| Mode mean distinct achievements | 2.833 | **3.133** |
| Sample mean distinct achievements | 2.867 | **3.300** |
| Sample empirical action entropy | **2.711** | 2.672 |

**但策略收益不等于 Reward Model 的提前预测能力改善**：两组 Actor 各自产生不同的状态分布，因此又做了下方同数据测试。

### 5.3 Shared-transition reward evaluation（公平离线比较）

两组 Reward Head 在**相同 5,209 个 transition**上逐一评估（共 30 episodes、130 positive、4,957 zero，另有负奖励；数据由两个 Actor 各采集部分），使用相同判定阈值 **0.05**。

| 指标 | Control 25k | Weighted 25k |
|---|---:|---:|
| Prior MAE（↓） | **0.03527** | 0.03613 |
| Prior RMSE（↓） | **0.14088** | 0.14677 |
| Prior positive recall（↑） | 36.15% | **36.92%** |
| Prior zero-reward FPR（↓） | 5.29% | **4.66%** |
| Posterior positive recall（↑） | 59.23% | **63.08%** |
| Posterior zero-reward FPR（↓） | 4.08% | **3.77%** |

\`do\` 的 Prior Recall **29.41% vs 31.37%**，但低阈值零奖励 FPR **29.72% vs 26.27%**，依然存在严重的“交互看起来会有收益”的误报问题。

**结论**：Weighted 模型在有限游戏评估中的表现较好，但 Prior 提前识别正奖励的改进仅 **0.77 个百分点**，不足以证明奖励预测问题已经解决。没有继续无限提高正样本权重。

## 6. Ensemble：额外三个预测头究竟预测了什么？

新增的不是三个完整的 RSSM，也不替换原模型。原 RSSM 的 Posterior 从下一帧观测得到 \`feat_{t+1}^{post}\`；将其经固定投影与 LayerNorm 压成 **128 维 target**。三个独立、bootstrap 训练的小 MLP 输入 \`(feat_t^{post}, action_t)\`，分别预测这个未来 128 维表示：

\`\`\`text
Crafter obs_t → Encoder → RSSM posterior → feat_t
                                      + action_t
                                        ├─ predictor_1 → y_hat_{t+1}^1
                                        ├─ predictor_2 → y_hat_{t+1}^2
                                        └─ predictor_3 → y_hat_{t+1}^3
真实 obs_{t+1} → RSSM posterior → fixed projection → target y_{t+1}
训练：每个 head MSE(y_hat, stop_gradient(target))
分歧：不同 head 的同维预测方差再取均值
\`\`\`

三头保持不同初始化，并通过 bootstrap mask 训练。它们的**目标**是估计模型之间对未来预测的不同意见（epistemic uncertainty 的一种粗略代理），将来可能作为探索奖励，而不是直接提高 Reward Head 的准确率。

### 6.1 离线 Stage A（20 Epoch）

| 指标 | 结果 |
|---|---:|
| 训练 Transition | 5,120（30 episodes） |
| 验证 Transition | 1,515（8 个不同 episodes） |
| 训练前 Validation MSE | 1.0025 |
| 训练后 Epoch 20 Validation MSE | **0.2783**（下降约 72.2%） |
| 最低观察到的 Validation MSE | **0.2751**（Epoch 18） |
| Mean disagreement：Epoch 1 → 20 | 0.02222 → 0.01035 |

解释：三头学到了部分下一状态代理特征；验证误差后期趋于平台。但**低 MSE 不证明高 Disagreement 必然对应新区域**，低分歧也可能意味着预测头共同出错。目前 Stage A 尚未独立完成“熟悉/陌生场景的分歧可分性”实验；因此不应宣称探索能力已验证。

## 7. Intrinsic Reward：从 25k 到 30k 的最终消融

### 7.1 两组严格共享起点

- 同一个 **Weighted 25k checkpoint + Replay Buffer**，共同继续至 **30k env steps**，基本训练超参数一致；
- **Control 30k**：\`intrinsic_beta=0.0\`，只用原 game Reward；
- **Intrinsic 30k**：加载 Stage A 训练好的三个冻结预测头，\`beta=0.05\`，\`scale=0.01035\`，\`clip=3.0\`；该探索奖励在 imagination Actor-Critic 目标中引入，**不写入 Replay Buffer 的真实奖励，也不改 Reward Head 的监督标签**；
- 单步奖励上限为 \`0.05×3=0.15\`，将 \`(feat_t, action_t)\` 分歧附着到 imagined **next-state** reward；
- Ensemble **冻结**，但基础 RSSM 继续变化：这会导致 Predictor 输入的 feature space drift，是当前实现的重要局限。

### 7.2 30-episode × 2 policy 行为对照（主要实验）

| 最终指标 | Control 30k | Intrinsic 30k | 观察 |
|---|---:|---:|---|
| **Mode 平均 Reward** | **3.133** | 2.200 | Intrinsic -29.8% |
| Sample 平均 Reward | **2.900** | 2.767 | Intrinsic -4.6% |
| **Mode 不同成就数** | **4.100** | 3.167 | Intrinsic -22.8% |
| Sample 不同成就数 | **3.833** | 3.700 | Intrinsic -3.5% |
| Mode 经验动作熵 | **2.576** | 2.034 | 下降 |
| Sample 经验动作熵 | **2.700** | 2.206 | 下降 |
| Mode episode length | 185.27 | **197.53** | Intrinsic 生存略久 |
| Sample episode length | 187.97 | **200.30** | Intrinsic 生存略久 |

**动作偏好竟然回来了**：Intrinsic Mode 的 \`move_right\` **28.81%** + \`move_up\` **25.45%** = **54.26%**；Control 两动作合计只 **25.62%**。Sample 的 \`move_up\` 26.76% + \`move_right\` 25.38% 也超过 52%。这是动作分布再次集中，而不是足以证明它原地不动（还需位置轨迹与 HUD 验证）。

很重要的是，20k 时我们刚处理过单一动作偏好；**并不是增加了一个“促进探索”的模块，就必然能扩大实际动作覆盖与成就**。

### 7.3 TensorBoard：为什么性能可能下降？

对应两组 25k→30k 的 event logs，训练后期观察到：

| TensorBoard 标量 | Control 起点→终点 | Intrinsic 起点→终点 |
|---|---:|---:|
| \`train/intrinsic/bonus_mean\` | 关闭 | 0.04670 → **0.05819** |
| \`train/intrinsic/bonus_max\` | 关闭 | 0.13222 → **0.14301** |
| \`train/intrinsic/disagreement_mean\` | 关闭 | 0.01031 → **0.01285** |
| \`train/rew\`（想象游戏 Reward 均值） | 0.02165 → 0.02401 | 0.02019 → 0.02716 |
| **\`train/val\`（Critic 预测值）** | 2.620 → **2.441** | 2.601 → **8.793** |
| **\`train/tar\`（回报目标）** | 2.701 → **2.498** | 2.990 → **9.160** |
| \`train/action_entropy\`（每状态策略熵） | 1.508 → **1.557** | 1.522 → **1.240** |
| \`episode/eval_score\`（训练期 10 局） | 2.10 → **2.30** | 2.00 → **1.10** |
| \`train/opt/nonfinite_grads\` | 0 → 0 | 0 → 0 |

这里 **\`train/val\` 增加不等于 Agent 真的获得了更多游戏奖励**：Critic/Return 可能被模型想象中的探索奖励推高。Frozen Ensemble 与 RSSM 表征漂移、奖励归一化/尺度、Actor 在高分歧动作附近的行为偏好，都属于合理机制假设；但还没有通过专门的反事实测试**唯一确定**成因。未出现非有限梯度，故不宜称作 NaN 数值溢出。

需要区分训练期 \`episode/eval_score\` 的两次、每次约 10-episode 结果，与独立 \`diagnose_actor.py\` 的各 30 局结果；**上表的第 7.2 节才是主要行为对照**。

### 7.4 对失败机制的技术讨论（不做过度归因）

1. **不确定性 ≠ 有意义的 novelty**：多网络不同意见可能来自感知噪声、随机性、稀疏数据和模型共同偏差；高方差动作未必带来成就。
2. **冻结 Ensemble + 更新 RSSM**：预测头在 25k 的 feature space 训练；RSSM 到 30k 后可能发生分布变化，所以分歧不再只反映“未见过的场景”。
3. **奖励尺度影响 Actor/Critic**：虽有 clip，\`bonus_mean≈0.058\` 在同时间区间里已大于报告的 \`imag_reward mean≈0.027\`，两者口径不同但说明 bonus 可能不可忽略；Critic 值目标实际明显抬高。
4. **混合价值目标的设计局限**：内在 reward 只用于 imagination，Replay 值回归仍含真实 reward 与想象 value bootstrap。长期更合理的版本可能需要单独 intrinsic critic、在线更新 ensemble、目标/表征固定与跨多 seed 验证。
5. **行为需看成就，不止看熵**：高/低动作熵只是分布指标，必须结合真实环境 Reward、不同成就、连续重复率、位移与 HUD 观察。

**阶段性结论**：Control 30k 当前表现更好。将 Intrinsic 30k 明确记录为**负消融实验**。现有结果不支持“Ensemble 已提高探索”或“30k 内所有策略都会随着训练单调提高”的主张。

## 8. 全阶段结果速览与正确解释

| 阶段 | 关键操作 | 有证据的结果 | 不应推断的事 |
|---|---|---|---|
| CarRacing smoke | 训练基础 World Model | Loss 下降，但驾驶回报未改善 | Loss 下降 = 学会驾驶 |
| Crafter 初期 | 在线采样、基本 Actor-Critic | \`do\` / \`place_plant\` 等单动作偏好 | 环境能跑 = Agent 能完成任务 |
| 5k entropy ablation | Entropy 系数×10 | Sample Reward 1.9→2.2，动作熵改善 | 完全消除动作偏好 |
| 5k→20k ε+entropy | 额外 5% 随机采集、更多 env steps | 动作分布明显广泛，Reward 约 2.6 | 多样性 = 有效探索 |
| 20k→25k reward weights | 正奖励 NLL ×3 | Weighted Actor 评估略好 | Prior 正奖励预测已准确 |
| Shared reward comparison | 相同 5,209 transitions | Prior Recall 36.15% vs 36.92% | 加权一定显著有效 |
| Ensemble offline | 三个 predictor 训练 20 epochs | 验证 MSE 降至 0.2783 | 分歧已被证明是有效 novelty |
| 25k→30k intrinsic | bonus vs matched control | Mode Reward 2.20 vs 3.13（变差） | 接入探索 reward 就会变好 |

### 模型选择

当前推荐将 **\`logdir/crafter_intrinsic_control_30k/latest.pt\`** 作为项目展示的最佳**候选**，同时保留 \`crafter_reward_weighted_25k/checkpoint_25000.pt\` 作实验共同起点。模型权重位于本地训练目录，不应将数百 MB/GB 的完整 Replay Checkpoint 提交到 Git。

## 9. 如何复现实验与评估

### 9.1 环境及数据注意事项

本项目使用本地 Windows / Conda/PyTorch，需安装仓库对应依赖与 Crafter extra；请以你本机已验证的 \`pytorch_env\` 为准。启用 \`model.rep_loss=dreamer\` 时使用 Dreamer 视觉重建路线（与 R2-Dreamer 的默认 \`rep_loss\` 不同）。

每次续训前确认 **resume checkpoint 含完整 replay/trainer state**；如果 \`checkpoint_25000.pt\` 不在，\`latest.pt\` 只有验证其确实为 25k 后才可替代。不要对原有 checkpoints 执行覆盖。

### 9.2 25k→30k 对照组（PowerShell）

\`\`\`powershell
python train.py \`
  env=crafter model.rep_loss=dreamer model.compile=false \`
  model.log_grads=true model.act_entropy=0.003 \`
  +model.reward_pos_weight=3.0 \`
  batch_size=16 batch_length=64 \`
  env.steps=30000 env.train_ratio=512 \`
  env.eval_episode_num=10 trainer.eval_every=5000 \`
  trainer.update_log_every=500 trainer.video_pred_log=false \`
  buffer.max_size=30000 buffer.storage_device=cpu \`
  device=cuda:0 seed=0 \`
  +checkpoint_every=5000 +trainer.explore_epsilon=0.05 \`
  +resume=logdir/crafter_reward_weighted_25k/checkpoint_25000.pt \`
  +intrinsic_beta=0.0 \`
  logdir=logdir/crafter_intrinsic_control_30k
\`\`\`

### 9.3 25k→30k Intrinsic 组（PowerShell）

\`\`\`powershell
python train.py \`
  env=crafter model.rep_loss=dreamer model.compile=false \`
  model.log_grads=true model.act_entropy=0.003 \`
  +model.reward_pos_weight=3.0 \`
  batch_size=16 batch_length=64 \`
  env.steps=30000 env.train_ratio=512 \`
  env.eval_episode_num=10 trainer.eval_every=5000 \`
  trainer.update_log_every=500 trainer.video_pred_log=false \`
  buffer.max_size=30000 buffer.storage_device=cpu \`
  device=cuda:0 seed=0 \`
  +checkpoint_every=5000 +trainer.explore_epsilon=0.05 \`
  +resume=logdir/crafter_reward_weighted_25k/checkpoint_25000.pt \`
  +intrinsic_beta=0.05 +intrinsic_scale=0.01035 +intrinsic_clip=3.0 \`
  +intrinsic_ensemble_path=logdir/crafter_reward_weighted_25k/ensemble_diagnostics/disagreement_ensemble.pt \`
  logdir=logdir/crafter_intrinsic_bonus_30k
\`\`\`

### 9.4 训练 Ensemble（Stage A）

\`\`\`powershell
python train_disagreement.py \`
  --run logdir/crafter_reward_weighted_25k \`
  --episodes 30 --validation-episodes 8 --epochs 20 --batch-size 128
python test_intrinsic_math.py
\`\`\`

### 9.5 最终行为评估（每组 Mode 30 + Sample 30）

\`\`\`powershell
python diagnose_actor.py --run logdir/crafter_intrinsic_control_30k --episodes 30
python diagnose_actor.py --run logdir/crafter_intrinsic_bonus_30k --episodes 30
\`\`\`

输出 \`actor_diagnostics/summary.json\`、\`episodes.csv\`、\`action_frequency.csv\`。诊断报告反映最终策略，不等于 \`train/intrinsic/bonus_mean\` 等训练中指标。若要评估真正的重复/卡住行为，另结合 \`--record-first\` 生成 HUD 录像、角色位移和动作连续 run length。

### 9.6 配置、数据与许可

- 保持 \`logdir/\`、\`*.pt\`、录像、大型 Replay、个人配置/密钥不进入 Git。
- 受信任的本地 checkpoint 加载才能允许 \`weights_only=False\`；不要下载并执行不明来源的 PyTorch pickle 文件。
- 研究报告描述的是用户对上游实现的扩展和本地复现，仍保留仓库中原研究团队的 README、citation、license。
- 不把不同时间、不同 Episode 数和不同 seed 的单个均值当作严格显著性结果；如需论文级结论，应做多 seed、统一评估与置信区间。

## 10. 当前结论与下一阶段

**已完成**：CarRacing 到 Crafter 的研究迁移、线上/回放训练与断点续训、动作偏好诊断与 Entropy/ε 调整、20k–25k Reward 加权消融、共享轨迹 Reward Head 对照、Ensemble 三头离线训练、25k–30k Intrinsic vs Control 消融、训练标量与最终行为对照。

**确定性最高的结论**：

1. 早期严重 Action Bias 是真实遇到的问题，且随 entropy/采样策略调整得到了缓解；有多轮指标和代码佐证。
2. Weighted Reward Loss 改善过有限预算下的 Actor 游戏表现，但 Shared Prior 正奖励 Recall 几乎没有提升。
3. Ensemble Head 预测未来 128-D 表征的能力得到验证，但**目前没有验证其 novelty 排序的有效性**。
4. 当前 Intrinsic 实现对实际游戏表现有负面影响，与 Critic target/value 抬升和动作集中化同时出现；不能宣称机制已被证明，也不能据此否定所有 intrinsic motivation 方法。
5. **训练更多 steps/epochs 并不保证更好**，必须用真实环境评估与最佳 checkpoint 选择。

**最终推荐**：保留 Control 30k 作为目前最佳候选，不继续盲目延长 Intrinsic 训练。若未来返回此课题，再研究固定/缓慢变化的 target encoder、可在线更新或校准的 Ensemble、分离的 intrinsic critic、状态-动作新颖度与多 seed 消融；本阶段项目可以正式收尾，进入 V-JEPA / action-conditioned world representation 方向。

---

**实验记录引用范围**：用户提供的训练终端日志、Actor/Reward JSON+CSV、共享 transition 诊断、Ensemble summary、30k 两份 TensorBoard events，以及当前仓库代码。CarRacing 早期数据来自此前相关试验记录而非此分支可运行结果。所有数据均为局部实验观测，非官方 SOTA 指标。
