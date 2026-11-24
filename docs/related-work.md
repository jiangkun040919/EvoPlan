# 相关工作清单

精读笔记放在 `docs/papers/`，每篇一个文件。下表的作者、年份和发表信息在写进论文前需要逐条核对原文。

## 正在精读

| 论文 | 关注点 | 状态 |
|---|---|---|
| Hu et al., When Continual Learning Moves to Memory | 持续学习与智能体记忆的关系 | 精读中 |

## 2026 年前沿与对标工作

编号来自 arXiv，引用前逐条核对。该方向更新快，建议每周检索一次（agent memory / experience reuse / memory trust / negative transfer）。

| 论文 | 做了什么 | 与本课题的关系 |
|---|---|---|
| MemRL（arXiv 2601.03192） | 用环境反馈给每条经验算效用分，按效用检索；含 ALFWorld 实验 | 与"单一 Beta 打分"思路重合，必须作为基线；不区分适用场景 |
| FGSE（arXiv 2611.14326） | 把经验当假设，多组测试验证后才写入记忆 | 每条都验证，成本高；本课题只在拿不准时验证 |
| Useful Memories Become Faulty（arXiv 2605.12978） | 反复改写、合并后记忆变差，甚至不如无记忆 | 支持保留原始经历作证据（本仓库的证据日志） |
| MEMTRIM（arXiv 2610.07311） | 经验与任务"部分相似"时最容易误导 | 说明问题出在适用范围 |
| OEP（arXiv 2605.18930） | 攻击：用"局部正确但不可推广"的经验诱导出错误的通用规则 | 只有攻击、没有防御，是本课题错误经验注入实验的来源 |
| STALE（arXiv 2605.06527） | 记忆过时（隐式冲突）的基准 | 与时效性、证据打折相关 |
| Hu et al., When Continual Learning Moves to Memory（arXiv 2604.27003） | ALFWorld/BabyAI 持续任务实证，负迁移对难题伤害最大 | 直接参照，正在精读 |

## 时间衰减的依据

| 论文 | 与本课题的关系 |
|---|---|
| Raj & Kalyani, Taming Non-stationary Bandits: A Bayesian Approach（arXiv 1707.09727，2017） | 提出折扣 Thompson 采样（dTS）：对 Beta 后验的旧观测打折，适应非平稳环境；本仓库"旧证据打折"的依据 |

## 经验与记忆

| 论文 | 年份 | 与本课题的关系 |
|---|---|---|
| Reflexion: Language Agents with Verbal Reinforcement Learning（Shinn et al.） | 2023 | 用语言反思改进后续尝试；属于单任务内的经验利用 |
| ExpeL: LLM Agents Are Experiential Learners（Zhao et al.） | 2024 | 从轨迹中提取跨任务经验并检索复用，接近"直接经验检索"基线 |
| Agent Workflow Memory（Wang et al.） | 2024 | 归纳可复用的工作流 |
| Voyager: An Open-Ended Embodied Agent with Large Language Models（Wang et al.） | 2023 | 技能库持续积累 |
| Generative Agents: Interactive Simulacra of Human Behavior（Park et al.） | 2023 | 记忆检索综合考虑新近度、重要性和相关性 |
| MemGPT: Towards LLMs as Operating Systems（Packer et al.） | 2023 | 分层记忆管理 |

## 规划与执行

| 论文 | 年份 | 与本课题的关系 |
|---|---|---|
| ReAct: Synergizing Reasoning and Acting in Language Models（Yao et al.） | 2023 | 推理与行动交替，`LLMPolicy` 的基本形式 |

## 基准

| 基准 | 年份 |
|---|---|
| ALFWorld（Shridhar et al.） | 2021 |
| WebShop（Yao et al.） | 2022 |
| ScienceWorld（Wang et al.） | 2022 |
| TravelPlanner（Xie et al.） | 2024 |

## 需要补充检索的方向

- 智能体记忆中的经验可信度、冲突消解与遗忘机制（近两年）
- 迁移学习中的负迁移检测，及其在大模型智能体中的对应工作
- 持续学习/终身学习智能体的评测协议
