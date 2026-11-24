# 相关工作清单

精读笔记放在 `docs/papers/`，每篇一个文件。下表的作者、年份和发表信息在写进论文前需要逐条核对原文。

## 正在精读

| 论文 | 关注点 | 状态 |
|---|---|---|
| Hu et al., When Continual Learning Moves to Memory | 持续学习与智能体记忆的关系 | 精读中 |

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
