# EvoPlan

**面向持续任务的智能体可信经验迁移与自适应规划方法研究**
Trustworthy Experience Transfer and Adaptive Planning for LLM Agents in Continual Task Streams

智能体在一连串任务中积累经验时，旧经验在新环境里可能失效，甚至起反作用（负迁移）。EvoPlan 研究三件事：经验什么时候可信、什么时候该迁移、计划什么时候该改。

[研究计划](docs/research-plan.md) · [相关工作](docs/related-work.md) · [论文笔记](docs/papers/)

## 方法组成

| 研究内容 | 模块 | 要点 |
|---|---|---|
| 1. 可信经验评估与记忆更新 | `evoplan/memory/trust.py`、`store.py` | Beta 后验，同时给出可信度和不确定性；旧证据按时间打折（久未验证 → 更不确定，而不是更不可信）；每条证据带来源回合和场景记入日志；局部可信度向全局均值收缩 |
| 2. 跨任务经验迁移与负迁移抑制 | `ExperienceStore.retrieve` | 按 相似度 × 场景可信度 排序；门控过滤低相似、低可信、已隔离的经验 |
| 3. 状态感知自适应规划 | `evoplan/planning/` | 状态监控器发现工具损坏、记忆与观测矛盾、连续失败时触发重规划，并把被否定的经验回传给记忆库 |
| 4. 实验原型与评估 | `experiments/run.py`、`evoplan/metrics.py` | 基线、本文方法、消融；成功率、迁移收益、负迁移率、恢复率、经验可靠性、Token 与成本 |

一个回合的流程：检索经验 → 生成计划 → 执行并监控 → 必要时重规划 → 用结果更新可信度 → 反思提取新经验。

## 快速开始

只依赖 Python 3.10+ 标准库，测试需要 pytest。

```sh
pip install -e ".[dev]"
pytest
python -m experiments.run --seeds 5
```

离线实验在内置的 ToyKitchen 小环境上运行，几秒钟出结果，写入 `results/<时间>/`（`summary.md`、`summary.json`、逐回合 `episodes.jsonl`、记忆快照）。

接入大模型（任何 OpenAI 兼容接口；论文实验用 DeepSeek `deepseek-flash` 和通义千问 `qwen-plus`）：

```sh
cp .env.example .env   # 填入地址、密钥、模型名
python -m experiments.run --policy llm --methods no_exp,naive_retrieval,evoplan --seeds 1 --schedule A:10,B:10
```

可信度随时间的处理方式可以切换，用于对比实验：

```sh
python -m experiments.run --decay-mode evidence   # 默认：旧证据打折，不确定性变大
python -m experiments.run --decay-mode to_half    # 对比：可信度往 0.5 回归
python -m experiments.run --decay-mode to_zero    # 对比：可信度往 0 衰减
```

## 对比方法

| 名称 | 说明 |
|---|---|
| `no_exp` | 无经验基线 |
| `naive_retrieval` | 直接经验检索基线：按相似度取 top-k，不评估可信度、不门控、不重规划 |
| `evoplan` | 本文方法 |
| `evoplan_no_trust` / `_no_ctx` / `_no_gate` / `_no_replan` | 消融：去掉可信度 / 改用全局可信度 / 去掉门控 / 去掉重规划 |

## 任务流

`--schedule` 用 `场景[@实际布局]:任务数` 描述持续任务序列，默认 `A:40,B:40,A:20,A@C:40`：

- `B:40`：切换到新场景，检验跨场景迁移；
- 再回到 `A`：检验迁移后旧知识是否被遗忘；
- `A@C:40`：标签仍是 A，但环境内部已经换成 C 的布局（环境漂移）。只靠相似度检索察觉不到这种变化。

## 当前状态与局限

- ToyKitchen 只用来验证流程和各模块的行为，**它的结果不能作为论文结论**。在这个环境里，有经验的方法明显优于无经验基线，但各方法之间差距很小，消融实验也没有稳定结论。
- 正式基准已定为 **ALFWorld**（Windows 上需在 WSL2 中运行），模型为 `deepseek-flash` / `qwen-plus`。新环境需实现 `evoplan/envs/base.py` 中的 `Env` 接口；如要计算经验可靠性，还要提供 `hint_valid`。
- `RulePlanner`、`HeuristicPolicy`、`init_beliefs` 和 `StateMonitor` 目前直接依赖 ToyKitchen 的物品、位置和状态字段；接 ALFWorld 前要先改成与环境无关的形式。
- 尚未实现：ALFWorld 接入；错误经验注入实验（核心实验）；失败时收窄经验的适用范围；按不确定性决定是否验证的规划；按阶段（act / reflect / verify）和缓存命中记录 Token；可信度准确性 AUROC、每成功任务 Token 等指标。
- 文本经验的相似度目前用词集合 Jaccard，接正式基准时换成向量检索。

## 目录

```
evoplan/
  types.py            Task / Step / Episode
  agent.py            智能体主循环与方法配置
  policy.py           动作策略（离线启发式 / LLM）
  memory/             经验、可信度、经验库、反思
  planning/           回合状态、状态监控、规划器
  envs/               环境接口、ToyKitchen、任务流
  llm/                OpenAI 兼容客户端
  metrics.py          评价指标
experiments/run.py    实验入口
tests/                单元测试
docs/                 研究计划、相关工作、论文笔记
```
