"""Run methods over continual task streams and report metrics per phase.

Offline (no API key, seconds):
    python -m experiments.run --methods all --seeds 5

With an LLM (OpenAI-compatible endpoint configured in .env):
    python -m experiments.run --policy llm --methods no_exp,naive_retrieval,evoplan --seeds 1 --schedule A:10,B:10
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from evoplan.agent import METHODS, Agent
from evoplan.envs import ToyKitchen, make_stream, parse_schedule
from evoplan.memory.reflection import LLMReflector, RuleReflector
from evoplan.memory.store import ExperienceStore
from evoplan.memory.trust import DECAY_MODES, TrustConfig
from evoplan.metrics import group, summarize
from evoplan.planning.planner import LLMPlanner, RulePlanner
from evoplan.policy import HeuristicPolicy, LLMPolicy
from evoplan.types import Episode

COLUMNS = ["success_rate", "avg_steps", "transfer_gain", "step_saving", "negative_transfer",
           "recovery_rate", "reliability", "avg_hints", "replans", "prompt_tokens"]


def build_agent(name: str, env: ToyKitchen, policy_kind: str, llm=None, decay_mode: str = "evidence",
                gamma: float = 0.99) -> Agent:
    cfg = METHODS[name]
    trust_cfg = TrustConfig(contextual=cfg.contextual, decay_mode=decay_mode, gamma=gamma)
    store = ExperienceStore(trust_cfg) if cfg.use_memory else None
    if policy_kind == "llm":
        return Agent(cfg, LLMPolicy(llm, env.admissible_actions), LLMPlanner(llm), LLMReflector(llm), store)
    return Agent(cfg, HeuristicPolicy(), RulePlanner(), RuleReflector(), store)


def run(args: argparse.Namespace) -> Path:
    methods = list(METHODS) if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    if "no_exp" not in methods:
        methods.insert(0, "no_exp")  # paired metrics need the baseline
    schedule = parse_schedule(args.schedule)
    llm = None
    if args.policy == "llm":
        from evoplan.llm import OpenAICompatLLM, load_dotenv
        load_dotenv()
        llm = OpenAICompatLLM()

    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: dict[str, list[Episode]] = {m: [] for m in methods}
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for seed in range(args.seeds):
            stream = make_stream(schedule, seed)
            for name in methods:
                env = ToyKitchen(seed=seed, fault_rate=args.fault_rate, relocate_rate=args.relocate_rate,
                                 max_steps=args.max_steps)
                agent = build_agent(name, env, args.policy, llm, args.decay_mode, args.gamma)
                for index, (phase, task) in enumerate(stream):
                    ep = agent.run(env, task, index, seed)
                    episodes[name].append(ep)
                    f.write(json.dumps({"phase": phase, **ep.to_dict()}, ensure_ascii=False) + "\n")
                if agent.store is not None and seed == 0:
                    agent.store.save(out_dir / f"memory-{name}-seed0.jsonl")

    phase_of = {}
    for seed in range(args.seeds):
        for phase, task in make_stream(schedule, seed):
            phase_of[(seed, task.id)] = phase
    labels = [f"{i}:{p.label}" for i, p in enumerate(schedule)]
    report: dict = {"config": vars(args), "overall": {}, "phases": {}}
    price = (args.price_in, args.price_out)
    for name in methods:
        report["overall"][name] = summarize(episodes[name], episodes["no_exp"], args.max_steps, price_per_1k=price)
        by_phase = group(episodes[name], lambda e: phase_of[(e.seed, e.task.id)])
        base_phase = group(episodes["no_exp"], lambda e: phase_of[(e.seed, e.task.id)])
        report["phases"][name] = {labels[p]: summarize(eps, base_phase[p], args.max_steps, price_per_1k=price)
                                  for p, eps in sorted(by_phase.items())}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render(report, methods, labels)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nResults written to {out_dir}")
    return out_dir


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def render(report: dict, methods: list[str], labels: list[str]) -> str:
    lines = ["## Overall", "", "| method | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for m in methods:
        r = report["overall"][m]
        lines.append(f"| {m} | " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |")
    cols = ["success_rate", "avg_steps", "negative_transfer", "recovery_rate"]
    for label in labels:
        lines += ["", f"## Phase {label}", "", "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for m in methods:
            r = report["phases"][m].get(label, {})
            lines.append(f"| {m} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="all", help=f"comma list or 'all': {', '.join(METHODS)}")
    p.add_argument("--schedule", default="A:40,B:40,A:20,A@C:40", help="phases as DOMAIN:N_TASKS")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--policy", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--fault-rate", type=float, default=0.08)
    p.add_argument("--relocate-rate", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--decay-mode", choices=DECAY_MODES, default="evidence",
                   help="how trust changes over time: discount evidence (default), pull to 0.5, or decay to 0")
    p.add_argument("--gamma", type=float, default=0.99, help="per-episode discount used by --decay-mode")
    p.add_argument("--price-in", type=float, default=0.0, help="price per 1k prompt tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="price per 1k completion tokens")
    p.add_argument("--out", default="results")
    run(p.parse_args())


if __name__ == "__main__":
    main()
