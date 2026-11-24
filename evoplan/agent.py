"""Agent loop and method configurations (proposed method, baselines, ablations)."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from evoplan.envs.base import Env
from evoplan.memory.reflection import Reflector
from evoplan.memory.store import ExperienceStore
from evoplan.planning.context import AgentContext, init_beliefs
from evoplan.planning.monitor import StateMonitor
from evoplan.planning.planner import RulePlanner
from evoplan.types import Episode, Step, Task


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_memory: bool = True     # retrieve and inject experience
    use_trust: bool = True      # rank by similarity x trust and update trust from outcomes
    gate: bool = True           # transfer gate: drop low-similarity / low-trust experience
    contextual: bool = True     # context-conditional trust (vs. one global score)
    replan: bool = True         # state monitor + replanning
    k: int = 3


METHODS: dict[str, MethodConfig] = {
    # baselines
    "no_exp": MethodConfig("no_exp", use_memory=False, use_trust=False, gate=False, contextual=False, replan=False),
    "naive_retrieval": MethodConfig("naive_retrieval", use_trust=False, gate=False, contextual=False, replan=False),
    # proposed
    "evoplan": MethodConfig("evoplan"),
    # ablations
    "evoplan_no_trust": MethodConfig("evoplan_no_trust", use_trust=False),
    "evoplan_no_ctx": MethodConfig("evoplan_no_ctx", contextual=False),
    "evoplan_no_gate": MethodConfig("evoplan_no_gate", gate=False),
    "evoplan_no_replan": MethodConfig("evoplan_no_replan", replan=False),
}


class Agent:
    def __init__(self, cfg: MethodConfig, policy, planner: RulePlanner, reflector: Reflector | None,
                 store: ExperienceStore | None, monitor: StateMonitor | None = None) -> None:
        self.cfg = cfg
        self.policy = policy
        self.planner = planner
        self.reflector = reflector
        self.store = store if cfg.use_memory else None
        self.monitor = (monitor or StateMonitor()) if cfg.replan else None

    def run(self, env: Env, task: Task, index: int, seed: int) -> Episode:
        cfg, store = self.cfg, self.store
        ep = Episode(task=task, method=cfg.name, seed=seed, index=index)
        obs, info = env.reset(task)

        hints = []
        if store is not None:
            hints = store.retrieve(task.context(), now=index, k=cfg.k, use_trust=cfg.use_trust, gate=cfg.gate,
                                   ctx=task.domain, min_trust=0.55 if cfg.use_trust else 0.0)
        rng = random.Random(zlib.crc32(f"{seed}:{task.id}".encode()))
        ctx = AgentContext(task=task, obs=obs, info=info, hints=hints, beliefs=init_beliefs(task, hints, rng))
        ep.used_experiences = [h.exp.id for h in hints]
        ep.context_chars = sum(len(h.exp.text) for h in hints)
        oracle = getattr(env, "hint_valid", None)  # evaluation only
        if oracle is not None:
            ep.hint_valid = [oracle(h.exp.key) for h in hints]
        self.planner.plan(ctx)

        for _ in range(env.max_steps):
            action = self.policy.act(ctx)
            obs, ok, done, info = env.step(action)
            step = Step(action, obs, ok, info.get("event"))
            ep.steps.append(step)
            ctx.observe(step, info)
            if done:
                ep.success = True
                break
            if self.monitor is not None:
                trig = self.monitor.check(ctx, step, info)
                if trig is not None:
                    ep.replans += 1
                    ep.contradicted += self.planner.replan(ctx, trig)

        ep.prompt_tokens += ctx.prompt_tokens
        ep.completion_tokens += ctx.completion_tokens
        if store is not None:
            if cfg.use_trust:
                store.record_outcome(ep.used_experiences, ep.contradicted, ep.success, now=index, ctx=task.domain)
            if self.reflector is not None:
                for exp in self.reflector.reflect(ep):
                    store.add(exp, ctx=task.domain)
        return ep
