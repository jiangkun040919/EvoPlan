"""Per-episode working state shared by the planner, monitor and policy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from evoplan.envs.toy_kitchen import KINDS, LOCATIONS
from evoplan.memory.store import Retrieved
from evoplan.types import Step, Task


@dataclass
class Beliefs:
    """What the agent currently assumes. Hints from experience seed these; replanning revises them."""

    obj_guess: str | None = None
    obj_hint: str | None = None          # id of the experience behind obj_guess
    tool_pref: str | None = None
    tool_hint: str | None = None
    avoid_tools: set[str] = field(default_factory=set)
    searched: set[str] = field(default_factory=set)
    search_order: list[str] = field(default_factory=list)
    tool_order: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    task: Task
    obs: str
    info: dict[str, Any]
    hints: list[Retrieved] = field(default_factory=list)
    beliefs: Beliefs = field(default_factory=Beliefs)
    plan: list[str] = field(default_factory=list)
    history: list[Step] = field(default_factory=list)
    consecutive_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def observe(self, step: Step, info: dict[str, Any]) -> None:
        self.history.append(step)
        self.obs, self.info = step.observation, info
        self.consecutive_failures = 0 if step.ok else self.consecutive_failures + 1

    def hints_text(self) -> str:
        return "\n".join(f"- {h.exp.text} (trust {h.trust:.2f} ± {h.uncertainty:.2f})" for h in self.hints) or "- (none)"


def init_beliefs(task: Task, hints: list[Retrieved], rng: random.Random) -> Beliefs:
    b = Beliefs()
    order = [l for l in LOCATIONS if l != task.target]
    rng.shuffle(order)
    b.search_order = order
    tools = list(KINDS[task.kind][1])
    rng.shuffle(tools)
    b.tool_order = tools
    for h in hints:
        key = h.exp.key
        if h.exp.kind == "location" and key.get("obj") == task.obj:
            b.obj_guess, b.obj_hint = key["loc"], h.exp.id
        elif h.exp.kind == "tool" and key.get("task") == task.kind and key.get("tool") in tools:
            b.tool_pref, b.tool_hint = key["tool"], h.exp.id
    return b
