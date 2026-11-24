"""Action policies.

HeuristicPolicy: deterministic stand-in for an LLM actor on ToyKitchen. It follows the
current beliefs (seeded by experience) and has a generic fallback: after 3 failed attempts
at the same action it gives that tool up. It lets the whole pipeline run offline.

LLMPolicy: ReAct-style actor for real runs; sees goal, plan, experience and recent history.
"""

from __future__ import annotations

import re

from evoplan.envs.toy_kitchen import KINDS
from evoplan.llm.base import LLM
from evoplan.planning.context import AgentContext
from evoplan.planning.planner import RulePlanner


class HeuristicPolicy:
    def act(self, ctx: AgentContext) -> str:
        t, b, s = ctx.task, ctx.beliefs, ctx.info
        if ctx.consecutive_failures >= 3 and ctx.history:  # generic fallback without a monitor
            last = ctx.history[-1].action.split()
            if last[0] == "use":
                b.avoid_tools.add(last[1])
        if ctx.consecutive_failures and ctx.history and ctx.history[-1].event == "transient":
            return ctx.history[-1].action  # retry
        here = s["loc"]

        if s["holding"] != t.obj:
            if s["closed"]:
                return f"open {here}"
            if t.obj in (s["visible"] or []):
                return f"take {t.obj}"
            b.searched.add(here)
            if b.obj_guess and b.obj_guess not in b.searched:
                return f"go {b.obj_guess}"
            nxt = next((l for l in b.search_order if l not in b.searched), None)
            return f"go {nxt}" if nxt else "go counter"

        need = KINDS[t.kind][0]
        if need and need not in s["obj_state"]:
            tool = RulePlanner.choose_tool(ctx)
            if tool is None:
                return f"go {t.target}"  # nothing usable left; give up gracefully
            if here != tool:
                return f"go {tool}"
            if s["closed"]:
                return f"open {tool}"
            return f"use {tool}"

        if here != t.target:
            return f"go {t.target}"
        if s["closed"]:
            return f"open {here}"
        return f"put {t.obj} {here}"


class LLMPolicy:
    SYSTEM = ("You are an agent acting in a text environment. Reply with one line: "
              "'ACTION: <action>' using exactly one of the admissible actions.")
    PROMPT = ("Goal: {goal}\nPlan:\n{plan}\nRelevant experience (may be wrong in this environment):\n{hints}\n"
              "Recent history:\n{history}\nCurrent observation: {obs}\nAdmissible actions: {actions}\n")

    def __init__(self, llm: LLM, admissible, history_window: int = 8) -> None:
        self.llm = llm
        self.admissible = admissible  # callable returning the env's admissible actions
        self.history_window = history_window

    def act(self, ctx: AgentContext) -> str:
        actions = self.admissible()
        history = "\n".join(f"> {s.action}\n{s.observation}" for s in ctx.history[-self.history_window:]) or "(start)"
        reply = self.llm.chat([
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": self.PROMPT.format(
                goal=ctx.task.goal, plan="\n".join(f"{i+1}. {p}" for i, p in enumerate(ctx.plan)),
                hints=ctx.hints_text(), history=history, obs=ctx.obs, actions="; ".join(actions))},
        ], max_tokens=40)
        ctx.prompt_tokens += reply.prompt_tokens
        ctx.completion_tokens += reply.completion_tokens
        return parse_action(reply.text, actions)


def parse_action(text: str, admissible: list[str]) -> str:
    m = re.search(r"ACTION:\s*(.+)", text)
    raw = (m.group(1) if m else text).strip().strip("`'\".").lower()
    if raw in admissible:
        return raw
    starts = [a for a in admissible if a.startswith(raw) or raw.startswith(a)]
    return starts[0] if starts else raw
