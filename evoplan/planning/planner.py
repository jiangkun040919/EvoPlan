"""Planners: build an initial plan from the task and retrieved experience, and revise it on triggers.

replan() returns the ids of experiences the trigger contradicted, so the store can
penalise them in this context.
"""

from __future__ import annotations

import re

from evoplan.envs.toy_kitchen import KINDS
from evoplan.llm.base import LLM
from evoplan.planning.context import AgentContext
from evoplan.planning.monitor import Trigger


class RulePlanner:
    """Deterministic planner for ToyKitchen. Keeps beliefs and a readable plan in sync."""

    def plan(self, ctx: AgentContext) -> None:
        t, b = ctx.task, ctx.beliefs
        find = f"find the {t.obj}" + (f" (memory: {b.obj_guess})" if b.obj_guess else " (search)")
        steps = [find]
        if KINDS[t.kind][0]:
            tool = self.choose_tool(ctx)
            steps.append(f"{t.kind} it with the {tool}" if tool else f"{t.kind} it (no usable tool known)")
        steps.append(f"put it on the {t.target}")
        ctx.plan = steps

    def replan(self, ctx: AgentContext, trig: Trigger) -> list[str]:
        contradicted = self.update_beliefs(ctx, trig)
        self.plan(ctx)
        return contradicted

    @staticmethod
    def update_beliefs(ctx: AgentContext, trig: Trigger) -> list[str]:
        b = ctx.beliefs
        contradicted: list[str] = []
        if trig.kind == "tool_broken":
            b.avoid_tools.add(trig.detail)
            if b.tool_pref == trig.detail:
                if b.tool_hint:
                    contradicted.append(b.tool_hint)
                b.tool_pref, b.tool_hint = None, None
        elif trig.kind == "hint_missing":
            b.searched.add(trig.detail)
            if b.obj_hint:
                contradicted.append(b.obj_hint)
            b.obj_guess, b.obj_hint = None, None
        elif trig.kind == "stalled":
            last = ctx.history[-1].action.split() if ctx.history else []
            if last[:1] == ["use"] and len(last) > 1:
                b.avoid_tools.add(last[1])
        ctx.hints = [h for h in ctx.hints if h.exp.id not in contradicted]
        return contradicted

    @staticmethod
    def choose_tool(ctx: AgentContext) -> str | None:
        b = ctx.beliefs
        if b.tool_pref and b.tool_pref not in b.avoid_tools:
            return b.tool_pref
        return next((x for x in b.tool_order if x not in b.avoid_tools), None)


class LLMPlanner(RulePlanner):
    """Asks the LLM for a short plan; keeps RulePlanner's belief updates for experience bookkeeping."""

    PROMPT = (
        "Goal: {goal}\nCurrent observation: {obs}\nRelevant experience:\n{hints}\n{extra}"
        "Write a short numbered plan (at most 5 steps). Do not follow experience that "
        "contradicts the current observation."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def plan(self, ctx: AgentContext, extra: str = "") -> None:
        reply = self.llm.chat([{"role": "user", "content": self.PROMPT.format(
            goal=ctx.task.goal, obs=ctx.obs, hints=ctx.hints_text(), extra=extra)}], max_tokens=200)
        ctx.prompt_tokens += reply.prompt_tokens
        ctx.completion_tokens += reply.completion_tokens
        ctx.plan = [m.group(1).strip() for m in re.finditer(r"^\s*\d+[.)]\s*(.+)$", reply.text, re.M)] or [reply.text]

    def replan(self, ctx: AgentContext, trig: Trigger) -> list[str]:
        contradicted = self.update_beliefs(ctx, trig)
        self.plan(ctx, extra=f"The previous plan failed ({trig.kind}: {trig.detail}). Revise it.\n")
        return contradicted
