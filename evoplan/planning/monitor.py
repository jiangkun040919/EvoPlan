"""State monitor (research content 3): detect when the current plan no longer fits the state.

Triggers:
- tool_broken  the environment reports a resource is unusable -> change the plan now
- hint_missing an experience-based expectation is contradicted (object not where memory said)
- stalled      repeated failures without progress
Transient failures are not triggers: retrying is the right response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evoplan.planning.context import AgentContext
from evoplan.types import Step


@dataclass(frozen=True)
class Trigger:
    kind: str
    detail: str = ""


class StateMonitor:
    def __init__(self, stall_after: int = 3) -> None:
        self.stall_after = stall_after

    def check(self, ctx: AgentContext, step: Step, info: dict[str, Any]) -> Trigger | None:
        b, t = ctx.beliefs, ctx.task
        if info.get("event") == "broken":
            return Trigger("tool_broken", info.get("loc", ""))
        if (b.obj_guess is not None and info.get("loc") == b.obj_guess and not info.get("closed")
                and info.get("holding") != t.obj and t.obj not in (info.get("visible") or [])):
            return Trigger("hint_missing", b.obj_guess)
        if ctx.consecutive_failures >= self.stall_after and info.get("event") != "transient":
            return Trigger("stalled", step.action)
        return None
