"""Turn finished episodes into experiences.

RuleReflector reads structured env info (offline, deterministic, used with ToyKitchen).
LLMReflector asks the model for insights (used with real benchmarks).
"""

from __future__ import annotations

import re
from typing import Protocol

from evoplan.llm.base import LLM
from evoplan.memory.experience import Experience
from evoplan.types import Episode


class Reflector(Protocol):
    def reflect(self, episode: Episode) -> list[Experience]: ...


class RuleReflector:
    def reflect(self, episode: Episode) -> list[Experience]:
        t = episode.task
        out: list[Experience] = []
        loc = None
        for step in episode.steps:
            verb, *args = step.action.split()
            if verb == "go" and step.ok:
                loc = args[0]
            elif verb == "take" and step.ok and args and args[0] == t.obj and loc:
                out.append(self._exp(episode, "location", {"obj": t.obj, "loc": loc},
                                     f"The {t.obj} can be found at the {loc}.",
                                     f"domain={t.domain} obj={t.obj}"))
            elif verb == "use" and step.ok and loc:
                out.append(self._exp(episode, "tool", {"task": t.kind, "tool": loc},
                                     f"To {t.kind} an object, use the {loc}.",
                                     f"domain={t.domain} kind={t.kind}"))
        return out

    @staticmethod
    def _exp(ep: Episode, kind: str, key: dict[str, str], text: str, context: str) -> Experience:
        sig = "-".join(f"{k}={v}" for k, v in sorted(key.items()))
        return Experience(id=f"{kind}:{sig}", text=text, kind=kind,
                          key=key, context=context, created=ep.index,
                          last_evidence=ep.index, sources=[ep.task.id])


class LLMReflector:
    PROMPT = (
        "You are reviewing an agent's attempt at a task.\n"
        "Task: {goal}\nOutcome: {outcome}\nTrajectory:\n{traj}\n\n"
        "Write at most 3 short, reusable insights that would help on similar future tasks. "
        "Only state what the trajectory shows. One per line, each starting with 'INSIGHT:'."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def reflect(self, episode: Episode) -> list[Experience]:
        t = episode.task
        traj = "\n".join(f"> {s.action}\n{s.observation}" for s in episode.steps[-30:])
        reply = self.llm.chat([{"role": "user", "content": self.PROMPT.format(
            goal=t.goal, outcome="success" if episode.success else "failure", traj=traj)}])
        episode.prompt_tokens += reply.prompt_tokens
        episode.completion_tokens += reply.completion_tokens
        out = []
        for i, m in enumerate(re.finditer(r"INSIGHT:\s*(.+)", reply.text)):
            text = m.group(1).strip()
            out.append(Experience(id=f"insight:{episode.index}:{i}", text=text, kind="insight",
                                  key={"text": text.lower()[:80]}, context=t.context(),
                                  created=episode.index, last_evidence=episode.index, sources=[t.id]))
        return out
