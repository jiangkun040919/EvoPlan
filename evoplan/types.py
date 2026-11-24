"""Core data types shared by environments, agents, memory and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """One task in a continual task stream.

    ``domain`` identifies the environment instance (e.g. a house layout). Agents may
    see it as part of the task context, but they never see the hidden rules of a domain.
    """

    id: str
    kind: str
    obj: str
    target: str
    domain: str
    goal: str
    layout: str = ""   # hidden environment layout; differs from `domain` during drift phases

    def context(self) -> str:
        """Short text used for experience retrieval."""
        return f"domain={self.domain} kind={self.kind} obj={self.obj}"


@dataclass
class Step:
    action: str
    observation: str
    ok: bool
    event: str | None = None


@dataclass
class Episode:
    task: Task
    method: str
    seed: int
    index: int
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    used_experiences: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    replans: int = 0
    hint_valid: list[bool | None] = field(default_factory=list)  # evaluation only, never seen by the agent
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_chars: int = 0

    @property
    def failures(self) -> int:
        return sum(1 for s in self.steps if not s.ok)

    @property
    def had_disruption(self) -> bool:
        return any(s.event in {"broken", "transient", "missing"} for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_steps"] = len(self.steps)
        data["failures"] = self.failures
        data["had_disruption"] = self.had_disruption
        return data
