"""Environment interface. Real benchmarks (ALFWorld, ScienceWorld, ...) plug in here."""

from __future__ import annotations

from typing import Any, Protocol

from evoplan.types import Task


class Env(Protocol):
    max_steps: int

    def reset(self, task: Task) -> tuple[str, dict[str, Any]]:
        """Start a task. Returns (observation text, structured info)."""

    def step(self, action: str) -> tuple[str, bool, bool, dict[str, Any]]:
        """Run one action. Returns (observation, action_ok, task_done, info).

        ``info`` must contain ``event``: None, or one of "transient" (temporary failure,
        retry may work), "broken" (resource unusable, the plan must change), "invalid".
        """

    def admissible_actions(self) -> list[str]:
        """Actions that are syntactically valid in the current state."""
