"""Continual task streams: ordered task sequences with distribution shift between phases."""

from __future__ import annotations

import random
from dataclasses import dataclass

from evoplan.envs.toy_kitchen import KINDS, OBJECTS, TARGETS, make_task
from evoplan.types import Task


@dataclass(frozen=True)
class Phase:
    domain: str          # label the agent sees
    n_tasks: int
    layout: str = ""     # layout the environment actually uses ("" = same as domain)

    @property
    def label(self) -> str:
        return self.domain if not self.layout or self.layout == self.domain else f"{self.domain}@{self.layout}"


def parse_schedule(text: str) -> list[Phase]:
    """Phases as DOMAIN[@LAYOUT]:N.

    "A:40,B:40,A:20" switches to a new domain and back (cross-domain transfer, stability).
    "A@B:40" keeps the label A but silently uses layout B (drift inside a known context).
    """
    phases = []
    for chunk in text.split(","):
        name, n = chunk.strip().split(":")
        domain, _, layout = name.strip().upper().partition("@")
        phases.append(Phase(domain, int(n), layout or domain))
    return phases


def make_stream(schedule: list[Phase], seed: int) -> list[tuple[int, Task]]:
    """Returns (phase index, task) pairs. The same seed yields the same stream for every method."""
    rng = random.Random(seed)
    stream: list[tuple[int, Task]] = []
    for p_idx, phase in enumerate(schedule):
        for i in range(phase.n_tasks):
            kind = rng.choice(list(KINDS))
            obj = rng.choice(OBJECTS)
            target = rng.choice(TARGETS)
            stream.append((p_idx, make_task(f"s{seed}-p{p_idx}-{i:03d}", kind, obj, target,
                                                 phase.domain, phase.layout or phase.domain)))
    return stream
