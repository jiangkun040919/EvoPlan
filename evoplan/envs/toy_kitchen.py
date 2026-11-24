"""ToyKitchen: a small text environment for validating the EvoPlan pipeline offline.

It is NOT a benchmark for paper results. It exists so that every component
(planning, experience extraction, trust, transfer gating, recovery) can be run and
tested in seconds without an API key.

Design, chosen so that experience can both help and hurt:
- Each domain (house) has fixed "home" locations for objects. Knowing them saves
  search steps, but homes differ across domains -> stale experience causes negative transfer.
- Some appliances are broken in some domains (e.g. the microwave in domain B), so a
  tool preference learned elsewhere can fail.
- Containers (fridge, microwave, cabinet) must be opened before looking inside or using them.
- Faults: transient action failures, and objects relocated away from their home.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from evoplan.types import Task

LOCATIONS = ["counter", "table", "shelf", "sink", "stove", "cabinet", "fridge", "microwave"]
OPENABLE = {"fridge", "microwave", "cabinet"}
OBJECTS = ["apple", "potato", "mug", "plate", "egg", "lettuce"]
TARGETS = ["table", "shelf", "counter"]

# task kind -> (required object state, appliances that produce it)
KINDS: dict[str, tuple[str | None, list[str]]] = {
    "pick": (None, []),
    "heat": ("heated", ["microwave", "stove"]),
    "cool": ("cooled", ["fridge"]),
    "clean": ("cleaned", ["sink"]),
}
ADJECTIVE = {"pick": "", "heat": "heated ", "cool": "cooled ", "clean": "clean "}


@dataclass(frozen=True)
class DomainSpec:
    name: str
    homes: dict[str, str]
    broken: frozenset[str] = frozenset()


# Domains share part of their layout, so transferred experience is partly right and partly wrong.
DOMAINS: dict[str, DomainSpec] = {
    "A": DomainSpec("A", {"apple": "fridge", "potato": "cabinet", "mug": "cabinet",
                          "plate": "shelf", "egg": "fridge", "lettuce": "sink"}),
    # B: apple, egg, lettuce moved; microwave broken
    "B": DomainSpec("B", {"apple": "cabinet", "potato": "cabinet", "mug": "cabinet",
                          "plate": "shelf", "egg": "microwave", "lettuce": "fridge"},
                    broken=frozenset({"microwave"})),
    # C: potato, mug, plate moved; stove broken
    "C": DomainSpec("C", {"apple": "fridge", "potato": "sink", "mug": "fridge",
                          "plate": "cabinet", "egg": "fridge", "lettuce": "sink"},
                    broken=frozenset({"stove"})),
}


def make_task(task_id: str, kind: str, obj: str, target: str, domain: str, layout: str | None = None) -> Task:
    goal = f"put a {ADJECTIVE[kind]}{obj} on the {target}".replace("  ", " ")
    return Task(id=task_id, kind=kind, obj=obj, target=target, domain=domain, goal=goal, layout=layout or domain)


@dataclass
class _State:
    loc: str = "counter"
    holding: str | None = None
    opened: set[str] = field(default_factory=set)
    contents: dict[str, set[str]] = field(default_factory=dict)
    obj_state: dict[str, set[str]] = field(default_factory=dict)


class ToyKitchen:
    def __init__(self, seed: int = 0, fault_rate: float = 0.05, relocate_rate: float = 0.1,
                 max_steps: int = 18) -> None:
        self.rng = random.Random(seed)
        self.fault_rate = fault_rate
        self.relocate_rate = relocate_rate
        self.max_steps = max_steps
        self.task: Task | None = None
        self.spec: DomainSpec | None = None
        self.s = _State()

    # ----------------------------------------------------------------- public API
    def reset(self, task: Task) -> tuple[str, dict[str, Any]]:
        self.task = task
        self.spec = DOMAINS[task.layout or task.domain]
        self.s = _State(contents={loc: set() for loc in LOCATIONS})
        for obj, home in self.spec.homes.items():
            loc = home
            if obj == task.obj and self.rng.random() < self.relocate_rate:
                loc = self.rng.choice([l for l in LOCATIONS if l not in (home, task.target)])
            if obj == task.obj and loc == task.target:  # never start already solved
                loc = home if home != task.target else "counter"
            self.s.contents[loc].add(obj)
            self.s.obj_state[obj] = set()
        obs = f"Task: {task.goal}. " + self._describe_here()
        return obs, self._info(None)

    def step(self, action: str) -> tuple[str, bool, bool, dict[str, Any]]:
        assert self.task is not None and self.spec is not None, "call reset() first"
        parts = action.strip().lower().split()
        if not parts:
            return "Nothing happens.", False, False, self._info("invalid")
        if self.rng.random() < self.fault_rate and parts[0] in {"open", "take", "put", "use"}:
            return "Nothing happens. (The action failed, it may work if retried.)", False, False, self._info("transient")
        handler = getattr(self, f"_do_{parts[0]}", None)
        if handler is None:
            return f"Unknown action '{parts[0]}'.", False, False, self._info("invalid")
        obs, ok, event = handler(parts[1:])
        done = self._solved()
        return obs, ok, done, self._info(event)

    def admissible_actions(self) -> list[str]:
        s = self.s
        acts = [f"go {l}" for l in LOCATIONS if l != s.loc]
        if s.loc in OPENABLE and s.loc not in s.opened:
            acts.append(f"open {s.loc}")
        if self._visible() is not None:
            if s.holding is None:
                acts += [f"take {o}" for o in sorted(self._visible() or [])]
            else:
                acts.append(f"put {s.holding} {s.loc}")
                if any(s.loc in tools for _, tools in KINDS.values()):
                    acts.append(f"use {s.loc}")
        return acts

    def hint_valid(self, key: dict[str, str]) -> bool | None:
        """Evaluation oracle: is a structured hint true in the current episode? Never used by agents."""
        assert self.task is not None and self.spec is not None
        if "obj" in key and "loc" in key:
            return key["obj"] in self.s.contents.get(key["loc"], set()) or self.s.holding == key["obj"]
        if "task" in key and "tool" in key:
            return key["tool"] in KINDS[key["task"]][1] and key["tool"] not in self.spec.broken
        return None

    # ----------------------------------------------------------------- actions
    def _do_go(self, args: list[str]) -> tuple[str, bool, str | None]:
        if not args or args[0] not in LOCATIONS:
            return "You can't go there.", False, "invalid"
        self.s.loc = args[0]
        return self._describe_here(), True, None

    def _do_open(self, args: list[str]) -> tuple[str, bool, str | None]:
        loc = args[0] if args else self.s.loc
        if loc != self.s.loc or loc not in OPENABLE:
            return "You can't open that here.", False, "invalid"
        self.s.opened.add(loc)
        return f"You open the {loc}. " + self._describe_here(), True, None

    def _do_take(self, args: list[str]) -> tuple[str, bool, str | None]:
        if not args:
            return "Take what?", False, "invalid"
        obj, vis = args[0], self._visible()
        if vis is None:
            return f"The {self.s.loc} is closed.", False, "invalid"
        if self.s.holding is not None:
            return "Your hands are full.", False, "invalid"
        if obj not in vis:
            return f"You don't see a {obj} here.", False, "invalid"
        self.s.contents[self.s.loc].discard(obj)
        self.s.holding = obj
        return f"You take the {obj}.", True, None

    def _do_put(self, args: list[str]) -> tuple[str, bool, str | None]:
        obj = args[0] if args else self.s.holding
        if self.s.holding is None or obj != self.s.holding:
            return "You are not holding that.", False, "invalid"
        if self._visible() is None:
            return f"The {self.s.loc} is closed.", False, "invalid"
        self.s.contents[self.s.loc].add(obj)
        self.s.holding = None
        return f"You put the {obj} on the {self.s.loc}.", True, None

    def _do_use(self, args: list[str]) -> tuple[str, bool, str | None]:
        tool = args[0] if args else self.s.loc
        assert self.spec is not None
        if tool != self.s.loc:
            return f"You are not at the {tool}.", False, "invalid"
        effect = next((st for st, tools in KINDS.values() if tool in tools), None)
        if effect is None:
            return f"The {tool} can't be used.", False, "invalid"
        if self.s.holding is None:
            return "You are not holding anything.", False, "invalid"
        if tool in OPENABLE and tool not in self.s.opened:
            return f"The {tool} is closed.", False, "invalid"
        if tool in self.spec.broken:
            return f"The {tool} is broken and does nothing.", False, "broken"
        self.s.obj_state[self.s.holding].add(effect)
        return f"You use the {tool}. The {self.s.holding} is now {effect}.", True, None

    # ----------------------------------------------------------------- helpers
    def _visible(self) -> set[str] | None:
        loc = self.s.loc
        if loc in OPENABLE and loc not in self.s.opened:
            return None
        return self.s.contents[loc]

    def _describe_here(self) -> str:
        vis = self._visible()
        where = f"You are at the {self.s.loc}."
        if vis is None:
            return f"{where} The {self.s.loc} is closed."
        items = ", ".join(sorted(vis)) or "nothing"
        hold = f" You are holding the {self.s.holding}." if self.s.holding else ""
        return f"{where} You see: {items}.{hold}"

    def _solved(self) -> bool:
        assert self.task is not None
        need, _ = KINDS[self.task.kind]
        placed = self.task.obj in self.s.contents[self.task.target]
        return placed and (need is None or need in self.s.obj_state[self.task.obj])

    def _info(self, event: str | None) -> dict[str, Any]:
        vis = self._visible()
        held = self.s.holding
        return {
            "event": event,
            "loc": self.s.loc,
            "closed": vis is None,
            "visible": sorted(vis) if vis is not None else None,
            "holding": held,
            "obj_state": sorted(self.s.obj_state.get(self.task.obj, set())) if self.task else [],
        }
