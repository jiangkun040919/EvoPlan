"""Experience records: what the agent learned, where it came from, and how it has performed since.

An experience is a *fact or insight* (e.g. "the apple is in the fridge"). Whether it holds
can depend on the context (domain), so evidence is kept both globally and per context.

The counters are a readable summary. The evidence log is the raw record trust is computed
from: one entry per observation, with the episode index, the context and the outcome, so old
evidence can be discounted and nothing is lost when an experience is merged or refuted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# kinds of evidence entries
EVIDENCE_KINDS = ("create", "merge", "success", "failure", "contradiction")


@dataclass
class Experience:
    id: str
    text: str                     # natural-language insight, injected into LLM prompts
    kind: str                     # "location" | "tool" | "insight" (free text from LLM reflection)
    key: dict[str, str]           # structured hint, e.g. {"obj": "apple", "loc": "fridge"}
    context: str                  # contexts it was derived in, used for similarity
    created: int                  # episode index when first extracted
    last_evidence: int            # episode index of the latest evidence of any kind
    support: int = 1              # how many episodes independently produced this experience
    uses: int = 0
    successes: int = 0            # episodes that used it and succeeded
    failures: int = 0             # episodes that used it and failed
    contradictions: int = 0       # times the environment directly contradicted it
    quarantined: bool = False     # globally unreliable (low trust without any context)
    # per-context evidence: context -> [confirmations, failures, contradictions]
    by_context: dict[str, list[int]] = field(default_factory=dict)
    # raw evidence log: [episode index, context or None, kind in EVIDENCE_KINDS]
    evidence: list[list] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def signature(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        return self.kind, tuple(sorted(self.key.items()))

    def local(self, ctx: str) -> list[int]:
        return self.by_context.setdefault(ctx, [0, 0, 0])

    def log(self, now: int, ctx: str | None, kind: str) -> None:
        assert kind in EVIDENCE_KINDS, kind
        self.evidence.append([now, ctx, kind])
        self.last_evidence = max(self.last_evidence, now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        return cls(**data)
