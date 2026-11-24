"""Experience store with retrieval and post-episode trust updates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from evoplan.memory.experience import Experience
from evoplan.memory.trust import TrustConfig, TrustEstimate, global_mean, trust_estimate

_TOKEN = re.compile(r"[a-z0-9_=]+")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Jaccard similarity over tokens. Swap for embeddings when moving to real benchmarks."""
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


@dataclass
class Retrieved:
    exp: Experience
    sim: float
    trust: float
    uncertainty: float
    score: float


class ExperienceStore:
    def __init__(self, trust_cfg: TrustConfig = TrustConfig()) -> None:
        self.trust_cfg = trust_cfg
        self._items: dict[str, Experience] = {}
        self._by_sig: dict[tuple, str] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def get(self, exp_id: str) -> Experience:
        return self._items[exp_id]

    def estimate(self, exp: Experience, now: int, ctx: str | None = None) -> TrustEstimate:
        return trust_estimate(exp, now, self.trust_cfg, ctx)

    def trust(self, exp: Experience, now: int, ctx: str | None = None) -> float:
        return self.estimate(exp, now, ctx).mean

    # ----------------------------------------------------------------- writing
    def add(self, exp: Experience, ctx: str | None = None) -> Experience:
        """Insert, or merge into an existing experience with the same signature.

        Creating or re-deriving an experience in context ``ctx`` counts as a local confirmation
        there, and is written to the experience's evidence log.
        """
        sig = exp.signature()
        if sig in self._by_sig:
            target = self._items[self._by_sig[sig]]
            target.support += 1
            target.log(exp.last_evidence, ctx, "merge")
            target.sources = (target.sources + exp.sources)[-20:]
            if exp.context not in target.context:
                target.context = f"{target.context} | {exp.context}"[-400:]
        else:
            target = exp
            self._items[exp.id] = exp
            self._by_sig[sig] = exp.id
            target.log(exp.created, ctx, "create")
        if ctx is not None:
            target.local(ctx)[0] += 1
        target.quarantined = global_mean(target, self.trust_cfg) < self.trust_cfg.quarantine_below
        return target

    def record_outcome(self, used: list[str], contradicted: list[str], success: bool, now: int,
                       ctx: str | None = None) -> None:
        """Attribute an episode outcome to the experiences it used."""
        contra = set(contradicted)
        for exp_id in set(used) | contra:
            exp = self._items.get(exp_id)
            if exp is None:
                continue
            loc = exp.local(ctx) if ctx is not None else [0, 0, 0]
            if exp_id in used:
                exp.uses += 1
            if exp_id in contra:
                exp.contradictions += 1
                loc[2] += 1
                exp.log(now, ctx, "contradiction")
            elif success:
                exp.successes += 1
                loc[0] += 1
                exp.log(now, ctx, "success")
            else:
                exp.failures += 1
                loc[1] += 1
                exp.log(now, ctx, "failure")
            exp.quarantined = global_mean(exp, self.trust_cfg) < self.trust_cfg.quarantine_below

    # ----------------------------------------------------------------- reading
    def retrieve(self, context: str, now: int, k: int = 3, use_trust: bool = True, gate: bool = True,
                 ctx: str | None = None, min_sim: float = 0.2, min_trust: float = 0.55) -> list[Retrieved]:
        """Rank experiences for a task.

        use_trust=False, gate=False reproduces naive similarity retrieval (baseline).
        gate=True skips experiences below the similarity/trust thresholds for this context
        and globally quarantined ones (research content 2: negative-transfer suppression).
        """
        out: list[Retrieved] = []
        for exp in self._items.values():
            sim = similarity(context, exp.context)
            est = self.estimate(exp, now, ctx)
            if gate and (exp.quarantined or sim < min_sim or est.mean < min_trust):
                continue
            score = sim * est.mean if use_trust else sim
            out.append(Retrieved(exp, sim, est.mean, est.uncertainty, score))
        out.sort(key=lambda r: r.score, reverse=True)
        return _one_per_slot(out)[:k]

    # ----------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for exp in self._items.values():
                f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path, trust_cfg: TrustConfig = TrustConfig()) -> "ExperienceStore":
        store = cls(trust_cfg)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                exp = Experience.from_dict(json.loads(line))
                store._items[exp.id] = exp
                store._by_sig[exp.signature()] = exp.id
        return store


def _one_per_slot(items: list[Retrieved]) -> list[Retrieved]:
    """Keep only the best-ranked hint per slot (one location per object, one tool per task kind)."""
    seen: set[tuple[str, str]] = set()
    out = []
    for r in items:
        slot = (r.exp.kind, r.exp.key.get("obj") or r.exp.key.get("task") or r.exp.id)
        if slot in seen:
            continue
        seen.add(slot)
        out.append(r)
    return out
