"""Context-conditional trust for experiences (research content 1).

Global posterior (does this experience tend to help anywhere?):
    a = prior_a + successes + support_weight * (support - 1)
    b = prior_b + failures  + contradiction_weight * contradictions
    m = a / (a + b)

Local posterior for the query context, shrunk toward the global mean:
    a' = m * shrink + confirmations_ctx
    b' = (1 - m) * shrink + failures_ctx + contradiction_weight * contradictions_ctx

trust = 0.5 + (mean - 0.5) * exp(-decay * episodes_since_last_validation)

Effects:
- In an unseen context the global mean acts as the prior, so useful experience transfers.
- A contradiction in context B lowers trust in B strongly, in other contexts only slightly
  (negative-transfer suppression without forgetting what still holds in A).
- Unvalidated experience drifts back to an uninformed 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from evoplan.memory.experience import Experience


@dataclass(frozen=True)
class TrustConfig:
    prior_a: float = 2.0
    prior_b: float = 1.0
    support_weight: float = 0.5
    contradiction_weight: float = 3.0
    shrink: float = 2.0
    decay: float = 0.01
    quarantine_below: float = 0.35
    contextual: bool = True       # False: global trust only (ablation)


def global_mean(exp: Experience, cfg: TrustConfig) -> float:
    a = cfg.prior_a + exp.successes + cfg.support_weight * (exp.support - 1)
    b = cfg.prior_b + exp.failures + cfg.contradiction_weight * exp.contradictions
    return a / (a + b)


def trust_score(exp: Experience, now: int, cfg: TrustConfig = TrustConfig(), ctx: str | None = None) -> float:
    mean = global_mean(exp, cfg)
    if cfg.contextual and ctx is not None and ctx in exp.by_context:
        conf, fail, contra = exp.by_context[ctx]
        a = mean * cfg.shrink + conf
        b = (1 - mean) * cfg.shrink + fail + cfg.contradiction_weight * contra
        mean = a / (a + b)
    recency = math.exp(-cfg.decay * max(0, now - exp.last_validated))
    return 0.5 + (mean - 0.5) * recency
