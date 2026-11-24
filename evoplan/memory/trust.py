"""Context-conditional trust for experiences (research content 1).

Every piece of evidence is kept in the experience's log with the episode it came from
(see ``Experience.evidence``). Trust is a Beta posterior over that evidence:

    for   = successes + support_weight * merges           (evidence the experience helps)
    against = failures + contradiction_weight * contradictions
    global:  a = prior_a + for,   b = prior_b + against,   m = a / (a + b)
    local, for the query context, shrunk toward the global mean:
             a' = m * shrink + confirmations_ctx
             b' = (1 - m) * shrink + failures_ctx + contradiction_weight * contradictions_ctx
    trust = a' / (a' + b'),   uncertainty = standard deviation of Beta(a', b')

How time is handled (``decay_mode``):

- ``"evidence"`` (default, the proposed method): each piece of evidence is weighted by
  gamma ** (episodes since it was observed). Old evidence counts for less, so the mean moves
  only slowly toward the prior while the uncertainty grows. "Not seen for a while" means
  "less sure", not "wrong" (a good experience is not starved) and not "unknown" (a refuted
  experience does not drift back to 0.5). High uncertainty is the signal for verification.
- ``"to_half"`` (ablation): the undiscounted mean is pulled toward 0.5 by gamma ** dt,
  dt = episodes since the latest evidence. Loses how much evidence there was.
- ``"to_zero"`` (ablation): the undiscounted mean is multiplied by gamma ** dt.
  Unused experience is treated as untrustworthy and eventually filtered out.

Quarantine uses the undiscounted global mean, so an experience the environment refuted
does not come back just because time passed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from evoplan.memory.experience import Experience

DECAY_MODES = ("evidence", "to_half", "to_zero")


@dataclass(frozen=True)
class TrustConfig:
    prior_a: float = 1.0          # uninformed Beta(1, 1): an untested experience starts at 0.5
    prior_b: float = 1.0
    support_weight: float = 0.5   # weight of an independent re-derivation (merge)
    contradiction_weight: float = 3.0
    shrink: float = 2.0           # pseudo-counts of the global mean in the local posterior
    gamma: float = 0.99           # per-episode discount of old evidence
    decay_mode: str = "evidence"  # one of DECAY_MODES
    quarantine_below: float = 0.35
    contextual: bool = True       # False: global trust only (ablation)

    def __post_init__(self) -> None:
        if self.decay_mode not in DECAY_MODES:
            raise ValueError(f"decay_mode must be one of {DECAY_MODES}, got {self.decay_mode!r}")


@dataclass(frozen=True)
class TrustEstimate:
    mean: float           # how much the experience is trusted in this context
    uncertainty: float    # how sure we are about that (std of the Beta posterior)


def _posterior(exp: Experience, cfg: TrustConfig, ctx: str | None, now: int | None) -> tuple[float, float]:
    """Beta(a, b) for ``ctx``. ``now=None`` weights all evidence equally (no discount)."""
    pos = neg = 0.0                      # global evidence
    conf = fail = contra = 0.0           # local evidence in ctx
    local = cfg.contextual and ctx is not None
    for t, e_ctx, kind in exp.evidence:
        w = 1.0 if now is None else cfg.gamma ** max(0, now - t)
        if kind == "success":
            pos += w
        elif kind == "merge":
            pos += cfg.support_weight * w
        elif kind == "failure":
            neg += w
        elif kind == "contradiction":
            neg += cfg.contradiction_weight * w
        if local and e_ctx == ctx:
            if kind in ("create", "merge", "success"):
                conf += w
            elif kind == "failure":
                fail += w
            elif kind == "contradiction":
                contra += w
    a, b = cfg.prior_a + pos, cfg.prior_b + neg
    if local and ctx in exp.by_context:
        m = a / (a + b)
        a = m * cfg.shrink + conf
        b = (1 - m) * cfg.shrink + fail + cfg.contradiction_weight * contra
    return a, b


def _beta_std(a: float, b: float) -> float:
    return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


def global_mean(exp: Experience, cfg: TrustConfig) -> float:
    """Undiscounted global mean, used for quarantine."""
    a, b = _posterior(exp, replace(cfg, contextual=False), None, None)
    return a / (a + b)


def trust_estimate(exp: Experience, now: int, cfg: TrustConfig = TrustConfig(),
                   ctx: str | None = None) -> TrustEstimate:
    if cfg.decay_mode == "evidence":
        a, b = _posterior(exp, cfg, ctx, now)
        return TrustEstimate(a / (a + b), _beta_std(a, b))
    a, b = _posterior(exp, cfg, ctx, None)
    mean, g = a / (a + b), cfg.gamma ** max(0, now - exp.last_evidence)
    mean = 0.5 + (mean - 0.5) * g if cfg.decay_mode == "to_half" else mean * g
    return TrustEstimate(mean, _beta_std(a, b))


def trust_score(exp: Experience, now: int, cfg: TrustConfig = TrustConfig(), ctx: str | None = None) -> float:
    return trust_estimate(exp, now, cfg, ctx).mean
