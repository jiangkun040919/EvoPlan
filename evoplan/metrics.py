"""Evaluation metrics (research content 4).

All comparisons against the no-experience baseline are paired: same seed, same task.

- success_rate         fraction of tasks solved
- avg_steps            mean actions per task (budget counted for failures)
- transfer_gain        success_rate(method) - success_rate(no_exp)
- step_saving          mean over paired tasks of steps(no_exp) - steps(method)
- negative_transfer    fraction of paired tasks the method failed but no_exp solved,
                       or solved with more than `margin` extra steps
- recovery_rate        among tasks with a disruption (fault, broken tool, wrong memory), fraction solved
- reliability          fraction of injected experiences that were actually true in that episode
                       (environment oracle; evaluation only)
- tokens / cost        LLM usage; context_chars is the injected-experience size (proxy when offline)
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from evoplan.types import Episode


def summarize(episodes: list[Episode], baseline: list[Episode] | None = None, budget: int = 18,
              margin: int = 2, price_per_1k: tuple[float, float] = (0.0, 0.0)) -> dict[str, float]:
    if not episodes:
        return {}
    steps = [len(e.steps) if e.success else budget for e in episodes]
    disrupted = [e for e in episodes if e.had_disruption]
    used = sum(len(e.used_experiences) for e in episodes)
    judged = [v for e in episodes for v in e.hint_valid if v is not None]
    p_tok = sum(e.prompt_tokens for e in episodes)
    c_tok = sum(e.completion_tokens for e in episodes)
    out = {
        "n": len(episodes),
        "success_rate": mean(e.success for e in episodes),
        "avg_steps": mean(steps),
        "recovery_rate": mean(e.success for e in disrupted) if disrupted else float("nan"),
        "reliability": mean(judged) if judged else float("nan"),
        "avg_hints": used / len(episodes),
        "replans": mean(e.replans for e in episodes),
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "cost": p_tok / 1000 * price_per_1k[0] + c_tok / 1000 * price_per_1k[1],
        "context_chars": mean(e.context_chars for e in episodes),
    }
    if baseline:
        base = {(e.seed, e.task.id): e for e in baseline}
        pairs = [(e, base[(e.seed, e.task.id)]) for e in episodes if (e.seed, e.task.id) in base]
        if pairs:
            def cost(ep: Episode) -> int:
                return len(ep.steps) if ep.success else budget
            out["transfer_gain"] = mean(m.success for m, _ in pairs) - mean(b.success for _, b in pairs)
            out["step_saving"] = mean(cost(b) - cost(m) for m, b in pairs)
            out["negative_transfer"] = mean(
                (b.success and not m.success) or (m.success and b.success and cost(m) > cost(b) + margin)
                for m, b in pairs)
    return out


def group(episodes: Iterable[Episode], key) -> dict:
    out: dict = defaultdict(list)
    for e in episodes:
        out[key(e)].append(e)
    return out
