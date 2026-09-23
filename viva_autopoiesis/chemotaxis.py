"""Study 3 — adaptive chemotaxis: move toward food to survive.

The cell is placed in an environment with a nutrient gradient (a food source). It
senses the local nutrient and runs-and-tumbles — biasing motion up-gradient by
tumbling less when the sensed nutrient is rising (bacterial chemotaxis). The nutrient
feeds the metabolism that maintains the membrane that keeps it alive. A chemotactic
agent climbs the gradient, finds food, and survives; a blind agent random-walks,
starves, and dissolves.

This is where life becomes mind: the gradient is MEANINGFUL from the perspective of
the precarious identity — up-gradient is viable, down-gradient is dissolution. The
value (food = good) is grounded in the cell's own viability, not assigned from
outside. Agency is in service of self-maintenance (Di Paolo's adaptivity, Varela's
sense-making).
"""
from __future__ import annotations

import numpy as np

# The per-step run-and-tumble math now lives in the REAL process-bigraph composite
# (``processes_chemotaxis.Chemotaxis``); ``simulate`` drives that composite and
# returns the same trajectories + survival. The environment constants/helper are
# re-exported here so ``viz`` and ``chemotaxis_metrics`` keep working unchanged.
from .processes_chemotaxis import (  # noqa: F401
    L, X_FOOD, SIGMA, nutrient, simulate,
)


def chemotaxis_metrics(n_seeds=12):
    """The study-3 measures (the meter extended to agency / viability), replicated
    across ``n_seeds`` (chemotactic vs blind at matched seeds).

    Reported measures are the cross-seed MEAN; the context carries a ``robustness``
    summary (per-measure mean / std / range + the seed list) so the study is a
    distribution, not a single run, plus a representative seed-0 run for figures.
    The blind agent is a matched motile-but-non-sensing negative control."""
    per = {"chemotaxis_survival": [], "survival_advantage": [], "gradient_advantage": []}
    blind_survival = []
    chemo0 = blind0 = None
    for s in range(n_seeds):
        chemo = simulate(chemotactic=True, seed=s)
        blind = simulate(chemotactic=False, seed=s)
        if s == 0:
            chemo0, blind0 = chemo, blind
        per["chemotaxis_survival"].append(chemo["survival"])
        per["survival_advantage"].append(chemo["survival"] / max(blind["survival"], 1e-9))
        per["gradient_advantage"].append(chemo["mean_nutrient"] / max(blind["mean_nutrient"], 1e-9))
        blind_survival.append(blind["survival"])
    measures = {k: float(np.mean(v)) for k, v in per.items()}
    robustness = {
        "n_replicates": int(n_seeds),
        "seeds": list(range(n_seeds)),
        "parameter_sweep": False,
        "per_measure": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                            "min": float(np.min(v)), "max": float(np.max(v))}
                        for k, v in per.items()},
        "control_blind_survival_mean": float(np.mean(blind_survival)),
        "seeds_with_advantage": int(sum(1 for a in per["survival_advantage"] if a > 1.0)),
    }
    return measures, {"chemo": chemo0, "blind": blind0, "robustness": robustness, "n_steps": 900}


if __name__ == "__main__":
    m, _ = chemotaxis_metrics()
    for k, val in m.items():
        print(f"{k:22s} {val:.3f}")
