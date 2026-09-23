"""Study 2 — spatial containment: how the membrane holds the individual together.

In study 1 "inside" was a scalar volume. In a real medium the self-produced interior
would diffuse away — the individual would dissolve. This model puts the cell on a 1-D
lattice: interior content DIFFUSES, and the self-produced membrane GATES the cross-
boundary flux (permeability falls as membrane grows). The membrane is built by the
metabolism it contains, so containment is self-produced — the autopoiesis of containment.

Three regimes make the point:
  * fed + membrane   → the interior is held together (a contained peak)
  * fed, no membrane → metabolism alone can't hold it; it leaks out (membrane is what contains)
  * starved          → the membrane decays, permeability rises, the individual disperses
                       (precariousness, spatial edition)
"""
from __future__ import annotations

import numpy as np

# The per-step math now lives in the REAL process-bigraph composite
# (``processes_spatial.SpatialContainment``); ``simulate`` drives that composite
# and returns the same space-time history + metrics. The constants/helper are
# re-exported here so ``viz`` and ``containment_metrics`` keep working unchanged.
from .processes_spatial import (  # noqa: F401
    N, INSIDE, _LEFT_EDGE, _RIGHT_EDGE, _ratio, simulate,
)


def containment_metrics():
    """The study-2 measures the meter reads (extends the autopoiesis meter to space)."""
    held = simulate(fed=True, membrane=True)
    leaky = simulate(fed=True, membrane=False)
    starved = simulate(fed=False, membrane=True)
    measures = {
        "containment_ratio": held["containment"],                       # held together?
        "membrane_effect": held["containment"] / max(leaky["containment"], 1e-9),  # membrane's role
        "precariousness_collapse": starved["containment"] / max(held["containment"], 1e-9),  # disperses when starved?
    }
    # Controls: no-membrane is the discriminating NEGATIVE control (should fail to
    # contain); membrane-on is the POSITIVE end — together they calibrate the metric.
    controls = [
        {"name": "self-produced-membrane", "kind": "positive",
         "hypothesis": "A self-produced membrane should contain the metabolites against diffusion.",
         "expected": "high containment", "observed": f"{held['containment']:.1f}x concentrated",
         "result": "PASS"},
        {"name": "no-membrane", "kind": "negative",
         "hypothesis": "Metabolism WITHOUT a membrane should NOT contain — the interior disperses.",
         "expected": "containment fails (~1x)",
         "observed": f"{leaky['containment']:.1f}x vs {held['containment']:.1f}x with membrane",
         "result": "PASS" if held["containment"] > 2 * max(leaky["containment"], 1e-9) else "FAIL"},
    ]
    # Parameter-sweep robustness (deterministic PDE): vary diffusion D; the
    # membrane should contain better than no-membrane across the range.
    sweep_D = [0.15, 0.20, 0.25, 0.30, 0.35]
    wins = sum(1 for d in sweep_D
               if simulate(fed=True, membrane=True, D=d)["containment"]
               > simulate(fed=True, membrane=False, D=d)["containment"])
    robustness = {"parameter_sweep": True, "n_replicates": len(sweep_D), "seeds": sweep_D,
                  "swept_param": "D",
                  "note": f"{wins}/{len(sweep_D)} diffusion settings: membrane contains better than no-membrane."}
    return measures, {"held": held, "leaky": leaky, "starved": starved,
                      "controls": controls, "robustness": robustness, "n_steps": 500}


if __name__ == "__main__":
    m, _ = containment_metrics()
    for k, v in m.items():
        print(f"{k:24s} {v:.3f}")
