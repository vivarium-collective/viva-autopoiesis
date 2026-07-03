"""Study 4 — growth & division as a REAL process-bigraph composite.

The numpy lineage model (``growth.py``) ported VERBATIM into a single
``process_bigraph.Process``. The variable-length population lives in one
``tree`` store (a list of cell dicts) the process REPLACES each tick; the
running ``deaths`` / ``divisions`` counters are scalar ``float`` stores
(additive deltas accumulate the totals), which doubles as the emitter's
persisted trajectory alongside ``pop_size``.

Determinism: numpy's setup (the single founder) draws no randomness, so the
founder is placed deterministically in the state builder and the process owns
``default_rng(seed)`` for the per-step division/partition/mutation/cull draws —
the composite trajectory is bit-identical to the numpy reference.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite, Process

M_DIV = 20.0      # membrane size that triggers division
M_MIN = 5.0       # below this a (daughter) cell cannot maintain itself -> dissolves
FOUNDER_THETA = 0.5


class GrowthDivision(Process):
    """One grow-and-divide step over the whole population, ported verbatim from
    ``growth.simulate``'s loop body (including the carrying-capacity cull)."""

    config_schema = {
        "seed": {"_type": "integer", "_default": 0},
        "supply": {"_type": "float", "_default": 0.55},
        # ``starved`` forces supply=0 (the no-self-maintenance negative control).
        # A boolean toggle rather than ``supply=0.0`` because a float config of
        # 0.0 is falsy and bigraph-schema substitutes the default (0.55) for it —
        # so the control must be expressed as a flag the engine carries faithfully.
        "starved": {"_type": "boolean", "_default": False},
        "decay": {"_type": "float", "_default": 0.018},
        "split_sigma": {"_type": "float", "_default": 0.15},
        "mut": {"_type": "float", "_default": 0.03},
        "carrying": {"_type": "integer", "_default": 300},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.rng = np.random.default_rng(self.config["seed"])

    def inputs(self):
        return {"population": "tree", "deaths": "float", "divisions": "float"}

    def outputs(self):
        return {"population": "tree", "deaths": "float", "divisions": "float"}

    def update(self, state, interval):
        c = self.config
        supply = 0.0 if c["starved"] else c["supply"]
        pop = state["population"]
        nxt = []
        d_deaths = 0
        d_divisions = 0
        for cell in pop:
            cell = {"M": cell["M"], "theta": cell["theta"]}
            cell["M"] += supply - c["decay"] * cell["M"]   # grow
            if cell["M"] >= M_DIV:                                # divide
                d_divisions += 1
                fM = float(np.clip(self.rng.normal(0.5, c["split_sigma"]), 0.12, 0.88))
                for frac in (fM, 1.0 - fM):
                    dM = cell["M"] * frac
                    if dM >= M_MIN:
                        nxt.append({"M": dM,
                                    "theta": cell["theta"] + float(self.rng.normal(0, c["mut"]))})
                    else:
                        d_deaths += 1                            # daughter too small -> dissolves
            else:
                nxt.append(cell)
        if len(nxt) > c["carrying"]:                             # cap (finite environment)
            keep = self.rng.choice(len(nxt), c["carrying"], replace=False)
            nxt = [nxt[i] for i in keep]
        return {"population": nxt, "deaths": float(d_deaths), "divisions": float(d_divisions)}


def growth_state(*, supply=0.55, decay=0.018, split_sigma=0.15, mut=0.03,
                 carrying=300, seed=0, starved=False):
    # supply=0 IS the no-self-maintenance control, but a 0.0 float config collapses
    # to the default — carry it as the ``starved`` flag instead.
    starved = bool(starved or supply == 0.0)
    return {
        "population": [{"M": 8.0, "theta": FOUNDER_THETA}],   # one founder
        "deaths": 0.0,
        "divisions": 0.0,
        "growth": {
            "_type": "process",
            "address": "local:GrowthDivision",
            "config": {"seed": int(seed), "supply": supply, "starved": starved,
                       "decay": decay, "split_sigma": split_sigma, "mut": mut,
                       "carrying": int(carrying)},
            "interval": 1.0,
            "inputs": {"population": ["population"], "deaths": ["deaths"],
                       "divisions": ["divisions"]},
            "outputs": {"population": ["population"], "deaths": ["deaths"],
                        "divisions": ["divisions"]},
        },
    }


def _maybe_emitter(state, run_id, db_file):
    if run_id is None or db_file is None:
        return state, None
    emit = {"deaths": "float", "divisions": "float", "global_time": "float"}
    try:
        from vivarium_workbench.lib.composite_runs import inject_sqlite_emitter
        state = inject_sqlite_emitter(state, run_id=run_id, db_file=db_file)
        node = state["sqlite_emitter"]
        node["config"]["emit"] = dict(emit)
        node["inputs"] = {k: [k] for k in emit}
        return state, "sqlite"
    except Exception:  # pragma: no cover
        from pathlib import Path as _P
        db = _P(db_file)
        state = dict(state)
        state["sqlite_emitter"] = {
            "_type": "step", "address": "local:SQLiteEmitter",
            "config": {"emit": dict(emit), "file_path": str(db.parent),
                       "db_file": db.name, "simulation_id": run_id},
            "inputs": {k: [k] for k in emit},
        }
        return state, "sqlite"


def simulate(steps=600, *, supply=0.55, decay=0.018, split_sigma=0.15, mut=0.03,
             carrying=300, seed=0, starved=False, run_id=None, db_file=None):
    """Run the growth-division composite, recording the same lineage history +
    metrics ``growth.simulate`` returned (bit-identical numbers). ``supply=0.0``
    (or ``starved=True``) is the no-self-maintenance control."""
    state = growth_state(supply=supply, decay=decay, split_sigma=split_sigma,
                         mut=mut, carrying=carrying, seed=seed, starved=starved)
    state, _label = _maybe_emitter(state, run_id, db_file)
    from .core import build_core
    composite = Composite({"schema": {"population": "tree"}, "state": state},
                          core=build_core())

    pop_size = [len(composite.state["population"])]
    hetero = [0.0]
    snapshots = []
    for t in range(steps):
        composite.run(1.0)
        pop = composite.state["population"]
        thetas = np.array([c["theta"] for c in pop])
        pop_size.append(len(pop))
        hetero.append(float(thetas.std()) if len(pop) > 1 else 0.0)
        if t in (0, steps // 3, 2 * steps // 3, steps - 1):
            snapshots.append((t, thetas.copy()))

    pop = composite.state["population"]
    return {
        "pop_size": pop_size, "hetero": hetero,
        "deaths": int(composite.state["deaths"]),
        "divisions": int(composite.state["divisions"]),
        "final_pop": len(pop),
        "final_theta": np.array([c["theta"] for c in pop]),
        "snapshots": snapshots,
    }
