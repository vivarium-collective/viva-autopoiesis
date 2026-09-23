"""Study 2 — spatial containment as a REAL process-bigraph composite.

The numpy 1-D containment model (``spatial.py``) ported VERBATIM into a single
``process_bigraph.Process`` over composite stores:

  * ``concentration`` — an ``array`` store of N interior-content bins (additive
    deltas compose the metabolism production + the forward-Euler diffusion flux).
  * ``membrane`` — a scalar store; the self-produced boundary material.
  * ``containment`` — a derived scalar observable (interior/exterior ratio),
    written set-by-difference each tick so the SQLite emitter persists the
    containment trajectory.

Why one process (not the Metabolism/Membrane/Diffusion triple the loop uses):
the numpy step is genuinely SEQUENTIAL within a tick — the membrane grows from
the just-produced interior content, decays, sets the permeability, and the
permeability gates that same tick's diffusion. process-bigraph applies every
process's update against the START-of-tick state and SUMS the deltas (verified),
so a split into independent processes would lag the permeability by a tick and
drift from the calibrated numpy bands. Keeping the coupled step in one
``update`` makes the composite trajectory BIT-IDENTICAL to the numpy reference,
so study-2's measures land exactly on their authored pass_if bands.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite, Process

N = 80                       # lattice bins
INSIDE = slice(30, 50)       # the cell interior
_LEFT_EDGE, _RIGHT_EDGE = 29, 49   # the two edges crossing the membrane boundary


def _ratio(C, inside):
    return float(C[inside].mean() / (C[~inside].mean() + 1e-9))


class SpatialContainment(Process):
    """The 1-D containment step: interior production + membrane growth/decay +
    permeability-gated forward-Euler diffusion, ported verbatim from
    ``spatial.simulate``'s loop body. Reads/writes the whole lattice each tick."""

    config_schema = {
        "fed": {"_type": "boolean", "_default": True},
        "membrane_on": {"_type": "boolean", "_default": True},
        "D": {"_type": "float", "_default": 0.25},
        "g": {"_type": "float", "_default": 0.05},
        "k_inc": {"_type": "float", "_default": 0.015},
        "k_decay": {"_type": "float", "_default": 0.006},
        "k_perm": {"_type": "float", "_default": 0.4},
    }

    def inputs(self):
        return {"concentration": "array", "membrane": "float", "containment": "float"}

    def outputs(self):
        return {"concentration": "array", "membrane": "float", "containment": "float"}

    def update(self, state, interval):
        c = self.config
        C0 = np.asarray(state["concentration"], dtype=float)
        M0 = float(state["membrane"])
        inside = np.zeros(N, bool)
        inside[INSIDE] = True

        dC = np.zeros(N)
        Cf = C0.copy()
        Mg = M0
        # metabolism (inside, if fed): make interior content + lipids that grow M
        if c["fed"]:
            dC[inside] += c["g"]
            Cf[inside] += c["g"]
            Mg = M0 + c["k_inc"] * Cf[inside].sum()
        # membrane decays without replenishment
        Md = max(Mg - c["k_decay"] * Mg, 0.0)
        dM = Md - M0
        # more membrane -> tighter boundary (permeability falls)
        perm = 1.0 / (1.0 + c["k_perm"] * Md) if c["membrane_on"] else 1.0
        # diffusion (forward Euler), two boundary edges gated by permeability
        edge_D = np.full(N - 1, c["D"])
        edge_D[_LEFT_EDGE] = c["D"] * perm
        edge_D[_RIGHT_EDGE] = c["D"] * perm
        flux = edge_D * (Cf[1:] - Cf[:-1])
        dC[:-1] += flux
        dC[1:] -= flux

        Cnew = C0 + dC
        new_ratio = _ratio(Cnew, inside)
        return {
            "concentration": dC,
            "membrane": dM,
            "containment": new_ratio - float(state["containment"]),
        }


def spatial_state(fed=True, membrane=True, *, D=0.25, g=0.05, k_inc=0.015,
                  k_decay=0.006, k_perm=0.4, seed_M=10.0, seed_C=2.0):
    """The composite document (state dict) for the spatial-containment model."""
    C = np.zeros(N)
    C[INSIDE] = seed_C
    inside = np.zeros(N, bool)
    inside[INSIDE] = True
    return {
        "concentration": C,
        "membrane": float(seed_M),
        "containment": _ratio(C, inside),
        "spatial": {
            "_type": "process",
            "address": "local:SpatialContainment",
            "config": {"fed": bool(fed), "membrane_on": bool(membrane),
                       "D": D, "g": g, "k_inc": k_inc,
                       "k_decay": k_decay, "k_perm": k_perm},
            "interval": 1.0,
            "inputs": {"concentration": ["concentration"], "membrane": ["membrane"],
                       "containment": ["containment"]},
            "outputs": {"concentration": ["concentration"], "membrane": ["membrane"],
                        "containment": ["containment"]},
        },
    }


def _maybe_emitter(state, run_id, db_file):
    """Append a SQLiteEmitter (membrane + containment trajectory) if a run-db is
    requested. Prefers the canonical dashboard helper; the bare composite has no
    emitter to mirror, so we set the emit schema/inputs explicitly for our stores.
    Returns (state, emitter_label)."""
    if run_id is None or db_file is None:
        return state, None
    emit = {"membrane": "float", "containment": "float", "global_time": "float"}
    try:
        from vivarium_workbench.lib.composite_runs import inject_sqlite_emitter
        state = inject_sqlite_emitter(state, run_id=run_id, db_file=db_file)
        node = state["sqlite_emitter"]
        node["config"]["emit"] = dict(emit)
        node["inputs"] = {k: [k] for k in emit}
        return state, "sqlite"
    except Exception:  # pragma: no cover - dashboard not importable
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


def simulate(fed=True, membrane=True, steps=500, *, D=0.25, g=0.05, k_inc=0.015,
             k_decay=0.006, k_perm=0.4, seed_M=10.0, seed_C=2.0,
             run_id=None, db_file=None):
    """Run the spatial-containment composite, recording the same space-time
    history + metrics ``spatial.simulate`` returned (bit-identical numbers)."""
    state = spatial_state(fed, membrane, D=D, g=g, k_inc=k_inc, k_decay=k_decay,
                          k_perm=k_perm, seed_M=seed_M, seed_C=seed_C)
    state, _label = _maybe_emitter(state, run_id, db_file)
    from .core import build_core
    composite = Composite({"state": state}, core=build_core())

    inside = np.zeros(N, bool)
    inside[INSIDE] = True

    def _C():
        return np.asarray(composite.state["concentration"], dtype=float).copy()

    hist_C = [_C()]
    hist_M = [float(composite.state["membrane"])]
    hist_ratio = [_ratio(hist_C[0], inside)]
    for _ in range(steps):
        composite.run(1.0)
        C = _C()
        hist_C.append(C)
        hist_M.append(float(composite.state["membrane"]))
        hist_ratio.append(_ratio(C, inside))
    return {
        "C": np.array(hist_C), "M": np.array(hist_M),
        "ratio": np.array(hist_ratio), "inside": inside,
        "containment": float(hist_ratio[-1]),
    }
