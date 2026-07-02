"""Study 3 — adaptive chemotaxis as a REAL process-bigraph composite.

The numpy run-and-tumble population (``chemotaxis.py``) ported VERBATIM into a
single ``process_bigraph.Process``. The whole agent ensemble lives in one
``tree`` store (a dict of numpy arrays: x / dir / viability / alive / N_prev)
which the process REPLACES each tick (``tree`` apply = whole-value replace);
scalar observables (``n_alive`` / ``mean_nutrient``) are written set-by-difference
so the SQLite emitter persists the survival trajectory.

Determinism: numpy used ONE ``default_rng(seed)`` for both the agent setup
(uniform start positions + initial directions) and the per-step draws (tumble
test + new directions). The composite splits those: the state builder draws the
initial ensemble; the process re-creates ``default_rng(seed)`` and BURNS the
identical setup draws in ``__init__`` so its per-step stream is byte-aligned with
the numpy reference — the composite trajectory is bit-identical.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite, Process

L = 100.0          # 1-D environment length
X_FOOD = 80.0      # the food source position
SIGMA = 20.0       # food gradient width


def nutrient(x):
    return np.exp(-((x - X_FOOD) ** 2) / (2.0 * SIGMA ** 2))


def _init_agents(rng, n_agents, start, V0):
    """The agent-ensemble setup draws (must match numpy ``simulate`` exactly so
    the process can burn-align its rng stream)."""
    x = rng.uniform(start[0], start[1], n_agents)
    d = rng.choice([-1.0, 1.0], n_agents)
    V = np.full(n_agents, float(V0))
    alive = np.ones(n_agents, bool)
    return {"x": x, "d": d, "V": V, "alive": alive, "N_prev": nutrient(x)}


class Chemotaxis(Process):
    """One run-and-tumble + viability step for the whole population, ported
    verbatim from ``chemotaxis.simulate``'s loop body."""

    config_schema = {
        "chemotactic": {"_type": "boolean", "_default": True},
        "seed": {"_type": "integer", "_default": 0},
        "n_agents": {"_type": "integer", "_default": 80},
        "v": {"_type": "float", "_default": 0.7},
        "base_tumble": {"_type": "float", "_default": 0.20},
        "alpha": {"_type": "float", "_default": 16.0},
        "uptake": {"_type": "float", "_default": 0.06},
        "maintenance": {"_type": "float", "_default": 0.016},
        "V0": {"_type": "float", "_default": 1.5},
        "start_lo": {"_type": "float", "_default": 38.0},
        "start_hi": {"_type": "float", "_default": 56.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        c = self.config
        self.rng = np.random.default_rng(c["seed"])
        # Burn the setup draws so the per-step stream matches the numpy reference.
        _init_agents(self.rng, c["n_agents"], (c["start_lo"], c["start_hi"]), c["V0"])

    def inputs(self):
        return {"agents": "tree", "n_alive": "float", "mean_nutrient": "float"}

    def outputs(self):
        return {"agents": "tree", "n_alive": "float", "mean_nutrient": "float"}

    def update(self, state, interval):
        c = self.config
        n = c["n_agents"]
        a = state["agents"]
        if not a or "x" not in a:
            # Lazy seed: the composite doc ships an empty ``agents`` store (so the
            # dashboard explorer can build it without inlining the ensemble); the
            # spine seeds it in the state builder, so this fires only in the
            # explorer path. Uses the (post-burn) rng — a valid ensemble, not
            # byte-identical to a seed-matched spine run, which is fine for browsing.
            a = _init_agents(self.rng, n, (c["start_lo"], c["start_hi"]), c["V0"])
        x, d, V, alive, N_prev = a["x"], a["d"], a["V"], a["alive"], a["N_prev"]

        Nx = nutrient(x)
        dN = Nx - N_prev
        if c["chemotactic"]:
            p_tumble = np.clip(c["base_tumble"] * (1.0 - c["alpha"] * dN), 0.02, 0.95)
        else:
            p_tumble = np.full(n, c["base_tumble"])
        tumble = self.rng.random(n) < p_tumble
        d = np.where(tumble, self.rng.choice([-1.0, 1.0], n), d)
        x = np.clip(x + c["v"] * d * alive, 0.0, L)
        V = np.clip(V + c["uptake"] * nutrient(x) - c["maintenance"], 0.0, 2.0)
        alive = alive & (V > 0.0)

        n_alive = float(alive.sum())
        mean_nut = float((nutrient(x) * alive).sum() / max(alive.sum(), 1))
        return {
            "agents": {"x": x, "d": d, "V": V, "alive": alive, "N_prev": Nx},
            "n_alive": n_alive - float(state["n_alive"]),
            "mean_nutrient": mean_nut - float(state["mean_nutrient"]),
        }


def chemotaxis_state(chemotactic=True, n_agents=80, *, v=0.7, base_tumble=0.20,
                     alpha=16.0, uptake=0.06, maintenance=0.016, V0=1.5,
                     start=(38.0, 56.0), seed=0):
    rng = np.random.default_rng(seed)
    agents = _init_agents(rng, n_agents, start, V0)
    return {
        "agents": agents,
        "n_alive": float(agents["alive"].sum()),
        "mean_nutrient": 0.0,
        "chemotaxis": {
            "_type": "process",
            "address": "local:Chemotaxis",
            "config": {"chemotactic": bool(chemotactic), "seed": int(seed),
                       "n_agents": int(n_agents), "v": v, "base_tumble": base_tumble,
                       "alpha": alpha, "uptake": uptake, "maintenance": maintenance,
                       "V0": V0, "start_lo": start[0], "start_hi": start[1]},
            "interval": 1.0,
            "inputs": {"agents": ["agents"], "n_alive": ["n_alive"],
                       "mean_nutrient": ["mean_nutrient"]},
            "outputs": {"agents": ["agents"], "n_alive": ["n_alive"],
                        "mean_nutrient": ["mean_nutrient"]},
        },
    }


def _maybe_emitter(state, run_id, db_file):
    if run_id is None or db_file is None:
        return state, None
    emit = {"n_alive": "float", "mean_nutrient": "float", "global_time": "float"}
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


def simulate(chemotactic=True, n_agents=80, steps=900, *, v=0.7, base_tumble=0.20,
             alpha=16.0, uptake=0.06, maintenance=0.016, V0=1.5, start=(38.0, 56.0),
             seed=0, run_id=None, db_file=None):
    """Run the chemotaxis composite, recording the same trajectories + survival
    metrics ``chemotaxis.simulate`` returned (bit-identical numbers)."""
    state = chemotaxis_state(chemotactic, n_agents, v=v, base_tumble=base_tumble,
                             alpha=alpha, uptake=uptake, maintenance=maintenance,
                             V0=V0, start=start, seed=seed)
    state, _label = _maybe_emitter(state, run_id, db_file)
    from .core import build_core
    composite = Composite({"schema": {"agents": "tree"}, "state": state},
                          core=build_core())

    a0 = composite.state["agents"]
    X = [np.asarray(a0["x"]).copy()]
    VV = [np.asarray(a0["V"]).copy()]
    AL = [np.asarray(a0["alive"]).copy()]
    Nexp = []
    for _ in range(steps):
        composite.run(1.0)
        a = composite.state["agents"]
        x = np.asarray(a["x"]); alive = np.asarray(a["alive"])
        X.append(x.copy()); VV.append(np.asarray(a["V"]).copy()); AL.append(alive.copy())
        Nexp.append(float((nutrient(x) * alive).sum() / max(alive.sum(), 1)))
    alive = np.asarray(composite.state["agents"]["alive"])
    return {
        "X": np.array(X), "V": np.array(VV), "alive": np.array(AL),
        "survival": float(alive.mean()),
        "mean_nutrient": float(np.mean(Nexp)),
    }
