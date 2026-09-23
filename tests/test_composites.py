"""Studies 2-4 acceptance tests — the spatial / chemotaxis / growth models are
REAL process-bigraph composites whose measures land in the authored pass_if bands.

Run with the v2ecoli venv (process-bigraph + bigraph-schema) and the dashboard +
pbg-superpowers on PYTHONPATH (the emitter wiring is optional / defensive):
    PYTHONPATH=<superpowers>:<dashboard>:. <pb-venv>/bin/python -m pytest tests/ -q
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from viva_autopoiesis.core import build_core
from viva_autopoiesis import spatial, chemotaxis, growth
from viva_autopoiesis import processes_spatial, processes_chemotaxis, processes_growth

WS = Path(__file__).resolve().parent.parent
COMPOSITES = WS / "viva_autopoiesis" / "composites"


# --- the new processes resolve as real registered composite nodes ----------

@pytest.mark.parametrize("name", ["SpatialContainment", "Chemotaxis", "GrowthDivision"])
def test_process_registers(name):
    core = build_core()
    assert core.access(f"local:{name}") is not None


@pytest.mark.parametrize("slug", ["spatial-containment", "adaptive-chemotaxis",
                                  "growth-division"])
def test_composite_doc_name_matches_slug(slug):
    """The dashboard discovers ``viva_autopoiesis.composites.<slug>`` from the
    .composite.yaml — its ``name`` MUST equal the slug the study references."""
    doc = yaml.safe_load((COMPOSITES / f"{slug}.composite.yaml").read_text(encoding="utf-8"))
    assert doc["name"] == slug
    # every required process is one we register
    core = build_core()
    for proc in doc["requires"]["processes"]:
        assert core.access(f"local:{proc}") is not None


# --- study 2: spatial containment, measures in-band ------------------------

def test_spatial_containment_in_band():
    m, _ = spatial.containment_metrics()
    assert m["containment_ratio"] >= 3.0            # containment
    assert m["membrane_effect"] >= 2.0              # membrane-counters-diffusion
    assert m["precariousness_collapse"] < 0.3       # spatial-precariousness


def test_spatial_no_membrane_control_fails_to_contain():
    """The discriminating no-membrane variant must NOT contain like the membrane one."""
    held = spatial.simulate(fed=True, membrane=True)
    leaky = spatial.simulate(fed=True, membrane=False)
    assert held["containment"] > 2.0 * leaky["containment"]


# --- study 3: adaptive chemotaxis, measures in-band ------------------------

def test_chemotaxis_in_band():
    m, _ = chemotaxis.chemotaxis_metrics()
    assert m["chemotaxis_survival"] >= 0.6          # chemotaxis-survival
    assert m["survival_advantage"] >= 1.5           # agency-advantage
    assert m["gradient_advantage"] >= 1.15          # sense-making


def test_chemotaxis_deterministic_per_seed():
    a = chemotaxis.simulate(chemotactic=True, seed=0)["survival"]
    b = chemotaxis.simulate(chemotactic=True, seed=0)["survival"]
    assert a == b


# --- study 4: growth & division, measures in-band --------------------------

def test_growth_division_in_band():
    m, _ = growth.growth_division_metrics()
    assert m["final_population"] >= 10              # reproduction
    assert m["composition_heterogeneity"] >= 0.05   # heterogeneity
    assert m["division_mortality"] >= 0.01          # division-precariousness


def test_growth_no_supply_control_collapses():
    """Reproduction without self-maintenance (supply=0) must collapse the lineage."""
    normal = growth.simulate(supply=0.55, seed=0)["final_pop"]
    starved = growth.simulate(supply=0.0, seed=0)["final_pop"]
    assert starved < 0.5 * normal


# --- emitter persistence (real trajectory in a run-db) ---------------------

def test_growth_persists_trajectory(tmp_path):
    db = tmp_path / "runs.db"
    growth.simulate(steps=120, supply=0.55, seed=0, run_id="t", db_file=db)
    assert db.exists()
    import sqlite3
    con = sqlite3.connect(db)
    has = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
    ).fetchone()
    assert has, "the SQLite emitter should have created a history table"
    rows = con.execute("SELECT COUNT(*) FROM history WHERE simulation_id='t'").fetchone()[0]
    assert rows > 1, "a multi-step run should persist more than one history row"
