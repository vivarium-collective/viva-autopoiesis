"""Increment 1 acceptance tests — operational closure + precariousness.

Run with a venv that has process-bigraph + bigraph-schema, e.g.:
    PYTHONPATH=. <pb-venv>/bin/python -m pytest tests/ -q
"""
from viva_autopoiesis.loop import build_loop, run_trajectory, closure_of_loop
from viva_autopoiesis.meter import operational_closure


# --- the autopoiesis meter ------------------------------------------------

def test_meter_pure_gap_math():
    network = [
        ("A", {"x": "float", "nutrient": "float"}, {"y": "float"}),
        ("B", {"y": "float"}, {"x": "float"}),
    ]
    res = operational_closure(network, boundary_inputs={"nutrient"})
    assert res["closed"]                       # x,y self-produced; nutrient imported
    assert res["gap"] == []
    assert res["boundary"] == ["nutrient"]


def test_loop_is_operationally_closed():
    """The membrane/metabolism network closes on itself except for nutrient."""
    c = closure_of_loop()
    assert c["closed"], c
    assert c["gap"] == []
    assert c["boundary"] == ["nutrient"]
    # every internal type is produced by the network
    assert set(c["self_produced"]) == {"lipid", "membrane_lipids", "nutrient",
                                        "precursor", "volume"}


# --- precariousness: the identity persists only through self-production ----

def test_fed_cell_persists():
    vols = run_trajectory(build_loop(supply_rate=2.0), steps=120)
    assert vols[-1] > vols[0], "fed cell should grow / persist its boundary"


def test_starved_cell_dissipates():
    vols = run_trajectory(build_loop(supply_rate=0.0), steps=120)
    assert vols[-1] < vols[0] * 0.5, "starved cell should lose its self-produced boundary"


def test_precariousness_fed_beats_starved():
    fed = run_trajectory(build_loop(supply_rate=2.0), steps=120)
    starved = run_trajectory(build_loop(supply_rate=0.0), steps=120)
    assert fed[-1] > starved[-1]
    # the boundary is genuinely self-produced, not a static container:
    # stop production and it decays toward nothing.
    assert starved[-1] < fed[-1] * 0.2


# --- the volume coupling is load-bearing (concentration matters) ----------

def test_volume_couples_metabolism():
    """A smaller initial cell is more concentrated, so the bimolecular lipid
    step runs faster -- volume genuinely couples the loop (it is not a label)."""
    from viva_autopoiesis.processes import Metabolism
    from viva_autopoiesis.loop import _core
    m = Metabolism({}, core=_core())
    # precursor low enough that the bimolecular rate (not the p/2 clamp) governs,
    # so the volume dependence is observable.
    big = m.update({"nutrient": 0.0, "precursor": 5.0, "lipid": 0.0, "volume": 2.0}, 1.0)
    small = m.update({"nutrient": 0.0, "precursor": 5.0, "lipid": 0.0, "volume": 1.0}, 1.0)
    assert small["lipid"] > big["lipid"]   # smaller, more concentrated cell makes lipid faster
