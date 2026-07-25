"""Wave-2 hardening tests — semantic closure (C-SEM) + invariant regression (C-INVAR).

Pure-function tests that do NOT require a fresh viva_superpowers (no Intervention /
ablation engine needed): they exercise ``meter.semantic_closure``, the trajectory
flux accumulation, and the ``spine.invariant_status`` band-comparison logic.

    PYTHONPATH=. <pb-venv>/bin/python -m pytest tests/test_wave2.py -q
"""
from pbg_autopoiesis.meter import semantic_closure, operational_closure, report
from pbg_autopoiesis.loop import build_loop, run_trajectory


# --- C-SEM: semantic vs interface (syntactic) closure ---------------------

def test_semantic_closure_all_fluxed():
    closure = {"closed": True, "self_produced": ["a", "b"]}
    sem = semantic_closure(closure, {"a": 1.0, "b": 0.2})
    assert sem["interface_closed"] is True
    assert sem["flux_nonzero"] == {"a": True, "b": True}
    assert sem["all_self_produced_fluxed"] is True
    assert sem["semantically_closed"] is True


def test_semantic_closure_inert_type_breaks_it():
    """A type wired to be self-produced but with ZERO flux is not semantically made."""
    closure = {"closed": True, "self_produced": ["a", "b"]}
    sem = semantic_closure(closure, {"a": 1.0, "b": 0.0})
    assert sem["flux_nonzero"] == {"a": True, "b": False}
    assert sem["all_self_produced_fluxed"] is False
    assert sem["semantically_closed"] is False


def test_semantic_closure_open_interface_cannot_be_semantic():
    closure = {"closed": False, "self_produced": ["a"]}
    sem = semantic_closure(closure, {"a": 5.0})
    assert sem["interface_closed"] is False
    assert sem["semantically_closed"] is False


def test_semantic_closure_no_self_produced():
    sem = semantic_closure({"closed": True, "self_produced": []}, {})
    assert sem["all_self_produced_fluxed"] is False
    assert sem["semantically_closed"] is False


def test_report_relabels_interface_closure():
    closure = operational_closure(
        [("A", {"x": "f", "n": "f"}, {"y": "f"}), ("B", {"y": "f"}, {"x": "f"})],
        boundary_inputs={"n"})
    assert "interface closure: CLOSED" in report(closure)


# --- the trajectory runner accumulates per-store fluxes -------------------

def test_run_trajectory_returns_fluxes():
    vols, fluxes = run_trajectory(build_loop(supply_rate=2.0), steps=30,
                                  return_fluxes=True)
    assert isinstance(vols, list) and len(vols) == 31
    # every self-produced store of the fed loop should actually flux
    for store in ("nutrient", "precursor", "lipid", "membrane_lipids", "volume"):
        assert fluxes[store] > 0.0, store


def test_run_trajectory_default_is_list():
    """Back-compat: the default return is still the volume list."""
    vols = run_trajectory(build_loop(supply_rate=2.0), steps=10)
    assert isinstance(vols, list) and len(vols) == 11


def test_semantic_closure_of_fed_loop():
    from pbg_autopoiesis.loop import closure_of_loop
    closure = closure_of_loop()
    _vols, fluxes = run_trajectory(build_loop(supply_rate=2.0), steps=40,
                                   return_fluxes=True)
    sem = semantic_closure(closure, fluxes)
    assert sem["semantically_closed"] is True


# --- C-INVAR: invariant-preservation band comparison ----------------------

def test_invariant_status_preserved_strengthened_weakened_invalidated():
    from pbg_autopoiesis.spine import invariant_status

    # "<" band (e.g. precariousness < 0.3): smaller value = stronger.
    band = {"op": "<", "value": 0.3}
    assert invariant_status(band, 0.10, 0.101) == "preserved"      # within tol
    assert invariant_status(band, 0.10, 0.02) == "strengthened"    # deeper into pass
    assert invariant_status(band, 0.10, 0.25) == "weakened"        # still passes, weaker
    assert invariant_status(band, 0.10, 0.40) == "invalidated"     # no longer passes

    # ">=" band (e.g. fed_volume_growth >= 1.0): larger value = stronger.
    up = {"op": ">=", "value": 1.0}
    assert invariant_status(up, 3.0, 3.05) == "preserved"
    assert invariant_status(up, 3.0, 6.0) == "strengthened"
    assert invariant_status(up, 3.0, 1.5) == "weakened"
    assert invariant_status(up, 3.0, 0.5) == "invalidated"


def test_invariant_status_range_band():
    from pbg_autopoiesis.spine import invariant_status
    band = {"op": "range", "low": 0.0, "high": 10.0}
    assert invariant_status(band, 5.0, 5.0) == "preserved"
    assert invariant_status(band, 5.0, 11.0) == "invalidated"
