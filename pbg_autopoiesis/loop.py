"""Increment 1 — build and run the minimal membrane/metabolism loop.

Wires Supply, Metabolism, Membrane, Boundary onto shared stores so the network
closes on itself except for the nutrient inflow. Provides:

  * build_loop(supply_rate)   -> a runnable Composite
  * run_trajectory(c, steps)  -> [volume per tick]
  * closure_of_loop()         -> the autopoiesis meter on this network
  * main()                    -> fed-vs-starved demo + the meter
"""
from __future__ import annotations

from process_bigraph import Composite

from .core import build_core as _core
from .processes import Supply, Metabolism, Membrane, Boundary, membrane_volume
from .meter import operational_closure, interface_of, report


def _proc_node(address, config, wires_in, wires_out, interval=1.0):
    return {
        "_type": "process",
        "address": f"local:{address}",
        "config": config,
        "interval": interval,
        "inputs": wires_in,
        "outputs": wires_out,
    }


def loop_state(supply_rate=2.0, *, seed_membrane=40.0, seed_precursor=10.0):
    """The raw network document (the state dict) for the minimal autopoietic loop.

    Exposed so callers (e.g. the ablation enumerator, which walks the document for
    ``_type == "process"`` entries) can read the wiring without instantiating a
    Composite."""
    return {
        # --- shared molecular stores (toy counts) ---
        "nutrient": 0.0,
        "precursor": float(seed_precursor),
        "lipid": 0.0,
        "membrane_lipids": float(seed_membrane),
        "volume": membrane_volume(seed_membrane),   # initial emergent boundary

        # --- the network ---
        "supply": _proc_node(
            "Supply", {"rate": supply_rate},
            {}, {"nutrient": ["nutrient"]}),
        "metabolism": _proc_node(
            "Metabolism", {},
            {"nutrient": ["nutrient"], "precursor": ["precursor"],
             "lipid": ["lipid"], "volume": ["volume"]},
            {"nutrient": ["nutrient"], "precursor": ["precursor"],
             "lipid": ["lipid"]}),
        "membrane": _proc_node(
            "Membrane", {},
            {"lipid": ["lipid"], "membrane_lipids": ["membrane_lipids"]},
            {"lipid": ["lipid"], "membrane_lipids": ["membrane_lipids"]}),
        "boundary": _proc_node(
            "Boundary", {},
            {"membrane_lipids": ["membrane_lipids"], "volume": ["volume"]},
            {"volume": ["volume"]}),
    }


def build_loop(supply_rate=2.0, *, seed_membrane=40.0, seed_precursor=10.0,
               external_membrane=False, injected_node=None):
    """A Composite of the minimal autopoietic loop. supply_rate=0 starves it.

    ``external_membrane=True`` adds a pbg-superpowers Intervention that CLAMPS
    membrane_lipids to its seed value every step — an externally-maintained
    boundary. This is the negative control the reviewers asked for: a system that
    is sustained from OUTSIDE rather than self-producing. Run it starved
    (supply_rate=0) and the identity should NOT collapse (precariousness fails),
    showing the metric discriminates self-production from external maintenance.

    ``injected_node`` accepts an arbitrary intervention-node dict (e.g. one emitted
    by ``viva_superpowers.ablation.generate_ablations`` / ``pbg_basic_processes.intervention.intervention_node``) and
    wires it into the loop — this is the rebuild hook the ablation suite's
    ``build_fn`` uses to inject a knockout/scale/decouple node into the network.
    """
    core = _core()
    state = loop_state(supply_rate, seed_membrane=seed_membrane,
                       seed_precursor=seed_precursor)
    if external_membrane:
        # Negative control: an externally-maintained boundary (clamped, not
        # self-produced) — the membrane is held at its seed value from outside.
        from pbg_basic_processes.intervention import register_intervention, intervention_node
        register_intervention(core)
        state["external_membrane"] = intervention_node(
            ["membrane_lipids"], mode="set", value=float(seed_membrane))
    if injected_node is not None:
        # Ablation hook: inject an externally-supplied intervention node (knockout /
        # scale / decouple / invert) targeting one of the loop's provided stores.
        from pbg_basic_processes.intervention import register_intervention
        register_intervention(core)
        state["ablation_node"] = injected_node
    return Composite({"state": state}, core=core)


def run_trajectory(composite, steps=160, *, return_fluxes=False):
    """Run one tick at a time, recording the volume (the size of the identity).

    When ``return_fluxes=True`` returns ``(vols, fluxes)`` where ``fluxes`` is
    ``{store: accumulated abs(delta)}`` over every scalar store across the run —
    the dynamical evidence ``meter.semantic_closure`` needs to confirm each wired
    self-produced type is actually produced (non-zero net flux)."""
    def _scalars(state):
        return {k: v for k, v in state.items() if isinstance(v, (int, float))}

    vols = [composite.state["volume"]]
    prev = _scalars(composite.state)
    fluxes = {k: 0.0 for k in prev}
    for _ in range(steps):
        composite.run(1.0)
        cur = _scalars(composite.state)
        for k, v in cur.items():
            fluxes[k] = fluxes.get(k, 0.0) + abs(v - prev.get(k, v))
        prev = cur
        vols.append(composite.state["volume"])
    if return_fluxes:
        return vols, fluxes
    return vols


def closure_of_loop():
    """The autopoiesis meter for this network. Supply is the environment: its
    product (nutrient) is the boundary input; the rest is the self-producing network."""
    core = _core()
    network = [interface_of(Metabolism({}, core=core)),
               interface_of(Membrane({}, core=core)),
               interface_of(Boundary({}, core=core))]
    boundary = Supply({}, core=core).outputs().keys()   # {'nutrient'} -- supplied from outside
    return operational_closure(network, boundary)


def main():
    print("=" * 64)
    print(" pbg-autopoiesis — increment 1: membrane/metabolism loop")
    print("=" * 64)
    print(report(closure_of_loop()))
    print()

    fed = run_trajectory(build_loop(supply_rate=1.0))
    starved = run_trajectory(build_loop(supply_rate=0.0))
    v0 = fed[0]
    print(f" initial volume (emergent boundary): {v0:.4f}")
    print(f" FED     (nutrient flowing): volume {v0:.4f} -> {fed[-1]:.4f}   "
          f"({'persists/grows' if fed[-1] >= v0 * 0.9 else 'shrinks'})")
    print(f" STARVED (nutrient = 0):     volume {v0:.4f} -> {starved[-1]:.4f}   "
          f"({'dissipates' if starved[-1] < v0 * 0.5 else 'holds'})")
    print()
    print(" => the identity is PRECARIOUS: it persists only through continuous")
    print("    self-production; cut the inflow and the self-produced boundary decays.")
    print("=" * 64)


if __name__ == "__main__":
    main()
