"""The autopoiesis meter — operational closure over a set of process interfaces.

    autopoietic_gap(S) = requires(S) \\ provides(S) \\ boundary

For the network processes S (their typed input/output ports) and a declared
boundary (what the environment supplies), the gap is everything the network
needs but cannot make itself. The system is operationally closed -- a cell --
when the gap is empty. Building a whole cell = driving the gap down to
{nutrients, energy}.

This is the same provides/requires reasoning as the capability-catalog spike,
turned into a closure metric: the catalog is the autopoiesis meter.
"""
from __future__ import annotations


def operational_closure(network, boundary_inputs):
    """Compute the autopoietic gap.

    Args:
        network: iterable of (name, inputs_dict, outputs_dict) for the
                 SELF-PRODUCING processes (exclude pure boundary/environment
                 processes -- their products are boundary inputs).
        boundary_inputs: iterable of port/store names the environment supplies
                 (e.g. the outputs of the supply/environment processes).

    Returns a dict: provides, requires, boundary, gap, closed.
    """
    provides: set[str] = set()
    requires: set[str] = set()
    for _name, ins, outs in network:
        requires |= set(ins)
        provides |= set(outs)
    boundary = set(boundary_inputs)
    gap = requires - provides - boundary
    return {
        "provides": sorted(provides),
        "requires": sorted(requires),
        "boundary": sorted(boundary),
        "gap": sorted(gap),
        "closed": not gap,
        "self_produced": sorted(requires & provides),
        "n_required": len(requires),
        "n_self_produced": len(requires & provides),
    }


def interface_of(proc):
    """(name, inputs, outputs) for a process-bigraph process/step instance."""
    return (type(proc).__name__, proc.inputs(), proc.outputs())


def semantic_closure(closure: dict, fluxes: dict) -> dict:
    """Semantic vs syntactic closure.

    ``operational_closure`` is a SYNTACTIC / interface check: it asks only whether
    the network's typed ports cover its requirements (gap empty). That a type is
    *wired* to be self-produced does not prove it is *actually produced* under the
    dynamics — the wiring could be inert. Semantic closure adds a cheap dynamical
    proxy: each ``self_produced`` type must carry NON-ZERO net flux over the fed
    trajectory (it is genuinely being made/turned over, not a dead label).

    Args:
        closure: an :func:`operational_closure` result (``closed`` + ``self_produced``).
        fluxes:  ``{store: accumulated abs(delta)}`` from the trajectory runner.

    Returns ``{interface_closed, flux_nonzero, all_self_produced_fluxed,
    semantically_closed}``. ``semantically_closed = interface_closed and
    all_self_produced_fluxed``. Pure.
    """
    interface_closed = bool(closure.get("closed"))
    self_produced = list(closure.get("self_produced", []))
    flux_nonzero = {s: float(fluxes.get(s, 0.0)) > 0.0 for s in self_produced}
    all_self_produced_fluxed = bool(self_produced) and all(flux_nonzero.values())
    return {
        "interface_closed": interface_closed,
        "flux_nonzero": flux_nonzero,
        "all_self_produced_fluxed": all_self_produced_fluxed,
        "semantically_closed": interface_closed and all_self_produced_fluxed,
    }


def report(closure: dict) -> str:
    """One-line-per-fact human summary of a closure result.

    The gap-empty result is the SYNTACTIC / INTERFACE closure (the ports cover the
    requirements). Whether the network is *semantically* closed — every wired
    self-produced type actually fluxing — is a separate signal computed by
    :func:`semantic_closure` from trajectory fluxes.
    """
    lines = [
        f"interface closure: {'CLOSED' if closure['closed'] else 'OPEN'}  "
        f"(self-produces {closure['n_self_produced']}/{closure['n_required']} required types)",
        f"  boundary  (imported): {{{', '.join(closure['boundary']) or '∅'}}}",
        f"  gap   (not yet made): {{{', '.join(closure['gap']) or '∅'}}}",
    ]
    return "\n".join(lines)
