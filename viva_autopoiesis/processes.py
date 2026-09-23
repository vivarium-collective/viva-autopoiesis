"""Increment 1 — the minimal membrane/metabolism co-construction loop.

Four components implement the smallest self-bounding precarious identity:

    Supply      boundary inflow of nutrient (the environmental coupling; rate=0 starves)
    Metabolism  nutrient -> precursor -> lipid; lipid synthesis is BIMOLECULAR so it
                runs on CONCENTRATION and the volume genuinely couples the loop
    Membrane    grows by incorporating free lipids; DECAYS without them (precariousness)
    Boundary    derives the volume from the membrane (the boundary EMERGES, never declared)

The loop: Metabolism makes lipids -> Membrane grows -> Boundary derives V ->
V sets concentration -> Metabolism. Volume is the coupling variable; the
count<->concentration conversion is inline here (the real ConversionAdapter
factors it out in a later increment).

Numbers are arbitrary toy units. What is *real* is the loop structure: a
self-produced boundary, a load-bearing volume coupling, and observable
precariousness.
"""
from __future__ import annotations

import math

from process_bigraph import Process

# Conversion scale standing in for 1/N_A: concentration = count / (CONC_SCALE * volume).
CONC_SCALE = 1.0
# Area per membrane lipid (one leaflet). Sized so the emergent volume is O(1-10)
# and concentrations stay sane — which keeps the bimolecular lipid step in its
# kinetic (volume-governed) regime rather than saturating the stoichiometric clamp.
AREA_PER_LIPID = 0.5


def concentration(count: float, volume: float) -> float:
    if volume <= 0:
        return 0.0
    return count / (CONC_SCALE * volume)


def membrane_volume(membrane_lipids: float) -> float:
    """The boundary EMERGES from the molecular domain: a bilayer of N lipids
    encloses a surface area, which for a sphere fixes the enclosed volume.
    volume = geometry(membrane lipids) -- derived, never declared."""
    if membrane_lipids <= 0:
        return 0.0
    area = AREA_PER_LIPID * membrane_lipids / 2.0    # bilayer = two leaflets
    radius = math.sqrt(area / (4.0 * math.pi))
    return (4.0 / 3.0) * math.pi * radius ** 3


class Supply(Process):
    """Boundary inflow of nutrient -- the environmental coupling. rate=0 starves the cell."""
    config_schema = {'rate': {'_type': 'float', '_default': 0.0}}

    def inputs(self):
        return {}

    def outputs(self):
        return {'nutrient': 'float'}   # nutrient count

    def update(self, state, interval):
        return {'nutrient': self.config['rate'] * interval}


class Metabolism(Process):
    """Toy metabolism: nutrient -> precursor, then 2 precursor -> 1 lipid.

    The lipid step is BIMOLECULAR (rate ~ [precursor]^2), so it runs on
    concentration and the volume actually couples: a smaller (more concentrated)
    cell makes lipid faster. Emits lipid -- the membrane building block."""
    config_schema = {
        'k_precursor': {'_type': 'float', '_default': 0.30},
        'k_lipid': {'_type': 'float', '_default': 0.06},
    }

    def inputs(self):
        return {'nutrient': 'float', 'precursor': 'float',
                'lipid': 'float', 'volume': 'float'}

    def outputs(self):
        return {'nutrient': 'float', 'precursor': 'float', 'lipid': 'float'}

    def update(self, state, interval):
        v = max(state['volume'], 1e-9)
        n, p = state['nutrient'], state['precursor']

        # nutrient -> precursor (first order; volume cancels, as it should)
        flow_np = min(self.config['k_precursor'] * n * interval, n)

        # 2 precursor -> 1 lipid (bimolecular: rate ~ [precursor]^2 -> volume couples)
        cp = concentration(p, v)
        flow_lipid = self.config['k_lipid'] * cp * cp * (CONC_SCALE * v) * interval
        flow_lipid = min(flow_lipid, p / 2.0)   # need 2 precursor per lipid

        return {
            'nutrient': -flow_np,
            'precursor': flow_np - 2.0 * flow_lipid,
            'lipid': flow_lipid,
        }


class Membrane(Process):
    """The self-produced boundary's material: grows by incorporating free lipids,
    and DECAYS without them. Precariousness lives here -- stop the lipid flow and
    the membrane (and the identity it bounds) dissipates."""
    config_schema = {
        'k_incorporate': {'_type': 'float', '_default': 0.6},
        'k_decay': {'_type': 'float', '_default': 0.01},
    }

    def inputs(self):
        return {'lipid': 'float', 'membrane_lipids': 'float'}

    def outputs(self):
        return {'lipid': 'float', 'membrane_lipids': 'float'}

    def update(self, state, interval):
        incorporated = min(self.config['k_incorporate'] * state['lipid'] * interval,
                           state['lipid'])
        decayed = self.config['k_decay'] * state['membrane_lipids'] * interval
        return {
            'lipid': -incorporated,
            'membrane_lipids': incorporated - decayed,
        }


class Boundary(Process):
    """Derives the volume from the membrane -- the emergent boundary. Recomputes the
    target volume from the current membrane each tick and returns the DELTA (the
    float store applies updates additively, so we set-by-difference)."""
    config_schema = {}

    def inputs(self):
        return {'membrane_lipids': 'float', 'volume': 'float'}

    def outputs(self):
        return {'volume': 'float'}

    def update(self, state, interval):
        target = membrane_volume(state['membrane_lipids'])
        return {'volume': target - state['volume']}
