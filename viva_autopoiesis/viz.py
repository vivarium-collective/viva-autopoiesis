"""Visual narrative of the autopoietic loop (increment 1).

Generates a gallery of detailed figures of the membrane/metabolism co-construction
loop: its precariousness, its self-maintained dynamics, the emergent boundary,
the volume coupling, the operational-closure cycle, and the phase portrait.

    PYTHONPATH=. <pb-venv>/bin/python -m viva_autopoiesis.viz
        -> writes figures/*.png + figures/index.html
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from .loop import build_loop, closure_of_loop
from . import processes as P

FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# --- a clean, consistent palette ------------------------------------------
INK = "#1c2330"; MUTED = "#5b6b82"; GRID = "#e7ecf3"
FED = "#1f9d6b"; STARVED = "#d1495b"; MEM = "#6a3fb5"; LIP = "#c98a17"
PREC = "#1f6feb"; ACCENT = "#11355e"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#c7d2e0", "axes.labelcolor": INK, "axes.titlecolor": ACCENT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def record(rate, steps=200, **seed):
    """Run the loop one tick at a time, recording the full molecular state."""
    c = build_loop(supply_rate=rate, **seed)
    keys = ["membrane_lipids", "lipid", "precursor", "nutrient", "volume"]
    hist = {k: [c.state[k]] for k in keys}
    hist["t"] = [0]
    for t in range(1, steps + 1):
        c.run(1.0)
        for k in keys:
            hist[k].append(c.state[k])
        hist["t"].append(t)
    out = {k: np.array(v, float) for k, v in hist.items()}
    out["conc_precursor"] = np.array(
        [P.concentration(p, v) for p, v in zip(out["precursor"], out["volume"])])
    out["radius"] = np.array([(3 * v / (4 * np.pi)) ** (1 / 3) if v > 0 else 0
                              for v in out["volume"]])
    return out


def _style(ax, title, xl, yl):
    ax.set_title(title, pad=10); ax.set_xlabel(xl); ax.set_ylabel(yl)


# === 1. precariousness — the hero shot =====================================
def fig_precariousness(fed, starved):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = fed["t"]
    ax.fill_between(t, fed["volume"], color=FED, alpha=0.10)
    ax.plot(t, fed["volume"], color=FED, lw=2.6, label="fed — self-maintained")
    ax.plot(starved["t"], starved["volume"], color=STARVED, lw=2.6, ls="--",
            label="starved — dissipation")
    v0 = fed["volume"][0]
    ax.axhline(v0, color=MUTED, lw=0.8, ls=":")
    ax.annotate("identity persists\nthrough material turnover",
                xy=(t[-1], fed["volume"][-1]), xytext=(t[-1] * 0.55, fed["volume"][-1] * 0.92),
                color=FED, fontsize=10, fontweight="bold", ha="left", va="top")
    ax.annotate("cut the inflow →\nthe self-produced\nboundary decays",
                xy=(starved["t"][-1], starved["volume"][-1]),
                xytext=(starved["t"][-1] * 0.5, v0 * 0.55),
                color=STARVED, fontsize=10, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=STARVED, lw=1.4))
    _style(ax, "Precariousness — the identity persists only through self-production",
           "time", "cell volume  (the emergent boundary)")
    ax.legend(frameon=False, loc="center left", fontsize=10)
    fig.tight_layout(); return fig


# === 2. self-maintained dynamics ===========================================
def fig_dynamics(fed):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    t = fed["t"]
    panels = [
        (axs[0, 0], fed["volume"], "Cell volume (boundary)", MEM, "volume"),
        (axs[0, 1], fed["membrane_lipids"], "Membrane lipids (the boundary's material)", MEM, "count"),
        (axs[1, 0], fed["lipid"], "Free lipid (building block)", LIP, "count"),
        (axs[1, 1], fed["precursor"], "Precursor pool", PREC, "count"),
    ]
    for ax, y, title, col, yl in panels:
        ax.plot(t, y, color=col, lw=2.4)
        ax.fill_between(t, y, color=col, alpha=0.08)
        _style(ax, title, "time", yl)
    fig.suptitle("The autopoietic loop settles into a self-maintained steady state",
                 fontsize=14, fontweight="bold", color=ACCENT, y=1.0)
    fig.tight_layout(); return fig


# === 3. the emergent boundary — the cell building itself ===================
def fig_emergent_boundary(fed):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    idx = np.linspace(0, len(fed["t"]) - 1, 6).astype(int)
    radii = fed["radius"][idx]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(idx)))
    x = 0.0
    for r, c, ti in zip(radii, cmap, fed["t"][idx]):
        ax.add_patch(Circle((x, 0), r, facecolor=c, alpha=0.18, edgecolor=c, lw=2.4))
        ax.text(x, -max(radii) * 1.35, f"t={int(ti)}", ha="center", color=MUTED, fontsize=9)
        x += 2 * max(radii) * 1.25
    ax.set_xlim(-max(radii) * 1.4, x); ax.set_ylim(-max(radii) * 1.7, max(radii) * 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("The cell builds its own boundary — volume = geometry(membrane lipids)",
                 color=ACCENT, fontweight="bold")
    fig.tight_layout(); return fig


# === 4. the volume coupling ================================================
def fig_volume_coupling():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    vols = np.linspace(0.5, 20, 200)
    m = P.Metabolism({}, core=__import__("viva_autopoiesis.loop", fromlist=["_core"])._core())
    rate = [m.update({"nutrient": 0, "precursor": 8.0, "lipid": 0, "volume": v}, 1.0)["lipid"]
            for v in vols]
    ax.plot(vols, rate, color=ACCENT, lw=2.8)
    ax.fill_between(vols, rate, color=ACCENT, alpha=0.08)
    ax.annotate("smaller cell → more concentrated\n→ faster lipid synthesis",
                xy=(vols[15], rate[15]), xytext=(6, max(rate) * 0.6),
                color=ACCENT, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    _style(ax, "Volume couples the loop  (lipid synthesis is bimolecular → runs on concentration)",
           "cell volume", "lipid synthesis rate  (count / time)")
    fig.tight_layout(); return fig


# === 5. operational-closure cycle ==========================================
def fig_closure_cycle():
    c = closure_of_loop()
    fig, ax = plt.subplots(figsize=(7.6, 6.4)); ax.axis("off"); ax.set_aspect("equal")
    nodes = {"metabolism": (0, 1.6), "lipid": (1.55, 0.5),
             "membrane": (1.0, -1.4), "volume": (-1.0, -1.4), "concentration": (-1.55, 0.5)}
    cols = {"metabolism": PREC, "lipid": LIP, "membrane": MEM, "volume": MEM, "concentration": PREC}
    for name, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 0.42, facecolor=cols[name], alpha=0.16,
                            edgecolor=cols[name], lw=2.6))
        ax.text(x, y, name, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=cols[name])
    order = ["metabolism", "lipid", "membrane", "volume", "concentration"]
    for a, b in zip(order, order[1:] + order[:1]):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.18",
                     arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#8aa0bb",
                     shrinkA=26, shrinkB=26))
    # nutrient enters from outside (the boundary)
    ax.add_patch(FancyArrowPatch((-0.9, 2.7), (-0.1, 2.0), arrowstyle="-|>",
                 mutation_scale=18, lw=2.0, color=STARVED, shrinkA=2, shrinkB=26))
    ax.text(-1.1, 2.8, "nutrient\n(boundary)", ha="center", color=STARVED, fontsize=9.5, fontweight="bold")
    ax.text(0, -2.55,
            f"operational closure: {'CLOSED' if c['closed'] else 'OPEN'}   ·   "
            f"self-produces {c['n_self_produced']}/{c['n_required']} types   ·   "
            f"gap = {{{', '.join(c['gap']) or '∅'}}}",
            ha="center", fontsize=11, fontweight="bold", color=ACCENT)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.9, 3.1)
    ax.set_title("Operational closure — the network produces what it requires",
                 color=ACCENT, fontweight="bold", pad=4)
    fig.tight_layout(); return fig


# === 6. phase portrait =====================================================
def fig_phase(fed):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sc = ax.scatter(fed["conc_precursor"], fed["membrane_lipids"], c=fed["t"],
                    cmap="viridis", s=14, zorder=3)
    ax.plot(fed["conc_precursor"], fed["membrane_lipids"], color="#9fb2c8", lw=1.0, zorder=2)
    ax.scatter([fed["conc_precursor"][0]], [fed["membrane_lipids"][0]], color=STARVED,
               s=70, zorder=4, label="start")
    ax.scatter([fed["conc_precursor"][-1]], [fed["membrane_lipids"][-1]], color=FED,
               s=90, marker="*", zorder=4, label="self-maintained attractor")
    fig.colorbar(sc, ax=ax, label="time")
    _style(ax, "Trajectory to the self-maintained attractor",
           "precursor concentration", "membrane lipids")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.tight_layout(); return fig


GALLERY = [
    ("01_precariousness", "Precariousness", "Fed, the self-produced boundary persists through complete material turnover; starved, it dissipates. The identity is a process, not a container."),
    ("02_dynamics", "Self-maintained dynamics", "Membrane, lipid, precursor, and volume settle into a steady state held against decay by continuous self-production."),
    ("03_emergent_boundary", "The emergent boundary", "The cell builds its own boundary: volume is derived from membrane lipids, never declared — identity from the molecular domain."),
    ("04_volume_coupling", "The volume coupling", "Lipid synthesis is bimolecular, so it runs on concentration = count/volume: a smaller, more concentrated cell makes lipid faster. Volume is the loop's hinge."),
    ("05_closure_cycle", "Operational closure", "The autopoietic meter: the network's provides cover its requires; only nutrient crosses the boundary; the gap is empty."),
    ("06_phase_portrait", "Phase portrait", "The loop's trajectory through (precursor concentration, membrane) space to its self-maintained attractor."),
]


# === STUDY 2 — spatial containment ========================================
def _spatial_regimes():
    from .spatial import containment_metrics, INSIDE
    _, regimes = containment_metrics()
    return regimes, INSIDE


def fig_containment_profiles(regimes, INSIDE):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(regimes["held"]["C"].shape[1])
    ax.axvspan(INSIDE.start, INSIDE.stop - 1, color="#eef2f7", label="cell interior")
    ax.plot(x, regimes["held"]["C"][-1], color=FED, lw=2.8, label="fed + membrane → held together")
    ax.plot(x, regimes["leaky"]["C"][-1], color=LIP, lw=2.4, ls="--", label="fed, no membrane → leaks out")
    ax.plot(x, regimes["starved"]["C"][-1], color=STARVED, lw=2.4, ls=":", label="starved → disperses")
    _style(ax, "The membrane holds the individual together against diffusion",
           "space (lattice position)", "interior content concentration")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout(); return fig


def fig_kymograph(regimes):
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.6))
    for ax, key, title in ((axs[0], "held", "Held together (fed + membrane)"),
                           (axs[1], "starved", "Dissolving (starved)")):
        C = regimes[key]["C"]
        im = ax.imshow(C, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, C.shape[1], 0, C.shape[0]], vmin=0, vmax=regimes["held"]["C"].max())
        _style(ax, title, "space", "time"); ax.grid(False)
    fig.colorbar(im, ax=axs, label="interior content", fraction=0.046, pad=0.02)
    fig.suptitle("Space-time: a self-maintained individual vs. one dissolving into the medium",
                 fontsize=13.5, fontweight="bold", color=ACCENT, y=1.02)
    return fig


def fig_containment_over_time(regimes):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    styles = [("held", FED, "-", "fed + membrane"), ("leaky", LIP, "--", "fed, no membrane"),
              ("starved", STARVED, ":", "starved")]
    for key, col, ls, lab in styles:
        ax.plot(regimes[key]["ratio"], color=col, lw=2.6, ls=ls, label=lab)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.annotate("dispersed (ratio → 1)", xy=(0.6, 1.0), xycoords=("axes fraction", "data"),
                color=MUTED, fontsize=9.5, va="bottom")
    _style(ax, "Containment over time — the individual held together only as a process",
           "time", "interior / exterior concentration ratio")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.tight_layout(); return fig


SPATIAL_GALLERY = [
    ("s2_01_profiles", "Containment", "The self-produced membrane keeps the interior concentrated; without it, metabolism alone leaks; starved, the individual disperses into the medium."),
    ("s2_02_kymograph", "Space-time", "Left: a self-maintained individual holds a stable boundary. Right: starved, the boundary fails and the individual dissolves."),
    ("s2_03_containment", "Held as a process", "Containment is high and stable only while the membrane is self-produced; stop the metabolism and it collapses toward the dispersed limit."),
]


def spatial_main(figdir):
    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    regimes, INSIDE = _spatial_regimes()
    figs = {
        "s2_01_profiles": fig_containment_profiles(regimes, INSIDE),
        "s2_02_kymograph": fig_kymograph(regimes),
        "s2_03_containment": fig_containment_over_time(regimes),
    }
    for name, fig in figs.items():
        fig.savefig(figdir / f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return list(figs)


# === STUDY 3 — adaptive chemotaxis ========================================
def _chemo_runs():
    from .chemotaxis import chemotaxis_metrics, X_FOOD, nutrient
    _, ctx = chemotaxis_metrics()
    return ctx["chemo"], ctx["blind"], X_FOOD, nutrient


def fig_chemotaxis_trajectories(chemo, blind, X_FOOD):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, run, title, col in ((axs[0], chemo, "Chemotactic — climbs to food, survives", FED),
                                (axs[1], blind, "Blind — wanders, dissolves", STARVED)):
        X, alive = run["X"], run["alive"]
        T, n = X.shape
        t = np.arange(T)
        for i in range(n):
            ax.plot(X[:, i], t, color=col, lw=0.6, alpha=0.22)
            if not alive[-1, i]:
                d = int(np.argmin(alive[:, i]))
                ax.scatter(X[d, i], d, color="#3a3a3a", s=8, zorder=3)
        ax.axvline(X_FOOD, color=LIP, lw=2.2, ls="--", label="food source")
        _style(ax, title, "position in environment", "time"); ax.set_xlim(0, 100)
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    axs[0].scatter([], [], color="#3a3a3a", s=8, label="dissolved")
    fig.suptitle("Movement toward food for survival — agency in service of self-maintenance",
                 fontsize=13.5, fontweight="bold", color=ACCENT, y=1.0)
    fig.tight_layout(); return fig


def fig_survival_curves(chemo, blind):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(chemo["alive"].mean(1), color=FED, lw=2.8, label=f"chemotactic — {chemo['survival']*100:.0f}% survive")
    ax.plot(blind["alive"].mean(1), color=STARVED, lw=2.8, ls="--", label=f"blind — {blind['survival']*100:.0f}% survive")
    ax.fill_between(np.arange(len(chemo["alive"])), chemo["alive"].mean(1), color=FED, alpha=0.08)
    _style(ax, "Survival — sensing the gradient is the difference between living and dissolving",
           "time", "fraction of the population alive")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout(); return fig


def fig_nutrient_landscape(chemo, blind, X_FOOD, nutrient):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = np.linspace(0, 100, 300)
    ax.plot(xs, nutrient(xs), color=LIP, lw=2.4, label="nutrient field")
    ax.fill_between(xs, nutrient(xs), color=LIP, alpha=0.08)
    cf = chemo["X"][-1][chemo["alive"][-1]]
    bf = blind["X"][-1][blind["alive"][-1]]
    ax.scatter(cf, nutrient(cf), color=FED, s=26, zorder=3, label="chemotactic survivors (at food)")
    ax.scatter(bf, nutrient(bf), color=STARVED, s=26, zorder=3, marker="x", label="blind survivors (scattered)")
    _style(ax, "Sense-making — the gradient is meaningful to a self that has something at stake",
           "position in environment", "nutrient")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); return fig


def chemotaxis_main(figdir):
    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    chemo, blind, X_FOOD, nutrient = _chemo_runs()
    figs = {
        "s3_01_trajectories": fig_chemotaxis_trajectories(chemo, blind, X_FOOD),
        "s3_02_survival": fig_survival_curves(chemo, blind),
        "s3_03_landscape": fig_nutrient_landscape(chemo, blind, X_FOOD, nutrient),
    }
    for name, fig in figs.items():
        fig.savefig(figdir / f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return list(figs)


# === STUDY 4 — growth & division ===========================================
def _growth_run():
    from .growth import growth_division_metrics, FOUNDER_THETA
    _, ctx = growth_division_metrics()
    return ctx["h"], FOUNDER_THETA


def fig_population_growth(h):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(h["pop_size"], color=MEM, lw=2.8)
    ax.fill_between(np.arange(len(h["pop_size"])), h["pop_size"], color=MEM, alpha=0.08)
    ax.annotate("one founder", xy=(0, 1), xytext=(len(h["pop_size"]) * 0.12, max(h["pop_size"]) * 0.4),
                color=MEM, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MEM, lw=1.4))
    _style(ax, "One individual becomes a population", "time", "number of individuals")
    fig.tight_layout(); return fig


def fig_diversification(h, founder):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    snaps = [s for s in h["snapshots"] if len(s[1]) > 2]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.85, len(snaps)))
    for (t, th), c in zip(snaps, cmap):
        ax.hist(th, bins=28, density=True, histtype="stepfilled", alpha=0.35,
                color=c, label=f"t={t}", edgecolor=c, lw=1.5)
    ax.axvline(founder, color=STARVED, lw=1.6, ls=":", label="founder")
    _style(ax, "Lineages diversify — heterogeneous compositions across the population",
           "heritable trait", "density")
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout(); return fig


def fig_heterogeneity_cost(h):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(h["hetero"], color=MEM, lw=2.8)
    ax.fill_between(np.arange(len(h["hetero"])), h["hetero"], color=MEM, alpha=0.08)
    mort = h["deaths"] / max(h["divisions"], 1)
    ax.text(0.97, 0.06,
            f"division is precarious:\n{mort*100:.0f}% of daughters inherit too little\nmembrane to maintain themselves → dissolve",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=STARVED,
            fontweight="bold")
    _style(ax, "Composition heterogeneity grows over generations",
           "time", "population trait spread (std)")
    fig.tight_layout(); return fig


def growth_main(figdir):
    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    h, founder = _growth_run()
    figs = {
        "s4_01_population": fig_population_growth(h),
        "s4_02_diversification": fig_diversification(h, founder),
        "s4_03_heterogeneity": fig_heterogeneity_cost(h),
    }
    for name, fig in figs.items():
        fig.savefig(figdir / f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return list(figs)


def main():
    FIGDIR.mkdir(exist_ok=True)
    fed = record(2.0); starved = record(0.0)
    figs = {
        "01_precariousness": fig_precariousness(fed, starved),
        "02_dynamics": fig_dynamics(fed),
        "03_emergent_boundary": fig_emergent_boundary(fed),
        "04_volume_coupling": fig_volume_coupling(),
        "05_closure_cycle": fig_closure_cycle(),
        "06_phase_portrait": fig_phase(fed),
    }
    for name, fig in figs.items():
        fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    _write_gallery()
    print(f"wrote {len(figs)} figures + index.html to {FIGDIR}")


def _write_gallery():
    cards = "\n".join(
        f'<figure><img src="{n}.png" alt="{t}"><figcaption><b>{t}.</b> {c}</figcaption></figure>'
        for n, t, c in GALLERY)
    html = f"""<!doctype html><meta charset=utf-8><title>viva-autopoiesis — increment 1</title>
<style>
 body{{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#1c2330;max-width:980px;
  margin:0 auto;padding:32px 24px;background:#fbfcfe}}
 h1{{font-size:26px;letter-spacing:-.4px;margin:0 0 2px}}
 .sub{{color:#5b6b82;font-size:14px;margin:0 0 26px}}
 figure{{margin:0 0 34px;border:1px solid #e4e9f1;border-radius:12px;overflow:hidden;
  background:#fff;box-shadow:0 1px 3px rgba(20,40,80,.05)}}
 img{{width:100%;display:block}}
 figcaption{{padding:12px 18px;font-size:13.5px;color:#2b3950;line-height:1.5;border-top:1px solid #eef2f7}}
 figcaption b{{color:#11355e}}
</style>
<h1>viva-autopoiesis &middot; increment 1</h1>
<p class=sub>The minimal membrane/metabolism co-construction loop — a precarious, self-bounding identity that emerges from the molecular domain.</p>
{cards}
"""
    (FIGDIR / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
