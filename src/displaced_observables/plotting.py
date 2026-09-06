"""Shared figure style: palette, annotation, binning and saving.

Imported only by ``commands/plot.py``. Compute steps never import this, so a
headless analysis job never pulls in matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

plt.style.use(hep.style.ATLAS)

PYTHIA_VERSION = "8.312"

# Display names for every observable, in one place.
LABELS = {
    "girth": "girth $\\lambda^1_1$", "mass_ang": "mass ang. $\\lambda^1_2$",
    "eec_b1": "ECF$_2(\\beta{=}1)$", "ecf3": "ECF$_3(\\beta{=}1)$",
    "c2": "$C_2$", "d2": "$D_2$", "tau1": "$\\tau_1$",
    "tau21": "$\\tau_{21}$", "tau32": "$\\tau_{32}$", "ptd": "$p_T^D$",
    "ntrk": "$n_{trk}$", "jetmass": "jet mass [GeV]",
    "ang_00": "$\\Sigma_i w_i$", "ang_10": "$\\lambda^1_0(w)$",
    "ang_11": "$\\lambda^1_1(w)$", "ang_12": "$\\lambda^1_2(w)$",
    "deec_b1": "dECF$_2$(prod)", "deec_min": "dECF$_2$(min)",
    "decf3": "dECF$_3$(prod)", "dc2": "$C_2(w)$", "dd2": "$D_2(w)$",
    "L1": "$L_1$ [mm]", "Lratio": "$L_2 L_0/L_1^2$",
    "ip2d": "$\\langle|d_0|/\\sigma\\rangle$",
    "promptfrac": "prompt $p_T$ fraction",
    "tau1_disp": "$\\tau_1(w)$", "tau21_disp": "$\\tau_{21}(w)$",
    "tau32_disp": "$\\tau_{32}(w)$", "jetpt": "jet $p_{\\mathrm{T}}$ [GeV]",
}

# Nominal references are muted and dashed so the displaced observables read as
# the foreground. One shared style once made girth and ntrk indistinguishable,
# so each gets its own colour and marker.
STD_STYLES = [("v", ":", "#7f8fa6"), ("s", (0, (5, 2)), "#34495e")]


def decorate(ax, extra: str | None = None) -> None:
    """ATLAS-style annotation: generator, CoM energy, process, jet algorithm."""
    lines = [
        f"Pythia {PYTHIA_VERSION}, $\\sqrt{{s}} = 13.6$ TeV",
        "$Z'(1.5\\,\\mathrm{TeV}) \\to q_{\\mathrm{D}}\\bar{q}_{\\mathrm{D}}$",
        "anti-$k_t$ $R=1.0$",
    ]
    if extra:
        lines.append(extra)
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=14)


def binning(values, n: int = 60, hi_q: float = 0.9999, lo: float = 0.0):
    """Bin edges from ``lo`` to the ``hi_q`` quantile.

    Every observable here is bounded below by zero, so the axis is anchored
    there rather than at a low quantile. Integer-valued observables get
    integer-width, integer-centred bins: a linspace over an integer range
    gives a fractional bin width, so some bins collect two integers and their
    neighbours one, which shows up as a comb (n_trk did exactly this).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(lo, lo + 1, n)
    hi = float(np.quantile(v, hi_q))
    if hi <= lo:
        hi = lo + 1.0
    if np.all(v == np.floor(v)):
        step = max(1, int(np.ceil((hi - lo) / n)))
        return np.arange(np.floor(lo) - 0.5, hi + step, step)
    return np.linspace(lo, hi, n)


def save(fig, path) -> None:
    """PNG for inspection and PDF for the paper, side by side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
