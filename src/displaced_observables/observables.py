"""Displacement-weighted jet substructure observables (see paper/main.tex)
plus standard baselines. All functions are pure, awkward-vectorized maps
from a per-jet track table (output of ``tracking.apply_tracking``) to one
number per jet.

Conventions: z_i = pT_i / pT_jet, theta_i = dR(i, jet)/R, and the
displacement weight w_i = log(1 + |d0_i| / sigma_i).
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import vector

vector.register_awkward()

JET_R = 1.0


def weight(trk) -> ak.Array:
    """Significance-based displacement weight, self-regulating against
    resolution: w = log(1 + |d0|/sigma)."""
    return np.log1p(trk.d0_abs / trk.sigma_d0)


# ---------------------------------------------------------------- tier 1

def angularity(jets, kappa: float, beta: float, signed: bool = False) -> ak.Array:
    """d0-weighted angularity  lambda^kappa_beta(w) = sum_i w_i z_i^kappa theta_i^beta.

    With ``signed=True`` the weight carries the jet-axis d0 sign
    (negative tails calibrate resolution in situ)."""
    trk = jets.trk
    w = weight(trk)
    if signed:
        w = w * np.sign(trk.d0)
    theta = trk.dr / JET_R
    return ak.sum(w * trk.z**kappa * theta**beta, axis=1)


# anchors: displaced-pT fraction, displaced girth, displaced-mass proxy
ANGULARITY_ANCHORS = {(1, 0): "disp. pT fraction", (1, 1): "disp. girth", (1, 2): "disp. mass"}


# ---------------------------------------------------------------- tier 2

def _pair_dr(pairs):
    dphi = np.mod(pairs.i.phi - pairs.j.phi + np.pi, 2 * np.pi) - np.pi
    return np.sqrt((pairs.i.eta - pairs.j.eta) ** 2 + dphi**2)


def deec(jets, beta: float = 1.0, g: str = "prod") -> ak.Array:
    """Displaced energy-energy correlator:
    dECF2(beta) = sum_{i<j} z_i z_j (dR_ij)^beta g(w_i, w_j),
    g = w_i*w_j ('prod') or min(w_i, w_j) ('min').

    Signal: dark-pion decays give *pairs* of displaced tracks from common
    vertices, so displacement is angularly correlated at small dR."""
    trk = jets.trk
    pairs = ak.combinations(trk, 2, axis=1, fields=["i", "j"])
    wi, wj = (np.log1p(p.d0_abs / p.sigma_d0) for p in (pairs.i, pairs.j))
    gij = wi * wj if g == "prod" else np.minimum(wi, wj)
    dr = _pair_dr(pairs)
    return ak.sum(pairs.i.z * pairs.j.z * dr**beta * gij, axis=1)


def wij_vs_dr(jets):
    """Ingredients of the differential profile <w_i w_j> vs dR_ij (money-plot
    candidate): flat arrays (dR_ij, w_i*w_j, z_i*z_j) over all pairs."""
    pairs = ak.combinations(jets.trk, 2, axis=1, fields=["i", "j"])
    wi, wj = (np.log1p(p.d0_abs / p.sigma_d0) for p in (pairs.i, pairs.j))
    return (
        ak.flatten(_pair_dr(pairs)),
        ak.flatten(wi * wj),
        ak.flatten(pairs.i.z * pairs.j.z),
    )


def decf(jets, n: int = 2, beta: float = 1.0, g: str = "prod") -> ak.Array:
    """Displaced energy correlation functions:
    dECF1 = sum_i z_i w_i,  dECF2 as above,
    dECF3 = sum_{i<j<k} z_i z_j z_k (dR_ij dR_ik dR_jk)^beta g(w_i,w_j,w_k),
    with g the product ('prod') or minimum ('min') of the leg weights.
    Sensitive to MULTIPLE displaced vertices per jet (dark showers have
    many; a B-decay chain has one)."""
    trk = jets.trk
    if n == 1:
        return ak.sum(trk.z * weight(trk), axis=1)
    if n == 2:
        return deec(jets, beta, g)
    if n == 3:
        t = ak.combinations(trk, 3, axis=1, fields=["i", "j", "k"])
        wi, wj, wk = (np.log1p(p.d0_abs / p.sigma_d0) for p in (t.i, t.j, t.k))
        gijk = wi * wj * wk if g == "prod" else np.minimum(np.minimum(wi, wj), wk)
        dij = _pair_dr(ak.zip({"i": t.i, "j": t.j}))
        dik = _pair_dr(ak.zip({"i": t.i, "j": t.k}))
        djk = _pair_dr(ak.zip({"i": t.j, "j": t.k}))
        return ak.sum(t.i.z * t.j.z * t.k.z * (dij * dik * djk) ** beta * gijk, axis=1)
    raise ValueError("decf supports n = 1, 2, 3")


def dc2(jets, beta: float = 1.0) -> ak.Array:
    """Displaced C2 = dECF3 dECF1 / dECF2^2 (product weighting: the powers
    of w cancel exactly as the powers of z do, so the ratio is normalized
    to the amount of displacement and probes its angular/vertex structure)."""
    e1, e2, e3 = (decf(jets, n, beta, "prod") for n in (1, 2, 3))
    return ak.where(e2 > 0, e3 * e1 / ak.where(e2 > 0, e2, 1) ** 2, 0.0)


def dd2(jets, beta: float = 1.0) -> ak.Array:
    """Displaced D2 = dECF3 dECF1^3 / dECF2^3 (product weighting)."""
    e1, e2, e3 = (decf(jets, n, beta, "prod") for n in (1, 2, 3))
    return ak.where(e2 > 0, e3 * e1**3 / ak.where(e2 > 0, e2, 1) ** 3, 0.0)


# ---------------------------------------------------------------- tier 3

def lifetime_moment(jets, n: int) -> ak.Array:
    """L_n = sum_i z_i |d0_i|^n."""
    trk = jets.trk
    return ak.sum(trk.z * trk.d0_abs**n, axis=1)


def lifetime_ratio(jets) -> ak.Array:
    """L2 L0 / L1^2 — dimensionless width of the |d0| distribution."""
    l0, l1, l2 = (lifetime_moment(jets, n) for n in (0, 1, 2))
    # ak.where evaluates both branches, so the denominator is guarded too:
    # a trackless jet would otherwise compute 0/0 and warn before it is masked
    safe = ak.where(l1 > 0, l1, 1.0)
    return ak.where(l1 > 0, l2 * l0 / safe**2, 0.0)


# ------------------------------------------------- standard (non-lifetime)
# Same functional forms with the displacement weight stripped (w_i -> 1):
# the direct comparison isolates what the lifetime weighting buys.

def angularity_std(jets, kappa: float, beta: float) -> ak.Array:
    """Standard track angularity  lambda^kappa_beta = sum_i z_i^kappa theta_i^beta.
    (1,1) girth · (1,2) mass-like · (2,0) -> pT^D squared · (0,0) n_trk."""
    trk = jets.trk
    theta = trk.dr / JET_R
    return ak.sum(trk.z**kappa * theta**beta, axis=1)


def eec(jets, beta: float = 1.0) -> ak.Array:
    """Standard two-point energy correlator sum_{i<j} z_i z_j (dR_ij)^beta."""
    trk = jets.trk
    pairs = ak.combinations(trk, 2, axis=1, fields=["i", "j"])
    return ak.sum(pairs.i.z * pairs.j.z * _pair_dr(pairs) ** beta, axis=1)


def ptd(jets) -> ak.Array:
    """pT^D = sqrt(sum pT^2) / sum pT (quark/gluon-style fragmentation hardness)."""
    trk = jets.trk
    tot = ak.sum(trk.pt, axis=1)
    return ak.where(tot > 0, np.sqrt(ak.sum(trk.pt**2, axis=1)) / ak.where(tot > 0, tot, 1), 0.0)


def n_tracks(jets) -> ak.Array:
    return ak.num(jets.trk.pt, axis=1)


def ecf(jets, n: int = 2, beta: float = 1.0) -> ak.Array:
    """Energy correlation function ECF_n (Larkoski-Salam-Thaler):
    ECF1 = sum z_i, ECF2 = sum_{i<j} z_i z_j dR_ij^beta,
    ECF3 = sum_{i<j<k} z_i z_j z_k (dR_ij dR_ik dR_jk)^beta."""
    trk = jets.trk
    if n == 1:
        return ak.sum(trk.z, axis=1)
    if n == 2:
        return eec(jets, beta)
    if n == 3:
        t = ak.combinations(trk, 3, axis=1, fields=["i", "j", "k"])
        dij = _pair_dr(ak.zip({"i": t.i, "j": t.j}))
        dik = _pair_dr(ak.zip({"i": t.i, "j": t.k}))
        djk = _pair_dr(ak.zip({"i": t.j, "j": t.k}))
        return ak.sum(t.i.z * t.j.z * t.k.z * (dij * dik * djk) ** beta, axis=1)
    raise ValueError("ecf supports n = 1, 2, 3")


def c2(jets, beta: float = 1.0) -> ak.Array:
    """C2 = ECF3 * ECF1 / ECF2^2 (two-prong discriminant)."""
    e1, e2, e3 = (ecf(jets, n, beta) for n in (1, 2, 3))
    return ak.where(e2 > 0, e3 * e1 / ak.where(e2 > 0, e2, 1) ** 2, 0.0)


def d2(jets, beta: float = 1.0) -> ak.Array:
    """D2 = ECF3 * ECF1^3 / ECF2^3."""
    e1, e2, e3 = (ecf(jets, n, beta) for n in (1, 2, 3))
    return ak.where(e2 > 0, e3 * e1**3 / ak.where(e2 > 0, e2, 1) ** 3, 0.0)


def _exclusive_axes(p4, jetdef, n: int):
    """Exclusive kT axes, without fastjet's spurious warning.

    ``fastjet._multievent._warn_for_exclusive`` compares the JetDefinition
    object against an algorithm *enum*, which is never equal, so it warns
    "for jet-finders other than kt, C/A or genkt" even for a genuine kt
    clustering. The axes here are kt by construction
    (``jetdef.jet_algorithm() == fastjet.kt_algorithm``), so the warning is a
    false positive and is filtered rather than propagated to every log.
    """
    import warnings

    import fastjet

    assert jetdef.jet_algorithm() == fastjet.kt_algorithm
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="dcut and exclusive jets for jet-finders other than",
            category=UserWarning)
        return fastjet.ClusterSequence(p4, jetdef).exclusive_jets(n_jets=n)


def nsubjettiness(jets, n: int, beta: float = 1.0) -> ak.Array:
    """tau_N with exclusive-kT axes on the jet's tracks:
    tau_N = sum_i z_i min_a(dR_ia)^beta / sum_i z_i R^beta.
    Jets with fewer than N tracks get tau_N = 0 (an N-track jet is exactly
    N-pronged, so 0 is the correct limit)."""
    import fastjet

    trk = jets.trk
    ok = ak.to_numpy(ak.num(trk.pt, axis=1) >= n)
    sub = trk[ok]
    p4 = ak.zip(
        {"pt": sub.pt, "eta": sub.eta, "phi": sub.phi,
         "M": ak.zeros_like(sub.pt)},
        with_name="Momentum4D",
    )
    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)
    axes = _exclusive_axes(p4, jetdef, n)

    pairs = ak.cartesian({"t": sub, "a": axes}, axis=1, nested=True)
    dphi = np.mod(pairs.t.phi - pairs.a.phi + np.pi, 2 * np.pi) - np.pi
    dr = np.sqrt((pairs.t.eta - pairs.a.eta) ** 2 + dphi**2)
    min_dr = ak.min(dr, axis=-1)
    norm = ak.sum(sub.z, axis=1) * JET_R**beta
    tau_sub = ak.where(norm > 0, ak.sum(sub.z * min_dr**beta, axis=1) / ak.where(norm > 0, norm, 1), 0.0)

    out = np.zeros(len(jets))
    out[ok] = ak.to_numpy(tau_sub)
    return ak.Array(out)


def nsubjettiness_disp(jets, n: int, beta: float = 1.0) -> ak.Array:
    """Displacement-aware N-subjettiness: exclusive-kT axes are found on the
    w-weighted track collection (pT_i -> w_i pT_i), so the axes follow the
    displaced substructure rather than the energy flow, and the sum is
    w-weighted as well:

        tau_N(disp) = sum_i w_i z_i min_a(dR_ia)^beta
                      / (sum_i w_i z_i R^beta),   axes from {w_i pT_i}.

    This counts *displaced prongs*: a b jet has a single displaced decay
    chain, while a dark shower has several vertices spread across the jet.
    Unlike a naive w-insertion into both numerator and denominator of the
    standard tau_N, the axis reweighting makes this more than a rescaled
    copy of its nominal partner."""
    import fastjet

    trk = jets.trk
    w = weight(trk)
    # a track needs a nonzero weight to define a displaced axis
    has_w = w > 0
    sub, wsub = trk[has_w], w[has_w]
    ok = ak.to_numpy(ak.num(sub.pt, axis=1) >= n)
    sub, wsub = sub[ok], wsub[ok]

    p4 = ak.zip(
        {"pt": sub.pt * wsub, "eta": sub.eta, "phi": sub.phi,
         "M": ak.zeros_like(sub.pt)},
        with_name="Momentum4D",
    )
    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)
    axes = _exclusive_axes(p4, jetdef, n)

    pairs = ak.cartesian({"t": sub, "a": axes}, axis=1, nested=True)
    dphi = np.mod(pairs.t.phi - pairs.a.phi + np.pi, 2 * np.pi) - np.pi
    dr = np.sqrt((pairs.t.eta - pairs.a.eta) ** 2 + dphi**2)
    min_dr = ak.min(dr, axis=-1)
    wz = wsub * sub.z
    norm = ak.sum(wz, axis=1) * JET_R**beta
    tau_sub = ak.where(norm > 0,
                       ak.sum(wz * min_dr**beta, axis=1) / ak.where(norm > 0, norm, 1),
                       0.0)

    out = np.zeros(len(jets))
    out[ok] = ak.to_numpy(tau_sub)
    return ak.Array(out)


def tau_ratio_disp(jets, n: int, m: int, beta: float = 1.0) -> ak.Array:
    """tau_n(disp) / tau_m(disp); 0 where the denominator vanishes."""
    tn = nsubjettiness_disp(jets, n, beta)
    tm = nsubjettiness_disp(jets, m, beta)
    return ak.where(tm > 0, tn / ak.where(tm > 0, tm, 1), 0.0)


def tau_ratio(jets, n: int, m: int, beta: float = 1.0) -> ak.Array:
    """tau_n / tau_m (e.g. tau21 = tau_ratio(jets, 2, 1)); 0 where tau_m = 0."""
    tn, tm = nsubjettiness(jets, n, beta), nsubjettiness(jets, m, beta)
    return ak.where(tm > 0, tn / ak.where(tm > 0, tm, 1), 0.0)


# ---------------------------------------------------------------- baselines

def mean_ip2d_significance(jets) -> ak.Array:
    """<|d0|/sigma> over tracks (0 for trackless jets)."""
    trk = jets.trk
    s = trk.d0_abs / trk.sigma_d0
    n = ak.num(s, axis=1)
    return ak.where(n > 0, ak.sum(s, axis=1) / ak.where(n > 0, n, 1), 0.0)


def prompt_pt_fraction(jets, nsig: float = 3.0) -> ak.Array:
    """alpha-like variable: pT fraction carried by prompt tracks
    (|d0/sigma| < nsig). Signal jets -> small, QCD -> ~1."""
    trk = jets.trk
    prompt = trk.d0_abs / trk.sigma_d0 < nsig
    tot = ak.sum(trk.pt, axis=1)
    return ak.where(tot > 0, ak.sum(trk.pt[prompt], axis=1) / ak.where(tot > 0, tot, 1), 1.0)
