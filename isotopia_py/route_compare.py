"""Is route A really better than route B, given the cross-section uncertainties?

Not in the Fortran. Takes the DIUncertainty of two routes (xs_uncertainty.di_uncertainty)
and compares their DI amounts through the log of their ratio:

    var[ln(N_A/N_B)] = (dN_A/N_A)^2 + (dN_B/N_B)^2 - 2 cov_AB

cov_AB is not zero when the routes share cross-section data. The typical case is the same
target at two energy windows: both rates come from the same cross-section curve, so if
that curve is too high it is too high for both, and the error cancels in the ratio. For a
channel c present in both routes (same library file):

    cov(R_c^A, R_c^B) = a_A^T rho(E_A, E_B) a_B,   a = scale * w * dsigma * 1e-27

which is the sandwich rule of xs_uncertainty.rate_uncertainty with the two routes'
weights. In relative terms, cov_AB = sum_c s_c^A s_c^B cov(R_c^A, R_c^B) / (R_c^A R_c^B).
Different channels of the same target add channel_correlation * s s' r r' (0 by default);
channels of different targets are independent.

Why two correlation models. For ONE route, "full" energy correlation is the cautious
choice (largest band). For a COMPARISON of two windows on the same curve it is the
optimistic one: it assumes the error is a pure scale factor and cancels completely in the
ratio. A curve can also be wrong in shape (too high at 14 MeV, right at 25 MeV), which a
finite correlation length L describes. So the comparison is evaluated with both models and
the larger standard deviation is kept. Both values are reported.

Verdict (USER INPUT comparison_sigma, default 2): A is better if ln(N_A/N_B) > k * sd,
B is better if < -k * sd, otherwise the routes are tied and the tie is for the rarity,
toxicity and half-life criteria to break.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .xs_uncertainty import DIUncertainty, _label


@dataclass
class RouteComparison:
    """Result of compare_routes.

    Attributes
    ----------
    ratio : float
        N_A / N_B at the chosen time.
    log_ratio : float
        ln(N_A/N_B).
    sd : float
        1-sigma uncertainty of log_ratio: the largest over the correlation models tried.
    sd_by_model : dict of str to float
        The standard deviation under each energy-correlation model.
    correlation : float
        Correlation between the two routes' errors under the model that set `sd`.
    shared : list of str
        Channels whose cross-section data both routes use.
    verdict : str
        "A better", "B better", "tie" or "not comparable".
    notes : list of str
    """

    ratio: float
    log_ratio: float
    sd: float
    sd_by_model: dict[str, float]
    correlation: float
    shared: list[str]
    verdict: str
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        models = ", ".join(f"{m} {v:.4f}" for m, v in self.sd_by_model.items())
        text = (f"ratio A/B = {self.ratio:.4g}, ln-ratio {self.log_ratio:+.4f} +/- {self.sd:.4f} "
                f"[{models}], error correlation {self.correlation:.2f}; verdict: {self.verdict}")
        if self.shared:
            text += "\n  shared channels: " + ", ".join(self.shared)
        return text + "".join(f"\n  note: {n}" for n in self.notes)


def _variance(res: DIUncertainty, t: int, model: str) -> float:
    """(dN/N)^2 of one route at time index t, recomputed under an energy-correlation model.

    Uses the per-point contributions kept in each ChannelRate, so no file is read again.
    """
    st = res.settings
    chans = res.channels
    u = []
    for c in chans:
        a = c.rate.wdsigma
        dR = np.sqrt(max(float(a @ st.rho(c.rate.E, c.rate.E, model) @ a), 0.0))
        u.append(c.sensitivity[t] * (dR / c.rate.R if c.rate.R > 0 else 0.0))
    u = np.array(u)
    rho_c = np.full((len(u), len(u)), st.channel_correlation)
    np.fill_diagonal(rho_c, 1.0)
    return float(u @ rho_c @ u) if len(u) else 0.0


def _covariance(a: DIUncertainty, ta: int, b: DIUncertainty, tb: int,
                model: str) -> tuple[float, list[str]]:
    """cov_AB of the two routes' relative errors under an energy-correlation model."""
    st = a.settings
    cov, shared = 0.0, []
    for ca in a.channels:
        for cb in b.channels:
            sa, sb = ca.sensitivity[ta], cb.sensitivity[tb]
            pa, pb = ca.rate.channel.path, cb.rate.channel.path
            if ca.rate.R <= 0 or cb.rate.R <= 0:
                continue
            if pa == pb:
                # Same cross-section curve in both routes: sandwich rule across the two grids
                rho = st.rho(ca.rate.E, cb.rate.E, model)
                c = float(ca.rate.wdsigma @ rho @ cb.rate.wdsigma)
                cov += sa * sb * c / (ca.rate.R * cb.rate.R)
                shared.append(_label(ca.rate.channel.key))
            elif st.channel_correlation and Path(pa).parent == Path(pb).parent:
                # Different channels of the same target and projectile (one model calculation)
                ra = np.sqrt(float(ca.rate.wdsigma @ st.rho(ca.rate.E, ca.rate.E, model)
                                   @ ca.rate.wdsigma)) / ca.rate.R
                rb = np.sqrt(float(cb.rate.wdsigma @ st.rho(cb.rate.E, cb.rate.E, model)
                                   @ cb.rate.wdsigma)) / cb.rate.R
                cov += sa * sb * st.channel_correlation * ra * rb
    return cov, shared


def compare_routes(a: DIUncertainty, b: DIUncertainty, when: str = "eoi",
                   sigma: float | None = None,
                   models: tuple[str, ...] = ("full", "exponential")) -> RouteComparison:
    """Compare the DI amounts of two routes, with shared-data covariance.

    Parameters
    ----------
    a, b : DIUncertainty
        Results of xs_uncertainty.di_uncertainty for the same DI.
    when : "eoi" or "cooled"
        Compare at the end of irradiation or at the end of cooling.
    sigma : float, optional
        Threshold in standard deviations; default: a.settings.comparison_sigma.
    models : tuple of str
        Energy-correlation models to try; the largest standard deviation is used.

    Returns
    -------
    RouteComparison
    """
    if a.di != b.di:
        raise ValueError(f"different desired isotopes: {_label(a.di)} and {_label(b.di)}")
    ta = a.eoi if when == "eoi" else len(a.times) - 1
    tb = b.eoi if when == "eoi" else len(b.times) - 1
    k = a.settings.comparison_sigma if sigma is None else sigma
    Na, Nb = float(a.N[ta]), float(b.N[tb])
    if Na <= 0 or Nb <= 0:
        return RouteComparison(np.nan, np.nan, np.nan, {}, 0.0, [], "not comparable",
                               ["one route does not produce the DI"])

    # --- Standard deviation of ln(N_A/N_B) under each correlation model ------------------
    sd_by_model, corr_by_model, shared = {}, {}, []
    for model in models:
        va, vb = _variance(a, ta, model), _variance(b, tb, model)
        cov, shared = _covariance(a, ta, b, tb, model)
        sd_by_model[model] = float(np.sqrt(max(va + vb - 2 * cov, 0.0)))
        corr_by_model[model] = cov / np.sqrt(va * vb) if va * vb > 0 else 0.0
    worst = max(sd_by_model, key=sd_by_model.get)
    sd = sd_by_model[worst]

    # --- Verdict -------------------------------------------------------------------------
    lr = float(np.log(Na / Nb))
    verdict = "A better" if lr > k * sd else "B better" if lr < -k * sd else "tie"
    notes = []
    if a.lower_bound or b.lower_bound:
        notes.append("an uncertainty is a lower bound (TENDL default not set): a 'better' "
                     "verdict may really be a tie")
    if when == "eoi" and a.times[ta] != b.times[tb]:
        notes.append("the routes have different irradiation times")
    return RouteComparison(Na / Nb, lr, sd, sd_by_model, float(corr_by_model[worst]),
                           shared, verdict, notes)
