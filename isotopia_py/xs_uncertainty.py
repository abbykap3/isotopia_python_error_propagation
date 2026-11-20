"""Cross-section uncertainty of the desired isotope (DI) for one ISOTOPIA route.

Not in the Fortran. This module answers one question for a route that the port has
already run: by how much could the DI amount be wrong because the cross sections are
uncertain? The answer is used to compare routes (route_compare.py), not to certify an
absolute yield.

Overview (the write-up "MIDAS cross-section uncertainty: theory and implementation"
explains each step in detail):

1. RateGrid        The port computes every reaction rate as a weighted sum over an
                   energy grid (manual Eq. 4.20 for charged particles, Eq. 4.11 for
                   neutrons and photons):  R = scale * sum_k w_k * sigma_k * 1e-27.
                   rates._reactionrates keeps that grid and its weights on the State.
2. XSChannel       For one channel (target -> product) we read the library file again,
                   including the 'dxs' column the port ignores, and decide the 1-sigma
                   uncertainty of every grid point from its data source:
                     IAEA            the evaluation's own dxs
                     IAEA_EXTENDED   dxs = 0 in the file (TENDL scaled to the IAEA end
                                     point): the TENDL default is used instead
                     TENDL           no uncertainty in the file: the TENDL default
                     USER            an 'xsfile' file: its dxs column if it has one
3. ChannelRate     The uncertainty of R follows from the "sandwich rule"
                   var(R) = scale^2 * (w*dsigma)^T rho (w*dsigma), exact because R is a
                   linear function of sigma. rho is the energy-correlation model.
4. DIUncertainty   The DI amount N depends on several rates (direct production, decay of
                   a parent, later double hitting). Its sensitivity to each rate,
                   s_c = dlnN / dlnR_c, is found by rerunning the kinetics with one rate
                   changed (finite differences), so any solver can be used. Then
                   (dN/N)^2 = sum_c sum_c' s_c s_c' rho_cc' (dR_c/R_c) (dR_c'/R_c').

Units: energies MeV, cross sections mb in the library files (converted to cm^2 by the
factor 1e-27 exactly where the port does it), rates s^-1, times s, amounts atoms.

Inputs the team must set are in xs_uncertainty_defaults.json, next to this file; the
places that use them are marked "USER INPUT".
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .constants import NUMTIME, PARSYM, nuc

Key = tuple[int, int, int]        # ISOTOPIA's (Z, A, is), is = -1 total, 0 ground, 1 isomer, ...
Nuclide = tuple[int, int, int]    # project convention (Z, A, I), I = 0 ground, 1 first isomer

MB_TO_CM2 = 1e-27                 # the port's own conversion factor (rates.py)

# Data-source tags, attached to every energy point of every channel
IAEA, IAEA_EXTENDED, TENDL, USER = "IAEA", "IAEA-extended", "TENDL", "user"
SOURCES = (IAEA, IAEA_EXTENDED, TENDL, USER)

DEFAULTS_FILE = Path(__file__).with_name("xs_uncertainty_defaults.json")


# =====================================================================================
# 0. Settings (USER INPUT lives in the JSON file, not in this code)
# =====================================================================================

@dataclass
class Settings:
    """Choices that are physics decisions, read from xs_uncertainty_defaults.json.

    Attributes
    ----------
    tendl_relative : dict of str to float or None
        USER INPUT. Default 1-sigma relative uncertainty of a model-only (TENDL) cross
        section, per projectile family: "charged" (p, d, t, h, a), "neutron", "photon".
        None means "not chosen yet": those channels then add nothing and every result
        that uses them is flagged as a lower bound.
    energy_correlation : str
        Correlation between energies within one channel: "full" (all energies move
        together; the cautious default), "none" (independent points; diagnostic only,
        it depends on the grid spacing), or "exponential" (rho = exp(-|E1-E2|/L)).
    correlation_length : float
        L for the exponential model [MeV].
    channel_correlation : float
        USER INPUT. Correlation between different channels, 0 (independent) to 1.
    comparison_sigma : float
        USER INPUT. Route A beats route B when ln(N_A/N_B) exceeds this many standard
        deviations; otherwise the routes are tied.
    fd_step : float
        Relative change of a rate used for finite-difference sensitivities.
    selfcheck_tolerance : float
        Allowed relative difference between our rebuilt rate and the port's rate.
    """

    tendl_relative: dict[str, float | None]
    energy_correlation: str = "full"
    correlation_length: float = 2.0
    channel_correlation: float = 0.0
    comparison_sigma: float = 2.0
    fd_step: float = 1e-4
    selfcheck_tolerance: float = 1e-3

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Settings:
        """Read the settings file [default: xs_uncertainty_defaults.json beside this module]."""
        with open(path or DEFAULTS_FILE, encoding="utf-8") as fh:
            raw = json.load(fh)
        # Keys starting with "_" are comments for the reader of the JSON file
        values = {k: v for k, v in raw.items() if not k.startswith("_")}
        settings = cls(**values)
        if settings.energy_correlation not in ("full", "none", "exponential"):
            raise ValueError(f"energy_correlation must be full, none or exponential, "
                             f"not {settings.energy_correlation!r}")
        return settings

    def tendl_default(self, k0: int) -> float | None:
        """TENDL default for projectile index k0 (0 photon, 1 neutron, 2-6 charged)."""
        family = "photon" if k0 == 0 else "neutron" if k0 == 1 else "charged"
        return self.tendl_relative.get(family)

    def rho(self, E1: np.ndarray, E2: np.ndarray, model: str | None = None) -> np.ndarray:
        """Energy-correlation matrix rho[j, k] between the points E1[j] and E2[k] [MeV].

        `model` overrides energy_correlation ("full", "none" or "exponential").
        """
        model = model or self.energy_correlation
        dE = np.abs(E1[:, None] - E2[None, :])
        if model == "full":
            return np.ones_like(dE)
        if model == "none":
            return (dE < 1e-9).astype(float)       # only identical energies are correlated
        return np.exp(-dE / self.correlation_length)


# =====================================================================================
# 1. The energy grid and weights of the port's rate integral
# =====================================================================================

@dataclass
class RateGrid:
    """Integration grid of rates._reactionrates, so that R = scale * sum(w * sigma) * 1e-27.

    Attributes
    ----------
    k0 : int
        Projectile index (0 photon, 1 neutron, 2-6 p, d, t, h, a).
    E : ndarray
        Integration energies [MeV]. Charged particles: 100 midpoints between Eback and
        Ebeam. Neutrons and photons: the source spectrum's energies.
    w : ndarray
        Weights. Charged: dE/S(E) [cm] (manual Eq. 4.20). Neutrons: spectrum times
        self-shielding factor G [dimensionless]. Photons: spectrum times dE.
    scale : float
        Charged: projectile rate density n_p [cm^-3 s^-1] (manual Eq. 4.21). Neutrons and
        photons: total flux [cm^-2 s^-1].
    """

    k0: int
    E: np.ndarray
    w: np.ndarray
    scale: float

    @classmethod
    def from_port(cls, k0, Eint, dEint, phi, G, scale) -> RateGrid:
        """Build from the local lists of rates._reactionrates (1-based, index 0 unused).

        The weights reproduce the port's sum exactly: neutrons use G*phi (the spectrum
        already holds the group widths), photons and charged particles use phi*dE.
        """
        E = np.asarray(Eint[1:], dtype=float)
        phi = np.asarray(phi[1:], dtype=float)
        if k0 == 1:
            w = np.asarray(G[1:], dtype=float) * phi
        else:
            w = phi * np.asarray(dEint[1:], dtype=float)
        return cls(k0, E, w, float(scale))


# =====================================================================================
# 2. One channel: cross section and its uncertainty on the grid
# =====================================================================================

@dataclass
class XSChannel:
    """Cross section of one channel and its 1-sigma uncertainty on the integration grid.

    Attributes
    ----------
    key : Key
        Product in ISOTOPIA's (Z, A, is) convention.
    path : str
        Library file read.
    library : str
        The file's 'library:' header line (e.g. "iaea.2024 + TENDL-2023", "tendl.2025").
    sigma, dsigma : ndarray
        Cross section and its uncertainty on RateGrid.E [mb]. Zero outside the file's range,
        exactly as the port does.
    source : ndarray of str
        Data-source tag of every grid point (IAEA, IAEA-extended, TENDL, user).
    warnings : list of str
    """

    key: Key
    path: str
    library: str
    sigma: np.ndarray
    dsigma: np.ndarray
    source: np.ndarray
    warnings: list[str] = field(default_factory=list)


def read_xs_file(path: str) -> tuple[dict[str, str], np.ndarray]:
    """Read a YANDF cross-section table, keeping every column.

    Returns the header as {key: value} (e.g. 'library', 'source', 'columns') and the data
    as an array with one row per energy: E [MeV], xs [mb] and, if present, dxs [mb].
    The port's reader (data._read_xs_table) keeps only the first two columns.
    """
    header: dict[str, str] = {}
    rows: list[list[float]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # '# library: tendl.2025' -> header['library'] = 'tendl.2025'
                text = line.lstrip("#").strip()
                if ":" in text:
                    name, _, value = text.partition(":")
                    header.setdefault(name.strip(), value.strip())
                continue
            rows.append([float(x) for x in line.split()[:3]])
    width = min(len(r) for r in rows) if rows else 2
    return header, np.array([r[:width] for r in rows], dtype=float).reshape(-1, width)


def channel_path(st, key: Key) -> tuple[str, bool]:
    """Library file the port reads for product `key`, and whether it is a user 'xsfile'.

    Mirrors data.crosssections: an 'xsfile' keyword overrides the library.
    """
    iz, ia, is_ = key
    override = st.xsfile.get(key, "")
    if override.strip():
        return override, True
    ZA = f"{iz:03d}{ia:03d}" + {0: "g", 1: "m", 2: "n"}.get(is_, "")
    name = f"{PARSYM[st.k0]}-{st.nuclide}-rp{ZA}.{st.libname}"
    return os.path.join(st.xspath, "residual", name), False


def _on_grid(Elib: np.ndarray, ylib: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Linear interpolation onto the grid, zero outside [Elib[0], Elib[-1]).

    Same rule as the port's rate loop (locate + pol1): points below the first or at/after
    the last library energy contribute nothing.
    """
    inside = (E >= Elib[0]) & (E < Elib[-1])
    return np.where(inside, np.interp(E, Elib, ylib), 0.0)


def load_channel(st, key: Key, settings: Settings) -> XSChannel:
    """Cross section and 1-sigma uncertainty of one channel on the port's grid.

    This is where the decision table of the write-up is applied, point by point.

    Parameters
    ----------
    st : State
        Port state after a run (needs st.rate_grid, st.xspath, st.k0, st.nuclide).
    key : Key
        Product (Z, A, is).
    settings : Settings

    Returns
    -------
    XSChannel
    """
    grid: RateGrid = st.rate_grid
    path, user_file = channel_path(st, key)
    header, data = read_xs_file(path)
    library = header.get("library", "unknown")
    Elib, xs = data[:, 0], data[:, 1]
    has_dxs = data.shape[1] >= 3
    sigma = _on_grid(Elib, xs, grid.E)
    warnings: list[str] = []
    label = f"{st.targetnuclide.strip()} -> {_label(key)}"

    # --- Which points carry their own uncertainty? -----------------------------------
    if has_dxs:
        dxs_grid = _on_grid(Elib, data[:, 2], grid.E)
        # dxs interpolates to a small nonzero value right next to the last evaluated
        # point; treat a point as evaluated only if both bracketing library points are.
        evaluated = _on_grid(Elib, (data[:, 2] > 0).astype(float), grid.E) > 0.999
        own = USER if user_file else IAEA
        source = np.where(evaluated, own, IAEA_EXTENDED if not user_file else USER)
    else:
        dxs_grid = np.zeros_like(sigma)
        evaluated = np.zeros(sigma.shape, dtype=bool)
        source = np.full(sigma.shape, USER if user_file else TENDL, dtype=object)
    source = np.asarray(source, dtype=object)

    # --- Points without their own uncertainty get the TENDL default ------------------
    need_default = ~evaluated & (sigma > 0)
    if need_default.any():
        f = settings.tendl_default(st.k0)          # USER INPUT (json: tendl_relative)
        if f is None:
            warnings.append(f"{label}: model-based cross section and no TENDL default set in "
                            f"{DEFAULTS_FILE.name}; this channel adds no uncertainty, so the "
                            "result is a LOWER BOUND")
            f = 0.0
        else:
            why = ("the energy window extends beyond the IAEA evaluated range"
                   if has_dxs and not user_file else "model-only cross section (no dxs)")
            warnings.append(f"{label}: {why}; default uncertainty of {100 * f:g}% assumed")
        dxs_grid = np.where(need_default, f * sigma, dxs_grid)

    return XSChannel(key, path, library, sigma, dxs_grid, source, warnings)


# =====================================================================================
# 3. Rate uncertainty of one channel (sandwich rule)
# =====================================================================================

@dataclass
class ChannelRate:
    """Reaction rate of one channel and its cross-section uncertainty.

    Attributes
    ----------
    channel : XSChannel
    R : float
        Rate rebuilt from the grid [s^-1 per target atom].
    dR : float
        1-sigma uncertainty of R [s^-1].
    R_port : float
        The port's own rate for this key [s^-1], for the self-check.
    wdsigma : ndarray
        scale * w * dsigma * 1e-27 on the grid: each point's contribution to dR before
        correlation [s^-1]. Kept for covariances between routes (route_compare.py).
    E : ndarray
        Grid energies [MeV].
    source_share : dict of str to float
        Fraction of R contributed by each data source (sum 1).
    """

    channel: XSChannel
    R: float
    dR: float
    R_port: float
    wdsigma: np.ndarray
    E: np.ndarray
    source_share: dict[str, float]

    @property
    def rel(self) -> float:
        """dR / R (0 if R = 0)."""
        return self.dR / self.R if self.R > 0 else 0.0


def rate_uncertainty(st, key: Key, settings: Settings) -> ChannelRate:
    """R +- dR for one channel: manual Eq. 4.20 (4.11 for n, gamma) plus the sandwich rule.

    var(R) = sum_j sum_k a_j rho_jk a_k,  a_k = scale * w_k * dsigma_k * 1e-27.

    With full correlation (rho = 1) this is simply dR = sum_k a_k: the relative rate
    uncertainty is the rate-weighted average of the pointwise relative uncertainties.
    """
    grid: RateGrid = st.rate_grid
    ch = load_channel(st, key, settings)
    contrib = grid.scale * grid.w * ch.sigma * MB_TO_CM2      # each point's share of R
    a = grid.scale * grid.w * ch.dsigma * MB_TO_CM2
    R = float(contrib.sum())
    # a @ rho @ a is the double sum written above ('@' is matrix multiplication)
    dR = float(np.sqrt(max(a @ settings.rho(grid.E, grid.E) @ a, 0.0)))
    share = {s: float(contrib[ch.source == s].sum() / R) if R > 0 else 0.0 for s in SOURCES}

    # Self-check: our rebuilt rate must be the port's rate, or the uncertainty would be
    # attached to a different number from the one reported.
    R_port = float(st.reaction_rate[key])
    if R_port > 0 and abs(R / R_port - 1) > settings.selfcheck_tolerance:
        ch.warnings.append(f"{_label(key)}: rebuilt rate {R:.6e} s^-1 differs from the port's "
                           f"{R_port:.6e} s^-1 by more than {settings.selfcheck_tolerance:g}")
    return ChannelRate(ch, R, dR, R_port, a, grid.E.copy(), share)


# =====================================================================================
# 4. From rates to the DI: sensitivities through a solver
# =====================================================================================

class Solver(Protocol):
    """What di_uncertainty needs from a kinetics solver (the port's or Nate's)."""

    times: np.ndarray            # output times [s]
    eoi: int                     # index of the end of irradiation in `times`

    def candidates(self) -> list[Key]:
        """Port keys whose rate can change the DI amount (DI and its decay parents)."""

    def base_rates(self) -> dict[Key, float]:
        """Current rate of each candidate [s^-1]."""

    def di_atoms(self, rates: dict[Key, float]) -> np.ndarray:
        """DI atoms at `times` with the given candidate rates (others unchanged)."""


def port_key(st, nuclide: Nuclide) -> Key:
    """ISOTOPIA key of a nuclide (Z, A, I): (Z, A, -1) unless the nuclide has isomers."""
    z, a, i = nuclide
    return (z, a, i) if st.Nisomer[(z, a)] >= 0 else (z, a, -1)


class PortSolver:
    """The port's own kinetics (kinetics.equations, ISOTOPIA's analytic solution).

    The port feeds a product from the target and from one beta parent (Z-1 or Z+1, same
    A), with the defects listed in equations_f90_findings.md. Uncertainties computed here
    belong to the numbers the port reports.

    How the rerun works: kinetics.equations recomputes every inventory from
    st.reaction_rate. We change one rate, run it, read the DI, and in the end restore the
    original rates and run it once more, which restores the State exactly. Only
    single-isotope targets are supported (a natural target keeps running sums).
    """

    def __init__(self, st, di: Nuclide):
        from .kinetics import equations
        if st.flagnatural:
            raise NotImplementedError("natural targets: call the uncertainty inside the "
                                      "isotope loop of run.natural (not implemented yet)")
        self.st, self._equations = st, equations
        self.di = port_key(st, di)
        self.times = np.asarray(st.Tgrid[: NUMTIME + 1], dtype=float)
        self.eoi = st.Ntime

    def candidates(self) -> list[Key]:
        st, (z, a, _) = self.st, self.di
        keys = [self.di]
        if st.flagdecay:
            # Same parent rule as kinetics._equations: beta- parent at Z-1, beta+ at Z+1
            for isob, rt in ((-1, 1), (1, 2)):
                for is_p in range(-1, st.Nisomer[(z + isob, a)] + 1):
                    pkey = (z + isob, a, is_p)
                    if st.rtyp[pkey] == rt:
                        keys.append(pkey)
        return [k for k in keys if float(st.reaction_rate[k]) > 0]

    def base_rates(self) -> dict[Key, float]:
        return {k: float(self.st.reaction_rate[k]) for k in self.candidates()}

    def di_atoms(self, rates: dict[Key, float]) -> np.ndarray:
        st = self.st
        saved = {k: st.reaction_rate[k] for k in rates}
        try:
            for k, r in rates.items():
                st.reaction_rate[k] = np.float32(r)
            self._equations(st)
            return np.asarray(st.Niso[self.di][: NUMTIME + 1], dtype=float).copy()
        finally:
            for k, r in saved.items():
                st.reaction_rate[k] = r
            self._equations(st)          # back to the original inventories


class MatrixSolver:
    """Nate's matrix solver (irradiation.irradiate), when that branch is available.

    Every decay branch feeds the DI, so the candidates are the DI and all its ancestors
    in the decay network. One rate of Irradiation.rates is changed per rerun with
    dataclasses.replace; nothing in Nate's code is edited.
    """

    def __init__(self, st, di: Nuclide, times: Iterable[float] | None = None):
        from dataclasses import replace

        from .irradiation import Irradiation, irradiate, isotopia_times  # feature/matrix-solver
        self.st, self.di, self._replace, self._irradiate = st, di, replace, irradiate
        self.irr = Irradiation.from_state(st)
        self.times = np.asarray(list(times) if times is not None else isotopia_times(st), float)
        self.eoi = int(np.searchsorted(self.times, self.irr.t_irr, side="right") - 1)
        self._ancestors = self._decay_ancestors()

    def _decay_ancestors(self) -> set[Nuclide]:
        """The DI and every nuclide that decays into it, from the decay matrix."""
        from .decaylib import DecayLibrary
        from .matrix import build_chain
        net = build_chain([self.irr.target, *self.irr.rates], DecayLibrary(), "warn")
        A = np.asarray(net.A)
        if self.di not in net.index:
            return {self.di}
        found, todo = {self.di}, [self.di]
        while todo:                              # walk the decay network upstream
            i = net.index[todo.pop()]
            for j, parent in enumerate(net.nuclides):
                if j != i and A[i, j] > 0 and parent not in found:
                    found.add(parent)
                    todo.append(parent)
        return found

    def candidates(self) -> list[Key]:
        return [port_key(self.st, n) for n in self.irr.rates if n in self._ancestors]

    def base_rates(self) -> dict[Key, float]:
        return {port_key(self.st, n): r for n, r in self.irr.rates.items()
                if n in self._ancestors}

    def di_atoms(self, rates: dict[Key, float]) -> np.ndarray:
        new = dict(self.irr.rates)
        for n in new:
            if port_key(self.st, n) in rates:
                new[n] = rates[port_key(self.st, n)]
        inv = self._irradiate(self._replace(self.irr, rates=new), self.times)
        return np.asarray(inv[self.di], dtype=float)


@dataclass
class ChannelResult:
    """One channel's part in the DI uncertainty (one row of the ranking table)."""

    rate: ChannelRate
    sensitivity: np.ndarray        # s_c(t) = dlnN/dlnR_c at each output time
    variance_share: np.ndarray     # s_c^2 (dR/R)^2 / (dN/N)^2 at each time (rho_cc' = 0)


@dataclass
class DIUncertainty:
    """Cross-section uncertainty of the DI amount for one route.

    Attributes
    ----------
    route : str
        Description, e.g. "p + Mo100, 16 -> 12 MeV".
    di : Nuclide
    times : ndarray
        Output times [s]; `eoi` indexes the end of irradiation, the last entry is the end
        of cooling.
    N : ndarray
        DI atoms at `times`. Activity and mass have the same relative uncertainty.
    rel : ndarray
        1-sigma relative uncertainty of N at `times` (cross sections only).
    channels : list of ChannelResult
        Sorted by variance share at EOI, largest first.
    source_share : dict of str to float
        Share of the DI production (at EOI) coming from each data source.
    settings : Settings
    warnings : list of str
    lower_bound : bool
        True if some channel had no uncertainty available (TENDL default not set).
    """

    route: str
    di: Nuclide
    times: np.ndarray
    eoi: int
    N: np.ndarray
    rel: np.ndarray
    channels: list[ChannelResult]
    source_share: dict[str, float]
    settings: Settings
    warnings: list[str]
    lower_bound: bool


def di_uncertainty(st, di: Nuclide, solver: str | Solver = "port",
                   settings: Settings | None = None) -> DIUncertainty:
    """Cross-section uncertainty of the DI amount for the route the port has just run.

    Parameters
    ----------
    st : State
        Port state after isotopia_py.run.run (single-isotope target).
    di : Nuclide
        Desired isotope (Z, A, I), e.g. (43, 99, 1) for Tc-99m.
    solver : "port", "matrix" or a Solver
        Kinetics used for the sensitivities. Use the one that produces the numbers you
        report: "port" (default) or "matrix" (Nate's solver, feature/matrix-solver).
    settings : Settings, optional
        Default: read from xs_uncertainty_defaults.json.

    Returns
    -------
    DIUncertainty
    """
    settings = settings or Settings.load()
    if st.rate_grid is None:
        raise RuntimeError("the State has no rate_grid: run the port with this branch's rates.py")
    if isinstance(solver, str):
        solver = {"port": PortSolver, "matrix": MatrixSolver}[solver](st, di)

    # --- Rates of the channels that can feed the DI, with their uncertainties ---------
    base = solver.base_rates()
    rates = {k: rate_uncertainty(st, k, settings) for k in base}
    warnings = [w for cr in rates.values() for w in cr.channel.warnings]
    lower_bound = any("LOWER BOUND" in w for w in warnings)

    # --- Sensitivities by central finite differences -----------------------------------
    # s_c = [N(R_c(1+h)) - N(R_c(1-h))] / (2 h N): the % change of N per % change of R_c.
    N0 = solver.di_atoms(base)
    h = settings.fd_step
    sens: dict[Key, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, r in base.items():
            up = solver.di_atoms({**base, k: r * (1 + h)})
            down = solver.di_atoms({**base, k: r * (1 - h)})
            sens[k] = np.where(N0 > 0, (up - down) / (2 * h * N0), 0.0)

    # --- Combine: (dN/N)^2 = sum_c sum_c' s_c s_c' rho_cc' r_c r_c' ---------------------
    keys = list(base)
    r = np.array([rates[k].rel for k in keys])                   # dR/R per channel
    S = np.array([sens[k] for k in keys]) if keys else np.zeros((0, len(N0)))
    rho_c = np.full((len(keys), len(keys)), settings.channel_correlation)  # USER INPUT
    np.fill_diagonal(rho_c, 1.0)
    u = S * r[:, None]                                             # s_c r_c at each time
    var = np.einsum("it,ij,jt->t", u, rho_c, u)                   # the double sum, per time
    rel = np.sqrt(np.maximum(var, 0.0))

    # --- Ranking and data-quality summary ------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        results = [ChannelResult(rates[k], sens[k], np.where(var > 0, u[i] ** 2 / var, 0.0))
                   for i, k in enumerate(keys)]
    results.sort(key=lambda c: -c.variance_share[solver.eoi])
    total = sum(abs(c.sensitivity[solver.eoi]) for c in results) or 1.0
    share = {s: sum(abs(c.sensitivity[solver.eoi]) * c.rate.source_share[s] for c in results)
             / total for s in SOURCES}
    if not keys or N0[solver.eoi] <= 0:
        warnings.append(f"{_label(di)}: not produced in this route; no uncertainty computed")

    route = (f"{PARSYM[st.k0]} + {st.targetnuclide.strip()}, "
             f"{float(st.Ebeam):g} -> {float(st.Eback):g} MeV" if st.k0 > 1 else
             f"{PARSYM[st.k0]} + {st.targetnuclide.strip()}")
    return DIUncertainty(route, di, solver.times, solver.eoi, N0, rel, results, share,
                         settings, warnings, lower_bound)


# =====================================================================================
# 5. Report
# =====================================================================================

def report(res: DIUncertainty) -> str:
    """Plain-text report: DI amount +- uncertainty, data quality, ranking, warnings."""
    e, end = res.eoi, len(res.times) - 1

    def pct(x: float) -> str:
        return f"{100 * x:6.2f}%"

    lines = [
        f"Cross-section uncertainty of {_label(res.di)}: {res.route}",
        "  Included: residual-production cross sections only. Not included: decay data,",
        "  stopping power, burn-up (non-elastic) cross section.",
        f"  Energy correlation: {res.settings.energy_correlation}"
        + (f" (L = {res.settings.correlation_length:g} MeV)"
           if res.settings.energy_correlation == "exponential" else "")
        + f"; channel correlation: {res.settings.channel_correlation:g}",
        f"  DI atoms at EOI        {res.N[e]:12.5e}  +/- {pct(res.rel[e])}",
        f"  DI atoms after cooling {res.N[end]:12.5e}  +/- {pct(res.rel[end])}",
        "  (activity and mass have the same relative uncertainty)",
        "  Data quality (share of DI production): " + " | ".join(
            f"{s} {100 * v:.0f}%" for s, v in res.source_share.items() if v > 5e-4),
    ]
    if res.lower_bound:
        lines.append("  ** LOWER BOUND: some channels have no uncertainty (see warnings) **")
    # Ranking table. Sensitivity = % change of the DI per % change of that rate; for
    # production channels it is close to the channel's share of the DI.
    lines += ["", f"  {'Channel':24s}{'Source':15s}{'dR/R':>8s}{'Sens. EOI':>11s}"
                  f"{'Var. EOI':>10s}{'Sens. cooled':>14s}{'Var. cooled':>13s}"]
    for c in res.channels:
        main = max(c.rate.source_share, key=c.rate.source_share.get)
        lines.append(f"  {'-> ' + _label(c.rate.channel.key):24s}{main:15s}"
                     f"{100 * c.rate.rel:7.2f}%{c.sensitivity[e]:11.4f}"
                     f"{100 * c.variance_share[e]:9.1f}%{c.sensitivity[end]:14.4f}"
                     f"{100 * c.variance_share[end]:12.1f}%")
    if res.warnings:
        lines += ["", "  Warnings"] + [f"  - {w}" for w in res.warnings]
    return "\n".join(lines) + "\n"


def _label(key) -> str:
    """Tc99_m1 style label for (Z, A, is) or (Z, A, I); total (is = -1) prints as Tc99."""
    z, a, i = key
    return f"{nuc(z).strip()}{a}" + (f"_m{i}" if i > 0 else "")
