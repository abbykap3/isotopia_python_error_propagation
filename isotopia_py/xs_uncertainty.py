"""Cross-section uncertainty of the desired isotope (DI) for one ISOTOPIA route.

Not in the Fortran. This module answers one question for a route that the port has
already run: by how much could the DI amount be wrong because the cross sections are
uncertain? The answer is used to compare routes (route_compare.py), not to certify an
absolute yield. The write-up docs/xs_uncertainty.md explains each step in detail.

Overview

1. Rate grid       ISOTOPIA computes every reaction rate as a weighted sum over an energy
                   grid (manual Eq. 4.20 for charged particles, Eq. 4.11 for neutrons and
                   photons):  R = scale * sum_k w_k * sigma_k * 1e-27.
                   The grid and weights come from daughters.rate_grid (Nate's module),
                   which rebuilds them from the port's own functions and is checked against
                   the port's rates (daughters.check_rate_grid). Nothing in rates.py changes.
2. XSChannel       For one channel (hit nuclide -> product) we read the library file,
                   including the 'dxs' column the port ignores, and decide the 1-sigma
                   uncertainty of every grid point from its data source:
                     IAEA            the evaluation's own dxs
                     IAEA-extended   dxs = 0 in the file (TENDL scaled to the IAEA end
                                     point): the TENDL default is used instead
                     TENDL           no uncertainty in the file: the TENDL default
                     user            an 'xsfile' file: its dxs column if it has one
3. ChannelRate     The uncertainty of R follows from the "sandwich rule"
                   var(R) = (a)^T rho (a), a_k = scale * w_k * dsigma_k * 1e-27, exact
                   because R is a linear function of sigma. rho is the energy correlation.
4. DIUncertainty   The DI amount N depends on several rates: direct production, decay of a
                   parent, and with double hitting reactions on products. Its sensitivity
                   to each rate, s_c = dlnN / dlnR_c, is found by rerunning the solver with
                   one rate scaled (finite differences), so any solver can be used:
                   (dN/N)^2 = sum_c sum_c' s_c s_c' rho_cc' (dR_c/R_c) (dR_c'/R_c').

Three solvers are supported (argument `solver` of di_uncertainty):
    "port"     the port's analytic equations (kinetics.py), target reactions only
    "matrix"   Nate's matrix solver (irradiation.irradiate), target reactions only
    "hits"     Nate's matrix solver with every product hit too (daughters.py); needs a Window

Units: energies MeV, cross sections mb in the library files (converted with the factor
1e-27 exactly where the port does it), rates s^-1, times s, amounts atoms.

Inputs the team must set are in xs_uncertainty_defaults.json, next to this file; the
places that use them are marked "USER INPUT".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .constants import NUMTIME, PARSYM, nuc
from .daughters import RateGrid, Window, _residual_files, check_rate_grid, rate_grid

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
        together; the cautious default for one route), "none" (independent points;
        diagnostic only, it depends on the grid spacing), or "exponential"
        (rho = exp(-|E1-E2|/L)).
    correlation_length : float
        USER INPUT. L for the exponential model [MeV].
    channel_correlation : float
        USER INPUT. Correlation between different channels, 0 (independent) to 1.
    comparison_sigma : float
        USER INPUT. Route A beats route B when ln(N_A/N_B) exceeds this many standard
        deviations; otherwise the routes are tied.
    fd_step : float
        Relative change of a rate used for finite-difference sensitivities.
    selfcheck_tolerance : float
        Allowed relative difference between our rebuilt rate and the solver's rate.
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
        settings = cls(**{k: v for k, v in raw.items() if not k.startswith("_")})
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
        dE = np.abs(E1[:, None] - E2[None, :])      # table of all differences, no loop
        if model == "full":
            return np.ones_like(dE)
        if model == "none":
            return (dE < 1e-9).astype(float)       # only identical energies are correlated
        return np.exp(-dE / self.correlation_length)


# =====================================================================================
# 1. Which channels exist, and where their data are
# =====================================================================================

@dataclass(frozen=True)
class ChannelSpec:
    """One reaction channel a solver can change: `hit` -> `product`.

    Attributes
    ----------
    hit : Nuclide
        The nuclide hit by the beam (the target, or a product when double hitting).
    product : Nuclide
        The product (Z, A, I).
    path : str
        Library file with its cross section.
    R_solver : float
        The rate the solver uses for this channel [s^-1 per atom of `hit`], for the
        self-check.
    user_file : bool
        True if the file came from the 'xsfile' keyword.
    """

    hit: Nuclide
    product: Nuclide
    path: str
    R_solver: float
    user_file: bool = False

    @property
    def name(self) -> str:
        return f"{_label(self.hit)} -> {_label(self.product)}"


def port_key(st, nuclide: Nuclide) -> Key:
    """ISOTOPIA key of a nuclide (Z, A, I): (Z, A, -1) unless the nuclide has isomers."""
    z, a, i = nuclide
    return (z, a, i) if st.Nisomer[(z, a)] >= 0 else (z, a, -1)


def target_channel(st, key: Key) -> ChannelSpec:
    """The target -> product channel for ISOTOPIA key `key`, as data.crosssections reads it.

    An 'xsfile' keyword overrides the library file.
    """
    iz, ia, is_ = key
    override = st.xsfile.get(key, "").strip()
    if override:
        path, user = override, True
    else:
        ZA = f"{iz:03d}{ia:03d}" + {0: "g", 1: "m", 2: "n"}.get(is_, "")
        name = f"{PARSYM[st.k0]}-{st.nuclide}-rp{ZA}.{st.libname}"
        path, user = os.path.join(st.xspath, "residual", name), False
    target = (st.Ztarget, st.Atarget, 0)
    return ChannelSpec(target, (iz, ia, max(is_, 0)), path, float(st.reaction_rate[key]), user)


# =====================================================================================
# 2. One channel: cross section and its uncertainty on the grid
# =====================================================================================

@dataclass
class XSChannel:
    """Cross section of one channel and its 1-sigma uncertainty on the integration grid.

    Attributes
    ----------
    spec : ChannelSpec
    library : str
        The file's 'library:' header line (e.g. "iaea.2024 + TENDL-2023", "tendl.2025").
    sigma, dsigma : ndarray
        Cross section and its uncertainty on the grid energies [mb]. Zero outside the
        file's range, exactly as the port does.
    source : ndarray of str
        Data-source tag of every grid point (IAEA, IAEA-extended, TENDL, user).
    warnings : list of str
    """

    spec: ChannelSpec
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


def _on_grid(Elib: np.ndarray, ylib: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Linear interpolation onto the grid, zero outside [Elib[0], Elib[-1]).

    Same rule as the port's rate loop (locate + pol1) and daughters.RateGrid.rate.
    """
    if len(Elib) < 2:
        return np.zeros_like(E)
    inside = (E >= Elib[0]) & (E < Elib[-1])
    return np.where(inside, np.interp(E, Elib, ylib), 0.0)


def load_channel(st, grid: RateGrid, spec: ChannelSpec, settings: Settings) -> XSChannel:
    """Cross section and 1-sigma uncertainty of one channel on the grid.

    This is where the decision table of the write-up (section 4.2) is applied, point by
    point.

    Parameters
    ----------
    st : State
        Port state after a run (for the projectile index st.k0).
    grid : RateGrid
        daughters.rate_grid(st).
    spec : ChannelSpec
    settings : Settings

    Returns
    -------
    XSChannel
    """
    header, data = read_xs_file(spec.path)
    library = header.get("library", "unknown")
    E = grid.energies
    Elib, xs = data[:, 0], data[:, 1]
    has_dxs = data.shape[1] >= 3
    sigma = _on_grid(Elib, xs, E)
    warnings: list[str] = []

    # --- Which points carry their own uncertainty? -----------------------------------
    if has_dxs:
        dxs_grid = _on_grid(Elib, data[:, 2], E)
        # dxs interpolates to a small nonzero value right next to the last evaluated
        # point; treat a point as evaluated only if both bracketing library points are.
        evaluated = _on_grid(Elib, (data[:, 2] > 0).astype(float), E) > 0.999
        outside = USER if spec.user_file else IAEA_EXTENDED
        source = np.where(evaluated, USER if spec.user_file else IAEA, outside)
    else:
        dxs_grid = np.zeros_like(sigma)
        evaluated = np.zeros(sigma.shape, dtype=bool)
        source = np.full(sigma.shape, USER if spec.user_file else TENDL)
    source = np.asarray(source, dtype=object)

    # --- Points without their own uncertainty get the TENDL default ------------------
    need_default = ~evaluated & (sigma > 0)
    if need_default.any():
        f = settings.tendl_default(st.k0)          # USER INPUT (json: tendl_relative)
        if f is None:
            warnings.append(f"{spec.name}: model-based cross section and no TENDL default set "
                            f"in {DEFAULTS_FILE.name}; this channel adds no uncertainty, so "
                            "the result is a LOWER BOUND")
            f = 0.0
        else:
            why = ("the energy window extends beyond the IAEA evaluated range"
                   if has_dxs and not spec.user_file else "model-only cross section (no dxs)")
            warnings.append(f"{spec.name}: {why}; default uncertainty of {100 * f:g}% assumed")
        dxs_grid = np.where(need_default, f * sigma, dxs_grid)

    return XSChannel(spec, library, sigma, dxs_grid, source, warnings)


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
        Rate rebuilt from the grid [s^-1 per atom hit].
    dR : float
        1-sigma uncertainty of R [s^-1].
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
    wdsigma: np.ndarray
    E: np.ndarray
    source_share: dict[str, float]

    @property
    def rel(self) -> float:
        """dR / R (0 if R = 0)."""
        return self.dR / self.R if self.R > 0 else 0.0


def rate_uncertainty(st, grid: RateGrid, spec: ChannelSpec, settings: Settings) -> ChannelRate:
    """R +- dR for one channel: manual Eq. 4.20 (4.11 for n, gamma) plus the sandwich rule.

    var(R) = sum_j sum_k a_j rho_jk a_k,  a_k = scale * w_k * dsigma_k * 1e-27.

    With full correlation (rho = 1) this is simply dR = sum_k a_k: the relative rate
    uncertainty is the rate-weighted average of the pointwise relative uncertainties.
    """
    ch = load_channel(st, grid, spec, settings)
    contrib = grid.scale * grid.weights * ch.sigma * MB_TO_CM2     # each point's share of R
    a = grid.scale * grid.weights * ch.dsigma * MB_TO_CM2
    R = float(contrib.sum())
    # a @ rho @ a is the double sum written above ('@' is matrix multiplication)
    dR = float(np.sqrt(max(a @ settings.rho(grid.energies, grid.energies) @ a, 0.0)))
    share = {s: float(contrib[ch.source == s].sum() / R) if R > 0 else 0.0 for s in SOURCES}

    # Self-check: our rebuilt rate must be the solver's rate, or the uncertainty would be
    # attached to a different number from the one reported.
    if spec.R_solver > 0 and abs(R / spec.R_solver - 1) > settings.selfcheck_tolerance:
        ch.warnings.append(f"{spec.name}: rebuilt rate {R:.6e} s^-1 differs from the solver's "
                           f"{spec.R_solver:.6e} s^-1 by more than "
                           f"{settings.selfcheck_tolerance:g}")
    return ChannelRate(ch, R, dR, a, grid.energies.copy(), share)


# =====================================================================================
# 4. From rates to the DI: sensitivities through a solver
# =====================================================================================

class Solver(Protocol):
    """What di_uncertainty needs from a kinetics solver.

    Any object with these attributes and methods works (a Protocol is an agreed list of
    methods, not a base class to inherit from).
    """

    times: np.ndarray            # output times [s]
    eoi: int                     # index of the end of irradiation in `times`
    warnings: list[str]          # the solver's own warnings (truncated chains, ...)

    def channels(self) -> list[ChannelSpec]:
        """Channels whose rate can change the DI amount."""

    def di_atoms(self, scale: dict[ChannelSpec, float]) -> np.ndarray:
        """DI atoms at `times`, with the rate of each channel in `scale` multiplied by its value."""


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
        self.warnings: list[str] = []
        self._keys: dict[ChannelSpec, Key] = {}

    def channels(self) -> list[ChannelSpec]:
        st, (z, a, _) = self.st, self.di
        keys = [self.di]
        if st.flagdecay:
            # Same parent rule as kinetics._equations: beta- parent at Z-1, beta+ at Z+1
            for isob, rt in ((-1, 1), (1, 2)):
                for is_p in range(-1, st.Nisomer[(z + isob, a)] + 1):
                    pkey = (z + isob, a, is_p)
                    if st.rtyp[pkey] == rt:
                        keys.append(pkey)
        specs = []
        for k in keys:
            if float(st.reaction_rate[k]) > 0:
                spec = target_channel(st, k)
                self._keys[spec] = k
                specs.append(spec)
        return specs

    def di_atoms(self, scale: dict[ChannelSpec, float]) -> np.ndarray:
        st = self.st
        keys = {self._keys[s]: f for s, f in scale.items()}
        saved = {k: st.reaction_rate[k] for k in keys}
        try:
            for k, f in keys.items():
                st.reaction_rate[k] = np.float32(float(saved[k]) * f)
            self._equations(st)
            return np.asarray(st.Niso[self.di][: NUMTIME + 1], dtype=float).copy()
        finally:
            for k, r in saved.items():
                st.reaction_rate[k] = r
            self._equations(st)          # back to the original inventories


class MatrixSolver:
    """Nate's matrix solver (irradiation.irradiate), optionally with double hitting.

    window = None: only the target is hit (Irradiation.from_state), as in ISOTOPIA.
    window = daughters.Window(...): every nuclide in the window is hit too
    (daughters.hit_reactions), so products can be made by second and later hits.

    The candidate channels are those whose product can reach the DI through decay or
    reactions: the DI's ancestors in the network matrix A + R. One rate is scaled per
    rerun on a copy of the reactions; nothing in Nate's code is edited. Burnup (loss)
    rates are kept fixed: the non-elastic cross section is not propagated.
    """

    def __init__(self, st, di: Nuclide, window: Window | None = None, times=None):
        from .daughters import hit_reactions
        from .decaylib import OUTSIDE, DecayLibrary
        from .irradiation import Irradiation, irradiate, isotopia_times
        from .matrix import Reactions, build_chain

        if st.flagnatural:
            raise NotImplementedError("natural targets are not implemented yet")
        self.st, self.di, self._irradiate, self._Reactions = st, di, irradiate, Reactions
        self.library = DecayLibrary()
        self.irr = Irradiation.from_state(st)
        self.warnings: list[str] = []
        if window is None:
            self.reactions = {self.irr.target: Reactions(self.irr.burnup, self.irr.rates)}
        else:
            self.reactions, self.warnings = hit_reactions(st, self.irr, window, self.library)
        self.times = np.asarray(list(times) if times is not None else isotopia_times(st), float)
        self.eoi = int(np.searchsorted(self.times, self.irr.t_irr, side="right") - 1)

        # --- The DI's ancestors: walk the network matrix upstream from the DI -----------
        net = build_chain([self.irr.target], self.library, "warn", reactions=self.reactions)
        M = np.asarray(net.A) + np.asarray(net.R)    # M[i, j] > 0: j feeds i
        found = set()
        if di in net.index:
            found, todo = {di}, [di]
            while todo:
                i = net.index[todo.pop()]
                for j in np.nonzero(M[i])[0]:
                    parent = net.nuclides[j]
                    if j != i and parent not in found:
                        found.add(parent)
                        todo.append(parent)

        # --- Channels whose product is the DI or an ancestor of it ----------------------
        self._specs: list[ChannelSpec] = []
        target = self.irr.target
        for hit, rx in self.reactions.items():
            files = None if hit == target else _residual_files(st, hit)[0]
            for product, rate in rx.products.items():
                if product == OUTSIDE or product not in found or rate <= 0:
                    continue
                if hit == target:
                    spec = target_channel(st, port_key(st, product))
                    spec = ChannelSpec(hit, product, spec.path, rate, spec.user_file)
                else:
                    spec = ChannelSpec(hit, product, files[product], rate)
                self._specs.append(spec)

    def channels(self) -> list[ChannelSpec]:
        return list(self._specs)

    def di_atoms(self, scale: dict[ChannelSpec, float]) -> np.ndarray:
        reactions = dict(self.reactions)
        for spec, f in scale.items():
            rx = reactions[spec.hit]
            products = dict(rx.products)
            products[spec.product] = products[spec.product] * f
            reactions[spec.hit] = self._Reactions(rx.loss, products)
        inv = self._irradiate(self.irr, self.times, self.library, "warn", reactions=reactions)
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
    solver : str
        Name of the solver used.
    times : ndarray
        Output times [s]; `eoi` indexes the end of irradiation, the last entry is the end
        of cooling.
    eoi : int
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
    solver: str
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
                   settings: Settings | None = None, window: Window | None = None,
                   grid: RateGrid | None = None) -> DIUncertainty:
    """Cross-section uncertainty of the DI amount for the route the port has just run.

    Parameters
    ----------
    st : State
        Port state after isotopia_py.run.run (single-isotope target).
    di : Nuclide
        Desired isotope (Z, A, I), e.g. (43, 99, 1) for Tc-99m.
    solver : "port", "matrix", "hits" or a Solver
        Kinetics used for the sensitivities; use the one that produces the numbers you
        report. "hits" is the matrix solver with double hitting and needs `window`.
    settings : Settings, optional
        Default: read from xs_uncertainty_defaults.json.
    window : daughters.Window, optional
        Network limit for solver="hits".
    grid : daughters.RateGrid, optional
        Default: daughters.rate_grid(st), checked against the port's rates.

    Returns
    -------
    DIUncertainty
    """
    settings = settings or Settings.load()
    if grid is None:
        grid = rate_grid(st)
        check_rate_grid(st, grid)           # raises if the weights do not match the port
    name = solver if isinstance(solver, str) else type(solver).__name__
    if solver == "port":
        solver = PortSolver(st, di)
    elif solver == "matrix":
        solver = MatrixSolver(st, di)
    elif solver == "hits":
        if window is None:
            raise ValueError("solver='hits' needs window=daughters.Window(...)")
        solver = MatrixSolver(st, di, window)

    # --- Rates of the channels that can feed the DI, with their uncertainties ---------
    specs = solver.channels()
    rates = {s: rate_uncertainty(st, grid, s, settings) for s in specs}
    warnings = list(solver.warnings) + [w for cr in rates.values() for w in cr.channel.warnings]
    lower_bound = any("LOWER BOUND" in w for w in warnings)

    # --- Sensitivities by central finite differences -----------------------------------
    # s_c = [N(R_c(1+h)) - N(R_c(1-h))] / (2 h N): the % change of N per % change of R_c.
    N0 = solver.di_atoms({})
    h = settings.fd_step
    sens: dict[ChannelSpec, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in specs:
            up, down = solver.di_atoms({s: 1 + h}), solver.di_atoms({s: 1 - h})
            sens[s] = np.where(N0 > 0, (up - down) / (2 * h * N0), 0.0)

    # --- Combine: (dN/N)^2 = sum_c sum_c' s_c s_c' rho_cc' r_c r_c' ---------------------
    r = np.array([rates[s].rel for s in specs])                   # dR/R per channel
    S = np.array([sens[s] for s in specs]) if specs else np.zeros((0, len(N0)))
    rho_c = np.full((len(specs), len(specs)), settings.channel_correlation)  # USER INPUT
    np.fill_diagonal(rho_c, 1.0)
    u = S * r[:, None]                                             # s_c r_c at each time
    # einsum computes sum_i sum_j u[i, t] rho[i, j] u[j, t] separately for every time t
    var = np.einsum("it,ij,jt->t", u, rho_c, u)
    rel = np.sqrt(np.maximum(var, 0.0))

    # --- Ranking and data-quality summary ------------------------------------------------
    e = solver.eoi
    with np.errstate(invalid="ignore", divide="ignore"):
        results = [ChannelResult(rates[s], sens[s], np.where(var > 0, u[i] ** 2 / var, 0.0))
                   for i, s in enumerate(specs)]
    results.sort(key=lambda c: -c.variance_share[e])
    total = sum(abs(c.sensitivity[e]) for c in results) or 1.0
    share = {src: sum(abs(c.sensitivity[e]) * c.rate.source_share[src] for c in results)
             / total for src in SOURCES}
    if not specs or N0[e] <= 0:
        warnings.append(f"{_label(di)}: not produced in this route; no uncertainty computed")

    target = st.targetnuclide.strip()
    route = (f"{PARSYM[st.k0]} + {target}, {float(st.Ebeam):g} -> {float(st.Eback):g} MeV"
             if st.k0 > 1 else f"{PARSYM[st.k0]} + {target}")
    return DIUncertainty(route, di, name, solver.times, e, N0, rel, results, share,
                         settings, warnings, lower_bound)


# =====================================================================================
# 5. Report
# =====================================================================================

def report(res: DIUncertainty) -> str:
    """Plain-text report: DI amount +- uncertainty, data quality, ranking, warnings."""
    e, end = res.eoi, len(res.times) - 1

    def pct(x: float) -> str:
        return f"{100 * x:6.2f}%"

    s = res.settings
    lines = [
        f"Cross-section uncertainty of {_label(res.di)}: {res.route} (solver: {res.solver})",
        "  Included: residual-production cross sections only. Not included: decay data,",
        "  stopping power, burnup (non-elastic) cross sections.",
        f"  Energy correlation: {s.energy_correlation}"
        + (f" (L = {s.correlation_length:g} MeV)" if s.energy_correlation == "exponential"
           else "") + f"; channel correlation: {s.channel_correlation:g}",
        f"  DI atoms at EOI        {res.N[e]:12.5e}  +/- {pct(res.rel[e])}",
        f"  DI atoms after cooling {res.N[end]:12.5e}  +/- {pct(res.rel[end])}",
        "  (activity and mass have the same relative uncertainty)",
        "  Data quality (share of DI production): " + " | ".join(
            f"{src} {100 * v:.0f}%" for src, v in res.source_share.items() if v > 5e-4),
    ]
    if res.lower_bound:
        lines.append("  ** LOWER BOUND: some channels have no uncertainty (see warnings) **")

    # Ranking table. Sensitivity = % change of the DI per % change of that rate; for
    # production channels it is close to the channel's share of the DI.
    lines += ["", f"  {'Channel':28s}{'Source':15s}{'dR/R':>8s}{'Sens. EOI':>11s}"
                  f"{'Var. EOI':>10s}{'Sens. cooled':>14s}{'Var. cooled':>13s}"]
    shown = [c for c in res.channels
             if max(abs(c.sensitivity[e]), abs(c.sensitivity[end])) >= 1e-4]
    for c in shown:
        main = max(c.rate.source_share, key=c.rate.source_share.get)
        lines.append(f"  {c.rate.channel.spec.name:28s}{main:15s}"
                     f"{100 * c.rate.rel:7.2f}%{c.sensitivity[e]:11.4f}"
                     f"{100 * c.variance_share[e]:9.1f}%{c.sensitivity[end]:14.4f}"
                     f"{100 * c.variance_share[end]:12.1f}%")
    if len(shown) < len(res.channels):
        lines.append(f"  ... and {len(res.channels) - len(shown)} more channel(s) with "
                     "sensitivity below 1e-4 at both times (negligible)")
    if res.warnings:
        lines += ["", "  Warnings"] + [f"  - {w}" for w in res.warnings]
    return "\n".join(lines) + "\n"


def _label(key) -> str:
    """Tc99_m1 style label for (Z, A, is) or (Z, A, I); total (is = -1) prints as Tc99."""
    z, a, i = key
    return f"{nuc(z).strip()}{a}" + (f"_m{i}" if i > 0 else "")
