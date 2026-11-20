"""Tests for xs_uncertainty.py and route_compare.py.

The first group uses small made-up cross-section files, so every expected value is known
exactly. The last group runs the p-Mo100-Tc099m sample and needs the library (skipped
without it, like tests/test_samples.py).
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isotopia_py import xs_uncertainty as xu
from isotopia_py.route_compare import compare_routes

# ------------------------------------------------------------------------------------
# Helpers: a fake port State and fake library files
# ------------------------------------------------------------------------------------

def settings(model="full", tendl=0.3, L=2.0):
    return xu.Settings(tendl_relative={"charged": tendl, "neutron": tendl, "photon": tendl},
                       energy_correlation=model, correlation_length=L)


def write_xs(path: Path, E, xs, dxs=None, library="iaea.2024 + TENDL-2023"):
    """A YANDF-like table with 2 (TENDL) or 3 (IAEA) columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# header:\n#   library: {library}\n# datablock:\n")
        for i, e in enumerate(E):
            row = f"{e:.6E} {xs[i]:.6E}" + (f" {dxs[i]:.6E}" if dxs is not None else "")
            fh.write(row + "\n")


def fake_state(tmp_path: Path, E_grid, w, scale=1e27, rates=None):
    """Just the State fields xs_uncertainty reads. scale = 1e27 makes R = sum(w * sigma)."""
    (tmp_path / "residual").mkdir(exist_ok=True)
    return SimpleNamespace(
        rate_grid=xu.RateGrid(2, np.asarray(E_grid, float), np.asarray(w, float), scale),
        xsfile={}, k0=2, nuclide="Mo100", libname="test", xspath=str(tmp_path) + "/",
        targetnuclide="Mo100", reaction_rate=defaultdict(float, rates or {}))


def lib(tmp_path, key):
    """Path of the library file the code will look for, for product key (Z, A, is)."""
    z, a, i = key
    ZA = f"{z:03d}{a:03d}" + {0: "g", 1: "m", 2: "n"}.get(i, "")
    return tmp_path / "residual" / f"p-Mo100-rp{ZA}.test"


KEY = (43, 99, 1)


# ------------------------------------------------------------------------------------
# Rate uncertainty of one channel: the sandwich rule against known answers
# ------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [10, 100])
def test_full_correlation_gives_pointwise_relative_uncertainty(tmp_path, n):
    """Constant sigma and dsigma/sigma = f: with full correlation dR/R = f on any grid."""
    E = np.linspace(10, 20, 11)
    write_xs(lib(tmp_path, KEY), E, np.full(11, 100.0), np.full(11, 5.0))
    grid = np.linspace(10.25, 19.75, n)
    st = fake_state(tmp_path, grid, np.ones(n))
    cr = xu.rate_uncertainty(st, KEY, settings("full"))
    assert cr.rel == pytest.approx(0.05, rel=1e-12)


def test_no_correlation_shrinks_with_the_grid(tmp_path):
    """Uncorrelated points, uniform weights: dR/R = f / sqrt(N). Grid-dependent, so a diagnostic."""
    n = 25
    E = np.linspace(10, 20, 11)
    write_xs(lib(tmp_path, KEY), E, np.full(11, 100.0), np.full(11, 5.0))
    st = fake_state(tmp_path, np.linspace(10.2, 19.8, n), np.ones(n))
    cr = xu.rate_uncertainty(st, KEY, settings("none"))
    assert cr.rel == pytest.approx(0.05 / np.sqrt(n), rel=1e-12)


def test_monte_carlo_agrees_with_sandwich_rule(tmp_path):
    """Sample sigma from its covariance; the spread of R must match sqrt(a^T rho a)."""
    E = np.linspace(10, 20, 11)
    xs = 50 + 30 * np.sin(E / 3)
    write_xs(lib(tmp_path, KEY), E, xs, 0.08 * xs)
    n = 40
    grid, w = np.linspace(10.1, 19.9, n), np.linspace(1, 2, n)
    s = settings("exponential", L=3.0)
    st = fake_state(tmp_path, grid, w)
    cr = xu.rate_uncertainty(st, KEY, s)
    ch = cr.channel
    cov = np.outer(ch.dsigma, ch.dsigma) * s.rho(grid, grid)
    draws = np.random.default_rng(1).multivariate_normal(ch.sigma, cov, size=20000)
    R = draws @ w
    assert R.std() == pytest.approx(cr.dR, rel=0.03)


def test_self_check_reproduces_rate(tmp_path):
    """The rebuilt rate is sum(w * sigma) and matches the 'port' rate without warning."""
    E = np.linspace(10, 20, 11)
    write_xs(lib(tmp_path, KEY), E, np.full(11, 100.0), np.full(11, 5.0))
    st = fake_state(tmp_path, [12.0, 14.0], [1.0, 2.0], rates={KEY: 300.0})
    cr = xu.rate_uncertainty(st, KEY, settings())
    assert cr.R == pytest.approx(300.0)
    assert not cr.channel.warnings


# ------------------------------------------------------------------------------------
# Decision table: which uncertainty each energy point gets
# ------------------------------------------------------------------------------------

def test_iaea_extended_region_uses_tendl_default(tmp_path):
    """Points where the file's dxs is 0 (beyond the IAEA range) get the TENDL default."""
    E = np.arange(10.0, 31.0)
    dxs = np.where(E <= 20, 5.0, 0.0)
    write_xs(lib(tmp_path, KEY), E, np.full(E.size, 100.0), dxs)
    st = fake_state(tmp_path, [15.0, 25.0], [1.0, 1.0])
    ch = xu.load_channel(st, KEY, settings(tendl=0.3))
    assert list(ch.source) == [xu.IAEA, xu.IAEA_EXTENDED]
    assert ch.dsigma == pytest.approx([5.0, 30.0])
    assert any("beyond the IAEA" in w for w in ch.warnings)


def test_tendl_file_and_missing_default_is_lower_bound(tmp_path):
    """A 2-column file is TENDL; with no default set it adds nothing and is flagged."""
    E = np.linspace(10, 20, 11)
    write_xs(lib(tmp_path, KEY), E, np.full(11, 100.0), library="tendl.2025")
    st = fake_state(tmp_path, [15.0], [1.0])
    ch = xu.load_channel(st, KEY, settings(tendl=None))
    assert ch.source[0] == xu.TENDL and ch.dsigma[0] == 0.0
    assert any("LOWER BOUND" in w for w in ch.warnings)


def test_defaults_file_loads():
    s = xu.Settings.load()
    assert s.energy_correlation in ("full", "none", "exponential")
    assert set(s.tendl_relative) == {"charged", "neutron", "photon"}


# ------------------------------------------------------------------------------------
# Sensitivities and the DI combination, with a fake solver
# ------------------------------------------------------------------------------------

class LinearSolver:
    """N(t) = sum_c coef_c(t) * R_c: known sensitivities s_c = coef_c R_c / N."""

    def __init__(self, coefs: dict, rates: dict):
        self.coefs, self.rates = coefs, rates
        self.times, self.eoi = np.array([0.0, 1.0, 2.0]), 1

    def candidates(self):
        return list(self.rates)

    def base_rates(self):
        return dict(self.rates)

    def di_atoms(self, rates):
        return sum(np.asarray(self.coefs[k]) * r for k, r in rates.items())


def test_sensitivities_and_combination(tmp_path):
    """Two channels: s_c is each one's share, and (dN/N)^2 = sum (s_c r_c)^2."""
    parent = (42, 99, -1)
    E = np.linspace(10, 20, 11)
    write_xs(lib(tmp_path, KEY), E, np.full(11, 100.0), np.full(11, 5.0))      # 5%
    write_xs(lib(tmp_path, parent), E, np.full(11, 100.0), np.full(11, 20.0))  # 20%
    st = fake_state(tmp_path, [15.0], [1.0])
    st.Ebeam = st.Eback = 0.0
    solver = LinearSolver({KEY: [0, 3.0, 3.0], parent: [0, 1.0, 1.0]},
                          {KEY: 100.0, parent: 100.0})
    res = xu.di_uncertainty(st, (43, 99, 1), solver, settings())
    s = {c.rate.channel.key: c.sensitivity[1] for c in res.channels}
    assert s[KEY] == pytest.approx(0.75) and s[parent] == pytest.approx(0.25)
    assert res.rel[1] == pytest.approx(np.hypot(0.75 * 0.05, 0.25 * 0.20))
    assert res.channels[0].rate.channel.key == parent      # ranked by variance share


# ------------------------------------------------------------------------------------
# Route comparison: shared-error cancellation
# ------------------------------------------------------------------------------------

def _route(tmp_path, grid, coef):
    st = fake_state(tmp_path, grid, np.ones(len(grid)))
    st.Ebeam = st.Eback = 0.0
    return xu.di_uncertainty(st, (43, 99, 1), LinearSolver({KEY: [0, coef, coef]},
                                                           {KEY: 100.0}), settings("full"))


def test_same_curve_cancels_under_full_correlation(tmp_path):
    """Constant dsigma/sigma: the error is a pure scale factor, so the ratio is exact."""
    E = np.linspace(10, 30, 21)
    write_xs(lib(tmp_path, KEY), E, np.full(21, 100.0), np.full(21, 5.0))
    a = _route(tmp_path, np.linspace(12, 16, 9), 1.0)
    b = _route(tmp_path, np.linspace(20, 24, 9), 1.5)
    cmp = compare_routes(a, b, models=("full",))
    assert cmp.sd == pytest.approx(0.0, abs=1e-9)
    assert cmp.verdict == "B better" and cmp.shared == ["Tc99_m1"]


def test_comparison_keeps_the_more_cautious_model(tmp_path):
    """With a finite correlation length the ratio is less certain; that value is used."""
    E = np.linspace(10, 30, 21)
    write_xs(lib(tmp_path, KEY), E, np.full(21, 100.0), np.full(21, 5.0))
    a = _route(tmp_path, np.linspace(12, 16, 9), 1.0)
    b = _route(tmp_path, np.linspace(20, 24, 9), 1.02)
    cmp = compare_routes(a, b)
    assert cmp.sd == cmp.sd_by_model["exponential"] > cmp.sd_by_model["full"]
    assert cmp.verdict == "tie"


# ------------------------------------------------------------------------------------
# The p-Mo100-Tc099m sample (needs the library)
# ------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
ISOTOPIA = Path(os.environ.get("ISOTOPIA_DIR") or ROOT.parent / "isotopia")
SAMPLE = ISOTOPIA / "samples" / "p-Mo100-Tc099m" / "org" / "isotopia.inp"
LIBS = Path(os.environ.get("ISOTOPIA_LIBS") or ISOTOPIA.parent / "isotopia.libs")
needs_library = pytest.mark.skipif(not (SAMPLE.is_file() and (LIBS / "p" / "Mo100").is_dir()),
                                   reason="needs ISOTOPIA samples and isotopia.libs/p/Mo100")


@needs_library
def test_mo100_sample(tmp_path):
    """Rates rebuilt exactly; Tc-99m within the IAEA band; inventories left unchanged."""
    from isotopia_py.run import run
    with open(tmp_path / "out.txt", "w") as out:
        st = run(SAMPLE.read_text().splitlines(), out=out, outdir=str(tmp_path))
    before = np.array(st.Niso[(43, 99, 1)], copy=True)
    res = xu.di_uncertainty(st, (43, 99, 1), "port", settings())
    tc = next(c for c in res.channels if c.rate.channel.key == (43, 99, 1))
    assert tc.rate.R == pytest.approx(tc.rate.R_port, rel=1e-5)
    assert tc.rate.source_share[xu.IAEA] == pytest.approx(1.0)
    assert 0.04 < res.rel[res.eoi] < 0.05                 # IAEA dxs is ~4.6% at 12-16 MeV
    assert np.array_equal(st.Niso[(43, 99, 1)], before)   # the State is restored
