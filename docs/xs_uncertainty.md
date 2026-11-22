# Cross-section uncertainty propagation (MIDAS extension 6)

Branch `feature/xs-uncertainty`, up to date with `main` at `683df9e` (matrix solver with double hitting, and layered targets, merged). Author: Abby, with Claude.

This extension attaches a cross-section uncertainty to the amount of the desired isotope
(DI) produced by a route, and uses it to decide whether one route really beats another.
It is built for **comparing routes**, not for certifying an absolute yield.

Contents

1. [Quick start](#1-quick-start)
2. [Inputs you must give](#2-inputs-you-must-give) (read this first)
3. [Theory](#3-theory)
4. [How each data source is handled, and why](#4-how-each-data-source-is-handled-and-why)
5. [Implementation](#5-implementation)
6. [Verification and results](#6-verification-and-results)
7. [How it fits the other extensions](#7-how-it-fits-the-other-extensions)
8. [Limitations and next steps](#8-limitations-and-next-steps)
9. [References](#9-references)

---

## 1. Quick start

```python
from isotopia_py.run import run
from isotopia_py import xs_uncertainty as xu
from isotopia_py.route_compare import compare_routes

lines = ["projectile p", "element Mo", "mass 100", "Ebeam 16.", "Eback 12.", "Ibeam 0.15"]
st = run(lines, outdir="results")                    # the normal port run, unchanged
res = xu.di_uncertainty(st, (43, 99, 1))             # DI = Tc-99m, key (Z, A, I)
print(xu.report(res))                                # amount +- %, ranking, warnings

# Compare with a second route
st_b = run([...], outdir="results_b")
print(compare_routes(res, xu.di_uncertainty(st_b, (43, 99, 1))))
```

Or run the example: `python scripts/xs_uncertainty_example.py --tendl 0.3`
(`--solver matrix` uses Nate's matrix solver; `--solver hits` adds double hitting).

With double hitting (every product inside a window around the target is hit too):

```python
from isotopia_py.daughters import Window
res = xu.di_uncertainty(st, (43, 99, 1), solver="hits",
                        window=Window(A_below=2, A_above=1, Z_below=1, Z_above=1))
```

Survey which library files carry their own uncertainty:
`python scripts/survey_xs_uncertainty.py --projectile p --target Mo100 > survey.csv`

---

## 2. Inputs you must give

These are physics or team decisions. The code runs without them, but the results are
flagged. All of them live in **`isotopia_py/xs_uncertainty_defaults.json`**; in the code
the places that use them are marked `USER INPUT`.

| # | Setting (JSON key) | Current value | What happens if not set | Who decides |
|---|---|---|---|---|
| 1 | `tendl_relative` (charged, neutron, photon) | `null` | TENDL channels add no uncertainty; results are flagged **LOWER BOUND** | You, with the team |
| 2 | `channel_correlation` | `0.0` | Different channels are independent | You |
| 3 | `comparison_sigma` | `2.0` | A route "wins" only beyond 2 standard deviations | You with Rome |
| 4 | `correlation_length` (MeV) | `2.0` | Used for the exponential correlation model in route comparisons | You |
| 5 | Solver: `di_uncertainty(..., solver="port", "matrix" or "hits")` | `"port"` | Uncertainties follow the port's kinetics | Team, with Nate |
| 7 | `window` for `solver="hits"` (`daughters.Window`) | none | Required for double hitting | Same window as the route search uses |
| 6 | Natural targets (`mass 0`) | not supported | `NotImplementedError` | Team: needed or not? |

### How to find the answers

**1. TENDL default (the most important number).** This is the 1-sigma relative
uncertainty given to every cross section that has no uncertainty of its own: all TENDL
channels, and IAEA files outside their evaluated range. For most MIDAS routes it will
dominate the result, because in our library sample only 3 of 474 residual files have an
uncertainty column (section 4.4). Ways to choose it, from quickest to best:

- *Literature anchors* (all in the Undermind workspace "PYSOTOPIA cross-section uncertainty
  propagation"): the spread of 18 TALYS model combinations gave **+-38%** on a thick-target
  yield of Sc-47 [Car20]; TALYS model-choice uncertainty was **20-26%** even on the p + Ni-58
  non-elastic cross section [Alh24]; global comparisons of TALYS with EXFOR up to 150 MeV
  [Kon08]; default model codes "largely failed" for Nb at 50-200 MeV [Fox20]. These point to
  a value in the tens of percent for charged particles, larger above about 50 MeV.
- *Check it against our own data.* `isotopia.libs` holds the raw measurements in
  `<projectile>/<target>/exfor/residual/`. For channels near your DIs that have both a TENDL
  file and EXFOR data, compute the ratio TENDL / measured at each point. The spread
  (root-mean-square of ln(ratio)) is a data-based estimate of the TENDL uncertainty. Remember
  that EXFOR data sets disagree with each other too (section 4.3), so use several.
- *Run TASMAN* for the few TENDL channels that matter most to the final ranking (section 8).
- *Neutrons* are different: thermal capture cross sections are often known to a few
  percent, so a single neutron default may be far too large. Consider a separate value.

**2. Channel correlation.** Leave 0 unless several TENDL channels of the same target feed
the DI with similar weight. To decide, run with 0 and with 1: if no verdict changes, it
does not matter for your routes.

**3. Comparison threshold.** 2 standard deviations is about 95% confidence for Gaussian
errors. Agree on it with Rome, since it decides which routes count as ties.

**4. Correlation length.** It describes how far apart in energy two points of one
cross-section curve still err together. The IAEA files give no covariance, so this is a
judgement. A physical guide is the width of the features of an excitation function (a few
MeV). Test 1, 2 and 5 MeV and check whether any verdict changes.

**5. Solver.** Uncertainties should come from the solver that produces the numbers MIDAS
reports. Ask Nate and Rome whether the route search will use the port's kinetics, the
matrix solver, or the matrix solver with double hitting ("hits").

**7. Window.** With `solver="hits"`, use exactly the `Window` the route search uses for
that route, so the uncertainty describes the same network as the reported number.

**6. Natural targets.** The port runs one isotope at a time and keeps only the last in the
State. Supporting natural targets means computing the uncertainty inside `run.natural`'s
loop and combining isotopes by abundance. Decide with Rome whether the search needs it.

---

## 3. Theory

### 3.1 What is propagated

Only the uncertainty of the **residual-production cross sections**. Not included: decay
data (half-lives, branching), stopping power, and the non-elastic cross section that sets
target burn-up. Every report states this.

An uncertainty "x +- 5%" is one standard deviation (1 sigma): if the error is Gaussian, the
true value is within +-5% about 68% of the time.

### 3.2 The reaction rate is a weighted sum of cross sections

The port computes every rate on an energy grid (`rates._reactionrates`). Nate's
`daughters.rate_grid` rebuilds that grid and its weights from the port's own functions,
and `daughters.check_rate_grid` confirms they reproduce every rate of the run. This
extension uses that grid, so it changes nothing inside the port:

$$R_c = s \sum_k w_k\,\sigma_{c,k}\times10^{-27}$$

| Projectile | Grid E_k | Weight w_k | Scale s | Manual |
|---|---|---|---|---|
| p, d, t, h, a | 100 midpoints between E_back and E_beam | dE / S(E_k) [cm] | n_p [cm^-3 s^-1] | Eq. 4.20-4.22 |
| neutrons | reactor spectrum groups | spectrum x self-shielding G | total flux [cm^-2 s^-1] | Eq. 4.11 |
| photons | bremsstrahlung spectrum | spectrum x dE | total flux | Eq. 4.11 |

The cross section is interpolated linearly onto the grid (`locate` + `pol1`), and zero
outside the file's energy range. The weights depend only on the beam and the target, never
on a cross section. That is the key fact: **R is exactly linear in sigma.**

### 3.3 Uncertainty of one rate: the sandwich rule

For a linear function of uncertain inputs, the variance is exact:

$$\operatorname{var}(R_c) = \sum_j\sum_k a_j\,\rho_{jk}\,a_k,\qquad a_k = s\,w_k\,\delta\sigma_k\times10^{-27}$$

delta-sigma_k is the 1-sigma uncertainty at energy k, and rho_jk is the correlation
between the errors at energies j and k. Three models are implemented:

| Model | rho_jk | Meaning | Use |
|---|---|---|---|
| `full` | 1 | the whole curve is off by one factor | default for one route (largest band) |
| `exponential` | exp(-\|E_j - E_k\| / L) | nearby energies err together, distant ones less | route comparisons |
| `none` | 1 only if j = k | independent points | diagnostic only |

With full correlation the formula reduces to delta-R = sum of a_k: the relative rate
uncertainty is the rate-weighted average of the pointwise relative uncertainties.

**Why not "none"?** Independent points average out, so the result shrinks as the grid gets
finer, although our knowledge has not changed. For p + Mo-100 -> Tc-99m at 12-16 MeV the
uncorrelated result is 0.72% on a 0.1 MeV grid and 2.1% on a 1 MeV grid, while the fully
correlated one is 4.6% on any grid. A result that depends on grid spacing is a sign of a
wrong assumption [Her08].

### 3.4 From rates to the DI

The DI amount N depends on several rates: direct production, decay of a parent, and later
reactions on products. Its sensitivity to rate c is

$$s_c = \frac{\partial \ln N}{\partial \ln R_c}$$

the % change of N for a 1% change of R_c. For a channel that only produces the DI, s_c is
close to its share of the DI. The DI uncertainty is

$$\left(\frac{\delta N}{N}\right)^2=\sum_c\sum_{c'} s_c\,s_{c'}\,\rho_{cc'}\,\frac{\delta R_c}{R_c}\,\frac{\delta R_{c'}}{R_{c'}}$$

with rho_cc' = 1 on the diagonal and `channel_correlation` elsewhere.

The sensitivities are found by **central finite differences**: rerun the kinetics with R_c
multiplied by (1 + h) and by (1 - h), h = 1e-4, and take
s_c = (N_up - N_down) / (2 h N). This works with any solver (the port's analytic equations
or Nate's matrix exponential) without knowing how it works inside, and stays correct when
burn-up or double hitting make N nonlinear in the rates. It costs two solver calls per
channel that feeds the DI.

Atoms, mass and activity share the same relative uncertainty, because mass = atoms x molar
mass / N_A and activity = lambda x atoms, and neither factor contains a cross section.

### 3.5 Comparing two routes

Routes are compared through the log of their ratio:

$$\operatorname{var}\!\left[\ln\frac{N_A}{N_B}\right]=\left(\frac{\delta N_A}{N_A}\right)^2+\left(\frac{\delta N_B}{N_B}\right)^2-2\,\mathrm{cov}_{AB}$$

The covariance is not zero when both routes use the same cross-section curve, e.g. the same
target at two energy windows. Then, for that channel,

$$\mathrm{cov}(R_c^A,R_c^B)=\sum_j\sum_k a^A_j\,\rho(E^A_j,E^B_k)\,a^B_k$$

the same sandwich rule with the two routes' weights, and
cov_AB = sum_c s_c^A s_c^B cov(R_c^A, R_c^B) / (R_c^A R_c^B).

**Why the comparison uses two correlation models.** For one route, full correlation is the
cautious choice. For a comparison on the same curve it is the *optimistic* one: it assumes
the error is a pure scale factor that cancels completely in the ratio. A curve can also be
wrong in shape. For Tc-99m at 16 -> 12 versus 20 -> 16 MeV, the standard deviation of
ln(ratio) is 0.004 with full correlation and 0.041 with the exponential model (L = 2 MeV).
`compare_routes` therefore evaluates both and keeps the larger. Verdict: A is better if
ln(N_A/N_B) > k sd, B is better if it is < -k sd, otherwise **tie**, and ties are for the
rarity, toxicity and half-life criteria to break.

### 3.6 Assumptions

- First-order propagation: errors small enough that a linear expansion around the central
  value holds. A Monte Carlo test checks this for the rate (section 6).
- Gaussian errors, so "2 standard deviations" means about 95%.
- Different channels independent unless `channel_correlation` is set.
- Nothing here changes a central value.

---

## 4. How each data source is handled, and why

### 4.1 What is in `isotopia.libs`

Each target has `<projectile>/<target>/iaea.2026/tables/residual/` (one file per product,
e.g. `p-Mo100-rp043099m.iaea.2026` = Z 43, A 99, isomer m) and `exfor/residual/` (raw
measurements). The folder is called `iaea.2026` even for pure TENDL files, so the code reads
each file's header (`# library: ...`) and its number of columns instead of trusting the
name. The port's own reader (`data.py`) keeps only the first two columns, so before this
extension `dxs` was never read.

### 4.2 Decision table

Every energy point of every channel gets a source tag and an uncertainty:

| File (header and columns) | Points | Uncertainty used | Tag | Warning |
|---|---|---|---|---|
| IAEA evaluation (`iaea.2024 + TENDL-2023`, `E xs dxs`) | dxs > 0 | the file's dxs | IAEA | none |
| same file | dxs = 0 (TENDL extension) | TENDL default x sigma | IAEA-extended | "beyond the IAEA evaluated range" |
| TENDL (`tendl.2025`, `E xs`) | all | TENDL default x sigma | TENDL | "model-only cross section" |
| user file (`xsfile` keyword) | all | its dxs column if present, else TENDL default | user | if the default is used |
| EXFOR (`exfor/`) | not used for propagation | | | |
| non-elastic (`tables/xs/`, `xslow xsup`) | not used | | | burn-up uncertainty not included |
| any channel, TENDL default not set | | 0 | | **LOWER BOUND** |

### 4.3 Why each source is handled this way

**IAEA evaluations: use their dxs.** About 150 medical channels were evaluated by IAEA
expert groups. They selected EXFOR measurements, corrected them (updated monitor cross
sections and decay data), fitted Pade functions, and estimated an energy-dependent
uncertainty with a 4% systematic floor [Her20, Tar19]. This is the best uncertainty
available, so we use it. Three caveats:

- No covariance is published, so the energy correlation is our assumption (section 3.3).
- It is probably optimistic: correlations between experiments using similar equipment were
  neglected [Tar19], and errors shared by all data sets do not show up in their scatter
  [Cap19]. For Tc-99m the IAEA band at the 12-18 MeV peak is about 4.6%, while individual
  measurements quote 8-15% and disagree by up to a factor 2 (176-345 mb at 14-17 MeV).
- Near threshold the file sets dxs = xs (100%); we keep that.

**IAEA files beyond the evaluated range: treat as TENDL.** ISOTOPIA extends every IAEA file
to 200 MeV with TENDL scaled to the IAEA end point (manual Section 5.3), and writes dxs = 0
there. Zero would mean "exact", which is wrong; that region is model-based, so it gets the
TENDL default. In our library: Mo-100 -> Tc-99m has dxs > 0 from 6 to 50 MeV, Mo-100 -> Mo-99
from 8.4 to 50 MeV, Ga-69 -> Ge-68 from 10.5 to 80 MeV. A small interpolation detail: a grid
point counts as evaluated only if both library points around it have dxs > 0, so a point
between the last evaluated energy and the first extended one is not given a misleadingly
small interpolated value.

**TENDL: a default, because the files carry none.** TENDL is TALYS output, tuned to
measurements where they exist and default elsewhere [Kon19]. The files give no uncertainty,
and TALYS itself produces none: its companion code TASMAN does, by rerunning TALYS with
varied parameters [Kon26d], which is too expensive to do for every channel. So a default
relative uncertainty is applied (input 1), and every channel using it is named in the
warnings. Channels with sensitivity below 1e-4 at both times are summarised in one line.

**EXFOR: not propagated, but useful for choosing the default.** The raw measurements are
already folded into the IAEA evaluations and into TENDL's tuning. Using them directly would
mean evaluating each channel ourselves, choosing between data sets that disagree beyond
their error bars [Cap19]. Their best use here is checking the TENDL default (section 2).

**Non-elastic cross sections: not propagated yet.** They set target burn-up, which matters
only when R_T t_irr is not small. Their `xslow` and `xsup` columns are not documented, so
they are left for later.

### 4.4 How much of the library has its own uncertainty

`scripts/survey_xs_uncertainty.py` on the targets used for testing (p + Mo-100, p + Ga-69,
n + Lu-176): **3 of 474** residual files have a dxs column. Everything else is TENDL. For
non-medical routes the TENDL default will therefore decide most results, which is why input
1 matters so much.

---

## 5. Implementation

### 5.1 Files

| File | Status | What it does |
|---|---|---|
| `isotopia_py/xs_uncertainty.py` | new | reader, rate uncertainty, DI propagation, report |
| `isotopia_py/xs_uncertainty_defaults.json` | new | settings and the user inputs |
| `isotopia_py/route_compare.py` | new | comparison of two routes with shared-data covariance |
| `scripts/survey_xs_uncertainty.py` | new | which library files have dxs, and over which energies |
| `scripts/xs_uncertainty_example.py` | new | Tc-99m example and a two-route comparison |
| `tests/test_xs_uncertainty.py` | new | 14 tests (section 6) |

**No existing module is edited.** The grid comes from Nate's `daughters.rate_grid`, and
the solvers are called, not changed. The port still reads its two columns exactly as
before, so it still matches the Fortran. One dependency to flag to Nate: the hit-network
solver uses his private helper `daughters._residual_files` to find a hit nuclide's
library files; if he renames it, `MatrixSolver` needs the same change.

### 5.2 Data flow

```
run(lines) ------------------------------------------------> State (unchanged)
                                                                  |
di_uncertainty(st, DI, solver)                                    v
   0. daughters.rate_grid(st)      grid E_k, weights w_k, scale; checked against the port
   1. solver.channels()            channels (hit -> product) that can feed the DI
   2. rate_uncertainty(...)        per channel:
        load_channel  -> reads the file again with dxs, applies the decision table
        sandwich rule -> R, dR, source shares; self-check R == the solver's rate
   3. solver.di_atoms({c: 1+-h})   rerun with one rate scaled -> sensitivities s_c(t)
   4. combine                      (dN/N)^2 at every output time; ranking by variance
   -> DIUncertainty --> report() / compare_routes()
```

### 5.3 The pieces

**Rate grid** (`daughters.rate_grid`, Nate's). The grid energies, the weights and the
scale, so that `scale * sum(w * sigma) * 1e-27` is the port's sum. `di_uncertainty`
calls `check_rate_grid` first, which stops with an error if the weights do not reproduce
the port's rates.

**`ChannelSpec`**. One channel: the nuclide hit, the product, its library file and the
rate the solver uses. For the target the file is the one `data.crosssections` reads (or
an `xsfile` override); for a hit product it is that nuclide's own residual file.

**`read_xs_file`, `load_channel`**. Read every column and the header, interpolate xs and
dxs onto the grid with the port's rule (zero outside the file's range), and apply the
decision table point by point. Missing defaults produce a LOWER BOUND warning instead of a
crash, so a route search can still run.

**`rate_uncertainty`**. The sandwich rule (section 3.3), the share of the rate from each
data source, and a **self-check**: the rebuilt rate must equal the port's
rate the solver uses to within 1e-3, otherwise a warning says the uncertainty would be
attached to a different number. It keeps the per-point contributions a_k so that route
comparisons need no second file read.

**Solvers.** `di_uncertainty` only needs two things from a solver: the channels that can
feed the DI, and the DI atoms when some channels' rates are scaled (a Python `Protocol`,
i.e. an agreed list of methods).

- `PortSolver` uses the port's `kinetics.equations`. Candidates follow the port's own rule:
  the DI plus its beta-minus parent (Z-1) or beta-plus parent (Z+1) of the same mass, using
  `st.rtyp`. To rerun, it changes one entry of `st.reaction_rate`, runs `equations`, reads
  the DI, then restores the rates and runs `equations` once more, which puts the State back
  exactly (a test checks this).
- `MatrixSolver` uses Nate's `irradiation.irradiate`. With `solver="matrix"` only the
  target is hit; with `solver="hits"` every nuclide in the window is hit too
  (`daughters.hit_reactions`). The candidate channels are those whose product is the DI or
  one of its ancestors, found by walking the network matrix A + R (decay and reactions)
  upstream from the DI. One rate is scaled per rerun on a copy of the reactions
  dictionary; Nate's code is not edited. Burnup (loss) rates stay fixed, because the
  non-elastic cross section is not propagated. Key conversion: Nate uses (Z, A, I); the
  port uses (Z, A, -1) for a nuclide without isomers. `port_key` translates.

**`di_uncertainty`**. Central finite differences, then the combination of section 3.4 at
every output time. `np.einsum("it,ij,jt->t", u, rho, u)` computes the double sum
sum_i sum_j u_i rho_ij u_j separately for each time t in one step.

**`report`**. Amount +- % at the end of irradiation and after cooling, data-quality shares,
the ranking table (dR/R, sensitivity and variance share per channel at both times), and all
warnings. Channels with sensitivity below 1e-4 at both times are summarised in one line.

**`compare_routes`**. Section 3.5. Channels are "the same" when both routes read the same
library file. It returns the ratio, ln(ratio) +- sd, the sd under each correlation model,
the error correlation, the shared channels and the verdict.

### 5.4 Python notes for readers new to numpy

- `a @ rho @ a`: `@` is matrix multiplication; with a vector a and a matrix rho this is the
  double sum sum_j sum_k a_j rho_jk a_k in one expression.
- `np.where(condition, x, y)`: element by element, x where the condition holds, else y.
- `E1[:, None] - E2[None, :]`: "broadcasting" makes the full table of differences
  E1[j] - E2[k] without a loop.
- `@dataclass`: a class that is mainly a named bundle of fields, like a struct.

---

## 6. Verification and results

### 6.1 Tests (`pytest tests/test_xs_uncertainty.py`, all passing)

| Test | Expected | Result |
|---|---|---|
| full correlation, constant dsigma/sigma = 5%, grid of 10 and 100 points | dR/R = 5% exactly | pass (1e-12) |
| no correlation, 25 points | 5% / sqrt(25) | pass |
| Monte Carlo: 20 000 samples of sigma from its covariance (exponential, L = 3 MeV) | spread of R = sandwich result | pass (within 3%) |
| self-check | rebuilt R = port's R, no warning | pass |
| IAEA file with dxs = 0 above 20 MeV | point at 25 MeV tagged IAEA-extended, given the default | pass |
| TENDL file, default not set | dsigma = 0 and LOWER BOUND warning | pass |
| defaults file | loads, has the three projectile families | pass |
| two channels, N = 3 R_1 + R_2 | s = 0.75 and 0.25; dN/N = hypot(0.75 x 5%, 0.25 x 20%); ranking order | pass |
| same curve, two windows, full correlation, constant dsigma/sigma | sd of ln(ratio) = 0 | pass |
| 2% difference, exponential model | larger sd kept; verdict tie | pass |
| p-Mo100-Tc099m sample, port (needs the library) | R = port R; 100% IAEA; 4% < dN/N < 5%; State restored | pass |
| same sample, matrix solver | same dN/N at EOI as the port (within 2%) | pass |
| same sample, double hitting (window 2/1/1/1) | second-hit channels found; no self-check warning | pass |

Regression: with the team's reference outputs (`MIDAS-22-033/isotopia_midas`), the
samples p-Mo100-Tc099m, p-Ga069-Ge068 and n-Lu176-Lu177 pass on this branch exactly as on
`main`, as they must, since no port module is edited. (Only samples whose library data I had
were run; please run the full `pytest` with the complete library before merging.)

### 6.2 Results on the sample routes

TENDL default set to 30% (charged) and 25% (neutron) **for illustration only**; input 1 is
still open.

| Route | DI | dN/N at EOI | after cooling | Data source | Notes |
|---|---|---|---|---|---|
| p + Mo-100, 16 -> 12 MeV, port | Tc-99m | 4.55% | 4.55% | IAEA 100% | Mo-99 feeds 1.7%; the port has no Mo-99 ingrowth in cooling (defect 4) |
| same, Nate's solver | Tc-99m | 4.56% | 3.89% | IAEA 100% | with ingrowth, Mo-99 makes 23% of the Tc-99m after cooling |
| same, with double hitting | Tc-99m | 4.56% | 3.89% | IAEA 100% | 15 second-hit channels found, each with sensitivity below 1e-4 for a 1-day irradiation |
| p + Mo-100, 30 -> 20 MeV, port | Tc-99m | 4.92% | 4.92% | IAEA 100% | Mo-99 now 27% of production |
| p + Ga-69, 25 -> 15 MeV, port | Ge-68 | 11.34% | 11.34% | IAEA 100% | |
| n + Lu-176, port | Lu-177 | 25.0% | 25.0% | TENDL 100% | entirely the default; LOWER BOUND (0%) if the default is not set |

Rate self-check on all channels above: agreement better than 1e-6.

Route comparison, Tc-99m from p + Mo-100 (port):

| A vs B | ratio A/B | sd, full | sd, exponential | verdict |
|---|---|---|---|---|
| 16 -> 12 vs 20 -> 16 MeV | 0.859 | 0.004 | 0.041 | B better |
| 16 -> 12 vs 30 -> 20 MeV | 0.694 | 0.015 | 0.044 | B better |

How the comparison depends on the correlation model (16 -> 12 vs 20 -> 16 MeV):

| Model | dN/N route A | sd of ln(ratio) | error correlation |
|---|---|---|---|
| full | 4.55% | 0.004 | 1.00 |
| exponential, L = 10 MeV | 4.27% | 0.030 | 0.78 |
| exponential, L = 5 MeV | 4.02% | 0.036 | 0.62 |
| exponential, L = 2 MeV | 3.44% | 0.041 | 0.34 |
| none | 0.46% | 0.007 | 0.00 |

The verdict is the same in every case here, because the routes differ by 16% or more. For
routes closer than a few percent, the choice of L (input 4) decides between "better" and
"tie".

---

## 7. How it fits the other extensions

**Nate, matrix solver and double hitting (in `main`).** Both work through `MatrixSolver`:
`solver="matrix"` for target reactions only, `solver="hits"` with his `Window` and
`hit_reactions`. The rate grid is his `daughters.rate_grid`, so there is one grid
implementation in the code, not two. Tested on `main` at `683df9e`: same Tc-99m result
at EOI as the port, Mo-99 ingrowth after cooling, and second-hit channels found. Some of
his own tests fail in my environment; they fail the same way on `main` alone, because my
copy of the library is partial.

**Aaron, layered targets (in `main`).** Nothing of his is edited. The rate grid comes from
the same `_charged_particle_spectrum` his stopping-power work feeds (SRIM or Bethe-Bloch),
so the weights automatically follow his stopping model. Layers in front
of the target only change the energy entering it, which the grid already reflects. Isotope
production in the layers is a separate run per layer, so each layer gets its own
uncertainty the same way. Stopping-power uncertainty (SRIM versus Bethe-Bloch, which he
measured at 2.7% on Tc-99m) is not included yet.

**Rome, route search (in `main`, `scripts/`).** His `run_one` runs the port and
reads `st.Niso` for the DI. The natural place to add uncertainty is right there: call
`xs_uncertainty.di_uncertainty(st, di)`, write `rel` at EOI and at end of cooling into
`scan.csv`, and run `compare_routes` on the best few routes. Our relative uncertainty
applies unchanged to his mass metric.

**Brett, economics and hazard (in `main`).** No dependency. The "tie" verdicts
are where rarity and hazard should decide.

---

## 8. Limitations and next steps

- **TENDL default not chosen yet** (input 1). Until it is, TENDL-dominated results are
  lower bounds.
- Natural targets are not supported (input 6).
- Not propagated: decay data, stopping power, the non-elastic (burnup) cross sections
  (including the burnup of the DI itself when it is hit), impurities.
- The energy correlation inside IAEA files is assumed, not known.
- Possible next steps: compare TENDL with EXFOR to justify the default; run TASMAN for the
  TENDL channels of the top routes; add the stopping-power term (Aaron's SRIM versus
  Bethe-Bloch comparison is a starting point); propagate burnup cross sections for long
  irradiations where they matter.

---

## 9. References

Project: ISOTOPIA manual 2.23 (Section 4.2.3 Eq. 4.11 and 4.20-4.28; Section 5.2-5.3;
Section 7.1); MIDAS Status Update (slides 24, 33, 54-55); MIDAS Project Context (item 6);
`equations_f90_findings.md` (defects 1-6); implementation plan "MIDAS Cross-Section
Uncertainty Propagation: Implementation Plan".

Literature:

- [Alh24] E. Alhassan et al., Bayesian model averaging for nuclear data evaluation (2024). https://doi.org/10.1007/s41365-024-01543-w
- [Cap19] R. Capote et al., Unrecognized sources of uncertainties (USU) in experimental nuclear data. https://doi.org/10.1016/j.nds.2019.12.004
- [Car20] M. P. Carante et al., A statistical description of nuclear reaction models for medical radionuclides: Sc-47 (2020). https://arxiv.org/abs/2006.07146
- [Fox20] M. B. Fox et al., Investigating high-energy proton-induced reactions on spherical nuclei, Phys. Rev. C 103, 034601 (2021). https://doi.org/10.1103/PHYSREVC.103.034601
- [Her08] N. Herranz et al., Impact of different correlation structures in cross-section covariance matrices on the inventory (2008).
- [Her20] Upgrade of IAEA recommended data for production of PET and SPECT isotopes, Nucl. Data Sheets (2021). https://doi.org/10.1016/J.NDS.2021.04.008
- [Kon08] A. Konobeyev et al., Uncertainty in activation cross-section calculations at intermediate proton energies, Kerntechnik (2008). https://doi.org/10.3139/124.100534
- [Kon19] A. J. Koning et al., TENDL: complete nuclear data library, Nucl. Data Sheets 155 (2019). https://doi.org/10.1016/J.NDS.2019.01.002
- [Kon26d] A. J. Koning, TASMAN-2.2: statistical software for TALYS, IAEA (2026). https://doi.org/10.61092/iaea.wk9h-13p7
- [Tar19] Recommended nuclear data for medical radioisotope production: diagnostic positron emitters, J. Radioanal. Nucl. Chem. (2019). https://doi.org/10.1007/s10967-018-6380-5
