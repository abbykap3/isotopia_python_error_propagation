"""Example: cross-section uncertainty of Tc-99m from p + Mo-100, and a comparison of two
energy windows.

    python scripts/xs_uncertainty_example.py [--solver port|matrix|hits] [--tendl 0.3]

--solver matrix uses Nate's matrix solver; --solver hits also hits every product inside a
window of 2 mass numbers below and 1 above, 1 charge below and 1 above the target.
--tendl sets the TENDL default for this run only (otherwise the value in
isotopia_py/xs_uncertainty_defaults.json is used).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isotopia_py import xs_uncertainty as xu  # noqa: E402
from isotopia_py.daughters import Window  # noqa: E402
from isotopia_py.route_compare import compare_routes  # noqa: E402
from isotopia_py.run import run  # noqa: E402


def route(ebeam: float, eback: float, solver: str, settings: xu.Settings) -> xu.DIUncertainty:
    """Run the port for p + Mo100 at one energy window and return the Tc-99m uncertainty."""
    lines = ["projectile p", "element Mo", "mass 100",
             f"Ebeam {ebeam}", f"Eback {eback}", "Ibeam 0.15"]
    with tempfile.TemporaryDirectory() as outdir, open(Path(outdir) / "isotopia.out", "w") as out:
        st = run(lines, out=out, outdir=outdir)
        window = Window(A_below=2, A_above=1, Z_below=1, Z_above=1) if solver == "hits" else None
        return xu.di_uncertainty(st, (43, 99, 1), solver, settings, window=window)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="port", choices=["port", "matrix", "hits"])
    parser.add_argument("--tendl", type=float, help="TENDL default, e.g. 0.3 for 30%%")
    args = parser.parse_args()

    settings = xu.Settings.load()
    if args.tendl is not None:
        settings.tendl_relative = dict.fromkeys(settings.tendl_relative, args.tendl)

    a = route(16.0, 12.0, args.solver, settings)
    b = route(20.0, 16.0, args.solver, settings)
    print(xu.report(a))
    print(xu.report(b))
    print("Route A (16 -> 12 MeV) vs route B (20 -> 16 MeV):")
    print(" ", compare_routes(a, b))


if __name__ == "__main__":
    main()
