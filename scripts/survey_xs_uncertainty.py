"""Survey isotopia.libs: which residual-production files carry their own uncertainty?

For every residual file (<projectile>/<target>/<lib>/tables/residual/*) this prints one
CSV row: projectile, target, product, the header's 'library:' line, whether the file has a
dxs column, and the energy range [MeV] where dxs > 0 (the IAEA evaluated range). Files
without dxs are model-only (TENDL) and get the TENDL default in xs_uncertainty.py.

Usage:
    python scripts/survey_xs_uncertainty.py [--projectile p] [--target Mo100] > survey.csv

The library path comes from ISOTOPIA_LIBS (or isotopia.libs next to the repository).
Reading all ~800 000 files takes a while; use --projectile/--target to limit it.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Run from the repository root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isotopia_py.xs_uncertainty import read_xs_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--projectile", default="*", help="n, g, p, d, t, h or a (default all)")
    parser.add_argument("--target", default="*", help="e.g. Mo100 (default all)")
    parser.add_argument("--libs", default=os.environ.get("ISOTOPIA_LIBS", "isotopia.libs"))
    args = parser.parse_args()

    out = csv.writer(sys.stdout)
    out.writerow(["projectile", "target", "product", "library", "has_dxs",
                  "dxs_Emin_MeV", "dxs_Emax_MeV", "file"])
    pattern = f"{args.projectile}/{args.target}/*/tables/residual/*"
    for path in sorted(Path(args.libs).glob(pattern)):
        header, data = read_xs_file(str(path))
        has_dxs = data.shape[1] >= 3
        emin = emax = ""
        if has_dxs and (data[:, 2] > 0).any():
            E = data[data[:, 2] > 0, 0]
            emin, emax = f"{E.min():g}", f"{E.max():g}"
        product = path.name.split("-rp")[-1].split(".")[0]          # e.g. 043099m
        projectile, target = path.parts[-6], path.parts[-5]
        out.writerow([projectile, target, product, header.get("library", ""),
                      has_dxs, emin, emax, str(path)])


if __name__ == "__main__":
    main()
