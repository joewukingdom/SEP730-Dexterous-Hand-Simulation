"""How the real HC-SR04 echo depends on the angle between the beam and the cube face.

    python real2sim/ultrasonic/incidence_analysis.py angle01 [run01 ...]

Reads real2sim/data/recordings/<name>/dataset.csv (process.py). Prints, per 5-degree bin of
face_incidence_deg, how often the real sensor saw the cube (reading <= 150 mm), the
agreement of a threshold rule "echo if incidence <= th" for each th, and writes
docs/figures/real2sim/incidence_<names>.png.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402

BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    names = sys.argv[1:]
    rows, dead = [], 0
    for n in names:
        for r in csv.DictReader(open(C.DATA_DIR / "recordings" / n / "dataset.csv", newline="")):
            if r["cube_found"] != "1":
                continue
            # dead zone: below ~2.5 cm the sensor reports a constant 17-20 mm whatever the angle
            if r["us_mm"] and float(r["us_mm"]) < 25:
                dead += 1
                continue
            rows.append(r)
    print(f"dropped {dead} samples in the sensor's dead zone (reading < 25 mm)")
    if not rows:
        raise SystemExit("nothing left outside the dead zone")
    inc = np.array([float(r["face_incidence_deg"]) for r in rows])
    sees = np.array([r["real_sees_cube"] == "1" for r in rows])
    dist = np.array([float(r["us_mm"]) if r["us_mm"] else np.nan for r in rows])
    print(f"{len(rows)} samples with the cube found ({', '.join(names)}); incidence {inc.min():.1f}..{inc.max():.1f} deg")
    edges = np.arange(0, np.ceil(inc.max() / 5) * 5 + 5, 5)
    centres, frac, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (inc >= lo) & (inc < hi)
        if sel.sum():
            centres.append((lo + hi) / 2)
            frac.append(sees[sel].mean())
            counts.append(int(sel.sum()))
            print(f"  {lo:4.0f}-{hi:<3.0f} deg: n={sel.sum():4d}  real sees cube {sees[sel].mean():4.0%}")
    ths = np.arange(2, 46, 1.0)
    agr = np.array([np.mean((inc <= th) == sees) for th in ths])
    best = ths[int(np.argmax(agr))]
    good = ths[agr >= agr.max() - 0.02]
    print(f"threshold rule 'echo if incidence <= th': best th = {best:.0f} deg (agreement {agr.max():.0%}); "
          f"within 2 points of best: {good.min():.0f}..{good.max():.0f} deg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    a1.bar(centres, np.array(frac) * 100, width=4.2, color=BLUE)
    for c, f_, n in zip(centres, frac, counts):
        a1.text(c, f_ * 100 + 2, f"n={n}", ha="center", fontsize=7, color=INK2)
    a1.set_xlabel("beam vs facing cube face (deg)", color=INK, fontsize=9)
    a1.set_ylabel("real sensor sees the cube (%)", color=INK, fontsize=9)
    a1.set_ylim(0, 110)
    a1.set_title("Echo from the cube vs incidence angle", loc="left", color=INK, fontsize=10)
    a2.plot(ths, agr * 100, color=ORANGE, lw=2)
    a2.axvline(best, color=INK2, lw=0.8, ls=":")
    a2.text(best + 0.8, agr.min() * 100 + 2, f"best {best:.0f} deg", fontsize=8, color=INK2)
    a2.set_xlabel("threshold th (deg)", color=INK, fontsize=9)
    a2.set_ylabel("rule 'echo if angle <= th' agrees (%)", color=INK, fontsize=9)
    a2.set_title("Choosing the sim threshold", loc="left", color=INK, fontsize=10)
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    out = C.REPO_ROOT / "docs" / "figures" / "real2sim" / f"incidence_{'_'.join(names)}.png"
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)
    seen = sees & ~np.isnan(dist)
    if seen.any():
        print(f"when the real sensor sees the cube: reading {np.nanmedian(dist[seen]):.0f} mm median")


if __name__ == "__main__":
    main()
