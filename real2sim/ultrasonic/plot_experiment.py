"""Chart for the ultrasonic distance experiment: sensor reading vs camera reference.

    python real2sim/ultrasonic/plot_experiment.py   -> docs/figures/real2sim/ultrasonic_vs_camera.png

Numbers from 2026-11-20 (twin_check.py compare + side-camera gap per position, cube face
parallel to the sensor). Camera = horizontal gap transducer face -> cube near face measured
in the side image at the sensor's height; real = median HC-SR04 reading; sim = calibrated
sim reading (cone of rangefinders + 2.9 mm offset, 20 mm floor).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POS = ["P1", "P2", "P4", "P3", "P5"]
CAMERA = [4.6, 27.0, 42.2, 61.4, 77.1]
REAL = [20.0, 33.0, 39.0, 61.0, 80.0]
SIM = [20.0, 26.8, 42.9, 60.7, 77.3]

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1, 2 (validated, light surface)


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "docs" / "figures" / "real2sim" / "ultrasonic_vs_camera.png"
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    lim = (0, 90)
    ax.axhspan(0, 20, color="#f0efec", zorder=0)
    ax.text(86, 10, "below 20 mm:\nsensor dead zone", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.plot(lim, lim, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(50, 52.5, "reading = camera", rotation=45, ha="center", va="bottom", fontsize=8.5, color=INK2,
            rotation_mode="anchor", transform_rotates_text=True)
    # real: larger ring, sim: smaller filled square on top, so coincident points stay visible
    ax.plot(CAMERA, REAL, "o", ms=11, mfc="none", mec=BLUE, mew=2.2, zorder=3, label="real HC-SR04")
    ax.plot(CAMERA, SIM, "s", ms=6, color=ORANGE, mec=SURF, mew=1.5, zorder=4, label="sim (calibrated)")
    for x, y, p in zip(CAMERA, REAL, POS):
        ax.annotate(p, (x, min(y, s := SIM[POS.index(p)])), xytext=(0, -16), textcoords="offset points",
                    fontsize=8.5, color=INK2, ha="center", va="top")
    ax.annotate("real", (CAMERA[-1], REAL[-1]), xytext=(9, 2), textcoords="offset points", fontsize=9, color=INK,
                va="center")
    ax.annotate("sim", (CAMERA[-1], SIM[-1]), xytext=(9, -4), textcoords="offset points", fontsize=9, color=INK,
                va="center")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("camera reference: transducer face to cube face (mm)", color=INK, fontsize=10)
    ax.set_ylabel("sensor reading (mm)", color=INK, fontsize=10)
    ax.set_title("Ultrasonic sensor vs camera reference, cube face parallel", color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
