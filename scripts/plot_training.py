"""Live training curves for a PPO run (refreshes every 30 s while training).

    python scripts/plot_training.py runs/ppo_reorient_random_main
    python scripts/plot_training.py runs/ppo_reorient_random_main --once --save curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

PANELS = [
    ("rollout/ep_rew_mean", "Mean episode return (training)"),
    ("rollout/ep_len_mean", "Mean episode length (200 = never dropped)"),
    ("eval/success_rate", "Eval success rate (held red-face-up)"),
    ("rollout/success_rate", "Training success rate"),
]


def read(run: Path) -> dict[str, tuple[list[float], list[float]]]:
    series: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k, _ in PANELS}
    path = run / "progress.csv"
    if not path.exists():
        return series
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = float(row["time/total_timesteps"])
            except (KeyError, ValueError):
                continue
            for key, _ in PANELS:
                val = row.get(key, "")
                if val not in ("", None):
                    series[key][0].append(step)
                    series[key][1].append(float(val))
    return series


def draw(fig, axes, run: Path) -> None:
    series = read(run)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.clear()
        xs, ys = series[key]
        ax.plot(xs, ys, marker="." if len(xs) < 60 else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.3)
        if "success" in key:
            ax.set_ylim(-0.02, 1.02)
    latest = max((s[0][-1] for s in series.values() if s[0]), default=0)
    fig.suptitle(f"{run.name}  —  {latest:,.0f} steps", fontsize=11)
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--once", action="store_true", help="draw once instead of refreshing")
    parser.add_argument("--save", type=Path, default=None, help="save the figure to this file")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    draw(fig, axes, args.run)
    if args.save:
        fig.savefig(args.save, dpi=110)
    if args.once:
        if not args.save:
            plt.show()
        return
    while plt.fignum_exists(fig.number):
        draw(fig, axes, args.run)
        plt.pause(30)


if __name__ == "__main__":
    main()
