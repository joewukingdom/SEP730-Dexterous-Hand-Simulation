"""Sweep the sim specular threshold (per ray) against recorded datasets.

    python real2sim/ultrasonic/threshold_sweep.py run01 angle02

For every row with a cube pose, the sim cube is placed at that pose and the specular sim
reading is computed for several max_incidence_deg values; prints the sees-cube agreement
with the real sensor (dead-zone readings < 25 mm excluded) per dataset and pooled.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402

THS = [15, 20, 25, 27, 30, 35, 90]


def main() -> None:
    rig = C.load_rig(C.HERE / "rig.yaml")
    m = mujoco.MjModel.from_xml_path(str((C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")).resolve()))
    d = mujoco.MjData(m)
    j = m.joint("cube_freejoint").qposadr[0]
    cube = m.body("task_cube").id
    hand = {b for b in range(1, m.nbody) if b != cube}
    pooled = {th: [0, 0] for th in THS}
    for name in sys.argv[1:]:
        agree = {th: 0 for th in THS}
        n = 0
        for r in csv.DictReader(open(C.DATA_DIR / "recordings" / name / "dataset.csv", newline="")):
            if r["cube_found"] != "1" or (r["us_mm"] and float(r["us_mm"]) < 25):
                continue
            d.qpos[j:j + 3] = [float(r[k]) for k in ("cube_x", "cube_y", "cube_z")]
            d.qpos[j + 3:j + 7] = [float(r[k]) for k in ("cube_qw", "cube_qx", "cube_qy", "cube_qz")]
            mujoco.mj_forward(m, d)
            real = r["real_sees_cube"] == "1"
            n += 1
            for th in THS:
                s = C.ultrasonic_reading_specular(m, d, rig["ultrasonic"], th, hand)
                agree[th] += int((s is not None and s * 1000 <= 150) == real)
        print(f"{name} ({n} samples): " + "  ".join(f"{th}deg {agree[th] / n:.0%}" for th in THS))
        for th in THS:
            pooled[th][0] += agree[th]
            pooled[th][1] += n
    print("pooled: " + "  ".join(f"{th}deg {a / t:.0%}" for th, (a, t) in pooled.items()) + "   (90deg = nearest model)")


if __name__ == "__main__":
    main()
