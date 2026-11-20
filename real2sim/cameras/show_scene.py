"""Open the MuJoCo viewer on the real2sim scene: hand + cube + calibration board.

    python real2sim/cameras/show_scene.py            # every board placement from rig.yaml
    python real2sim/cameras/show_scene.py --calibrated

Before calibration: use it to see where `board_in_world` puts the board relative to the
sim hand, so you can lay the real board the same way relative to the real hand.
After calibration (--calibrated): loads the generated <scene>_real2sim.xml; switch to
the real cameras in the viewer's "Rendering > Camera" menu (or press [ / ]).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

import camlib as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default=C.HERE / "rig.yaml")
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()
    rig = C.load_rig(args.rig)
    if args.calibrated:
        path = C.SCENES_DIR / rig["scene"].replace(".xml", "_real2sim.xml")
        m = mujoco.MjModel.from_xml_path(str(path))
    else:
        tex = "_preview_board.png"
        C.write_board_texture(rig["board"], tex)
        poses = {"default": rig["board_in_world"]}
        poses.update({n: c["board_in_world"] for n, c in rig["cameras"].items() if c.get("board_in_world")})
        Ts = [C.T_world_boardcv(rig["board"], p) for p in poses.values()]
        for label, p in poses.items():
            print(f"board for {label}: {p}")
        path = C.SCENES_DIR / "_preview_real2sim.xml"
        path.write_text(C.build_scene_xml(rig["scene"], rig["board"], Ts[0], [], texture_file=tex,
                                          extra_boards=Ts[1:]))
        try:
            m = mujoco.MjModel.from_xml_path(str(path))
        finally:
            path.unlink(missing_ok=True)
            (C.SCENES_DIR / tex).unlink(missing_ok=True)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.geomgroup[3] = 1
        v.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        v.cam.lookat[:] = np.array([0.1, -0.08, 0.1])
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, -60, -50
        while v.is_running():
            v.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
