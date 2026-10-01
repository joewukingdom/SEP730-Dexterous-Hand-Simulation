"""
ORCA hand interactive control with pose recording and replay.

Two windows open:
  * MuJoCo viewer   — right panel → Ctrl tab: drag sliders to move joints
  * Control panel   — pick a scene (e.g. v2 right hand on a Franka Panda) and
                      record / replay poses with buttons

Keyboard shortcuts (click the viewer window first):
  R  — record current joint positions as a keyframe
  P  — replay all recorded keyframes in order (interpolated)
  C  — clear all recorded keyframes
  S  — save keyframes to poses.json
  L  — load keyframes from poses.json

Usage:
    python slider_control.py
"""

import json
import queue
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    from orca_sim.versions import SCENES_ROOT
except ModuleNotFoundError as exc:
    if exc.name in {"mujoco", "numpy"}:
        raise SystemExit(
            f"Missing runtime dependency '{exc.name}'. "
            "Activate your environment and run `uv pip install -e .`."
        ) from exc
    raise

POSES_FILE = REPO_ROOT / "poses.json"
INTERP_STEPS = 60   # frames between each keyframe during replay
FPS = 30
MAX_LAG = 0.1       # s of sim time to catch up per frame before resyncing the clock
DEFAULT_SCENE = "v2/scene_right.xml"


def list_scenes() -> list[str]:
    """All scene files as 'version/file.xml', newest version first."""
    scenes = [
        f"{p.parent.name}/{p.name}"
        for p in SCENES_ROOT.glob("*/scene*.xml")
    ]
    return sorted(scenes, key=lambda s: (-int(s.split("/")[0].lstrip("v")), s))


class SliderControl:
    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer = None
        self.scene_name = ""

        self.keyframes: list[np.ndarray] = []
        # Replay state
        self.replay_queue: list[np.ndarray] = []  # full sequence including start pose
        self.replay_index = 0                     # current segment index
        self.replay_step = 0                      # interpolation step within segment
        self.is_replaying = False

        # The viewer's key callback runs on the viewer thread; hand keys to the
        # Tk thread through this queue so all state changes happen in one place.
        self.key_queue: queue.Queue[str] = queue.Queue()

        self._build_panel()

    # ------------------------------------------------------------------ UI
    def _build_panel(self) -> None:
        self.root = tk.Tk()
        self.root.title("ORCA Control Panel")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Scene").grid(row=0, column=0, sticky="w")
        scenes = list_scenes()
        self.scene_var = tk.StringVar(
            value=DEFAULT_SCENE if DEFAULT_SCENE in scenes else scenes[0]
        )
        combo = ttk.Combobox(frame, textvariable=self.scene_var, values=scenes,
                             state="readonly", width=34)
        combo.grid(row=1, column=0, columnspan=5, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _e: self.load_scene(self.scene_var.get()))
        ttk.Button(frame, text="Reload", command=lambda: self.load_scene(self.scene_var.get())
                   ).grid(row=1, column=5, padx=(6, 0))

        ttk.Label(frame, text="Pose recording").grid(row=2, column=0, sticky="w", pady=(10, 0))
        for col, (key, label) in enumerate([("R", "Record R"), ("P", "Play P"), ("C", "Clear C"),
                                            ("S", "Save S"), ("L", "Load L")]):
            ttk.Button(frame, text=label, width=7,
                       command=lambda k=key: self.handle_key(k)
                       ).grid(row=3, column=col, padx=2)

        self.status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.status_var, foreground="#555"
                  ).grid(row=4, column=0, columnspan=6, sticky="w", pady=(10, 0))

    def set_status(self, msg: str) -> None:
        print(msg)
        self.status_var.set(msg)

    # --------------------------------------------------------------- scene
    def load_scene(self, scene: str) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        scene_path = SCENES_ROOT / scene
        try:
            model = mujoco.MjModel.from_xml_path(str(scene_path))
        except ValueError as exc:
            self.set_status(f"Failed to load {scene}\n{exc}")
            return
        data = mujoco.MjData(model)
        if model.nkey > 0:
            # Start from the scene's first keyframe (e.g. the Panda "home" pose)
            mujoco.mj_resetDataKeyframe(model, data, 0)

        if self.keyframes and len(self.keyframes[0]) != model.nu:
            self.keyframes.clear()
            print("Keyframes cleared: new scene has a different number of actuators.")
        self.is_replaying = False
        self.replay_queue.clear()

        self.model, self.data, self.scene_name = model, data, scene
        self._reset_clock()
        self.viewer = mujoco.viewer.launch_passive(
            model, data, key_callback=self._on_viewer_key
        )
        mujoco.mjv_defaultFreeCamera(model, self.viewer.cam)
        self.set_status(f"Loaded {scene} ({model.nu} actuators)")

    def _reset_clock(self) -> None:
        self.wall_start = time.perf_counter()
        self.sim_start = self.data.time

    # ------------------------------------------------------------ commands
    def _on_viewer_key(self, keycode: int) -> None:
        if 32 <= keycode < 127:
            self.key_queue.put(chr(keycode))

    def handle_key(self, key: str) -> None:
        if self.data is None:
            return
        data, model = self.data, self.model

        if key == "R":
            pose = data.ctrl.copy()
            self.keyframes.append(pose)
            print(f"[R] ctrl=[{', '.join(f'{v:.3f}' for v in pose)}]")
            self.set_status(f"[R] Recorded keyframe #{len(self.keyframes)}")

        elif key == "P":
            if not self.keyframes:
                self.set_status("[P] No keyframes recorded. Press R first.")
                return
            if self.is_replaying:
                self.set_status("[P] Already replaying.")
                return
            self.replay_queue.clear()
            self.replay_queue.append(data.ctrl.copy())   # start from current pose
            self.replay_queue.extend(self.keyframes)
            self.replay_index = 0
            self.replay_step = 0
            self.is_replaying = True
            self.set_status(f"[P] Replaying {len(self.keyframes)} keyframe(s)...")

        elif key == "C":
            self.keyframes.clear()
            self.is_replaying = False
            self.replay_queue.clear()
            self.set_status("[C] Cleared all keyframes.")

        elif key == "S":
            with open(POSES_FILE, "w") as f:
                json.dump([kf.tolist() for kf in self.keyframes], f, indent=2)
            self.set_status(f"[S] Saved {len(self.keyframes)} keyframe(s) → {POSES_FILE.name}")

        elif key == "L":
            if not POSES_FILE.exists():
                self.set_status(f"[L] {POSES_FILE.name} not found.")
                return
            with open(POSES_FILE) as f:
                loaded = json.load(f)
            if any(len(kf) != model.nu for kf in loaded):
                self.set_status(f"[L] {POSES_FILE.name} was recorded for a different scene "
                                f"(expected {model.nu} values per keyframe). Not loaded.")
                return
            self.keyframes.clear()
            self.keyframes.extend(np.array(kf, dtype=np.float64) for kf in loaded)
            self.set_status(f"[L] Loaded {len(self.keyframes)} keyframe(s) from {POSES_FILE.name}")

    # ----------------------------------------------------------- main loop
    def tick(self) -> None:
        while not self.key_queue.empty():
            self.handle_key(self.key_queue.get_nowait())

        if self.viewer is not None and self.viewer.is_running():
            model, data = self.model, self.data
            if self.is_replaying:
                idx = self.replay_index
                if idx < len(self.replay_queue) - 1:
                    alpha = self.replay_step / INTERP_STEPS
                    data.ctrl[:] = ((1 - alpha) * self.replay_queue[idx]
                                    + alpha * self.replay_queue[idx + 1])
                    self.replay_step += 1
                    if self.replay_step >= INTERP_STEPS:
                        self.replay_step = 0
                        self.replay_index += 1
                        print(f"  → keyframe {self.replay_index}/{len(self.keyframes)}")
                else:
                    self.is_replaying = False
                    self.set_status("[P] Replay done.")
            # Step until sim time catches up with wall-clock time (real-time).
            target = self.sim_start + (time.perf_counter() - self.wall_start)
            max_steps = int(MAX_LAG / model.opt.timestep)
            steps = 0
            while data.time < target and steps < max_steps:
                mujoco.mj_step(model, data)
                steps += 1
            if steps == max_steps:
                # Computer can't keep up: resync the clock instead of piling up lag
                self._reset_clock()
            self.viewer.sync()

        self.root.after(int(1000 / FPS), self.tick)

    def run(self) -> None:
        print("Viewer: right panel → Ctrl tab : drag sliders to move joints")
        print("Control panel: pick a scene, R/P/C/S/L buttons (or keys in the viewer)")
        self.load_scene(self.scene_var.get())
        self.tick()
        self.root.mainloop()

    def quit(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.root.destroy()


if __name__ == "__main__":
    SliderControl().run()
