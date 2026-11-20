"""Live preview of the rig cameras; SPACE saves one frame from every camera.

    python real2sim/cameras/capture.py --list                 # which device indices work
    python real2sim/cameras/capture.py intrinsics_top --cams top
    python real2sim/cameras/capture.py extrinsics             # all cameras, same instant

Frames go to real2sim/data/<session>/<camera>/NNN.png (full resolution, lossless).
The preview draws the detected ChArUco corners (green) and how many were found;
a frame is only worth saving when most of the board's corners are detected.
Keys: SPACE save, A toggle auto-save every 1.5 s (handy for intrinsics), Q / ESC quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import camlib as C


def list_devices(n: int = 8) -> None:
    for i in range(n):
        try:
            cap = C.open_camera(i, 1920, 1080)
        except RuntimeError:
            print(f"device {i}: -")
            continue
        ok, f = cap.read()
        print(f"device {i}: {f.shape[1]}x{f.shape[0]}" if ok else f"device {i}: opened, no frame")
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="folder name under real2sim/data/")
    ap.add_argument("--rig", default=C.HERE / "rig.yaml")
    ap.add_argument("--cams", nargs="*", help="camera names from rig.yaml (default: all)")
    ap.add_argument("--list", action="store_true", help="probe device indices and exit")
    args = ap.parse_args()
    if args.list:
        list_devices()
        return
    if not args.session:
        ap.error("session name required")

    rig = C.load_rig(args.rig)
    names = args.cams or list(rig["cameras"])
    det = C.BoardDetector(rig["board"])
    n_total = (rig["board"]["squares_x"] - 1) * (rig["board"]["squares_y"] - 1)
    caps = {}
    for n in names:
        c = rig["cameras"][n]
        caps[n] = C.open_camera(c["device"], c["width"], c["height"])
        print(f"opened {n} (device {c['device']})")
    out = C.DATA_DIR / args.session
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)
    idx = max([int(p.stem) for n in names for p in (out / n).glob("*.png")] + [-1]) + 1
    auto, last_auto = False, 0.0
    win = "capture  (SPACE save, A auto, Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        frames = {}
        for n, cap in caps.items():
            # grab all first, then retrieve, so the cameras are as close in time as possible
            cap.grab()
        for n, cap in caps.items():
            ok, f = cap.retrieve()
            if ok:
                frames[n] = f
        tiles = []
        for n in names:
            f = frames.get(n)
            if f is None:
                tiles.append(np.zeros((540, 960, 3), np.uint8))
                continue
            vis = f.copy()
            corners, ids = det.detect(f)
            for (x, y) in corners:
                cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), -1)
            col = (0, 255, 0) if len(ids) >= 0.6 * n_total else (0, 0, 255)
            cv2.putText(vis, f"{n}: {len(ids)}/{n_total} corners", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)
            tiles.append(cv2.resize(vis, (960, int(960 * f.shape[0] / f.shape[1]))))
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        view = np.hstack(tiles)
        cv2.putText(view, f"saved: {idx}{'   AUTO' if auto else ''}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if key in (ord("q"), 27):
            break
        if key in (ord("a"), ord("A")):
            auto = not auto
        if key == ord(" ") or (auto and now - last_auto > 1.5):
            last_auto = now
            if len(frames) == len(names):
                for n, f in frames.items():
                    cv2.imwrite(str(out / n / f"{idx:03d}.png"), f)
                print(f"saved {idx:03d}")
                idx += 1
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print(f"{idx} frame sets in {out}")


if __name__ == "__main__":
    main()
