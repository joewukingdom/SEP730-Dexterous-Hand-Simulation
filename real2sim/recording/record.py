"""Record the top camera, the side camera and the ultrasonic sensor together, every sample
stamped with ONE host clock (time.perf_counter, seconds since the recording started).

    python real2sim/recording/record.py run01                 # Ctrl+C (or --seconds) to stop
    python real2sim/recording/record.py run01 --seconds 60

Writes real2sim/data/recordings/<name>/:
  top/000000.jpg ...          top camera frames (JPEG q95)
  side/000000.jpg ...         side camera frames
  top_frames.csv              frame, t, file        (t = when the frame arrived on the host)
  side_frames.csv             frame, t, file
  ultrasonic.csv              sample, t, dist_mm, raw     (dist_mm empty = no echo)
  meta.json                   wall-clock start, devices, resolution, port, notes

Each stream runs in its own thread and stamps a sample the moment it arrives, so nothing
is paired here; align.py pairs them afterwards by time. Arrival time is not capture time:
a camera frame arrives one exposure + USB transfer later (tens of ms), an ultrasonic line
after the echo and the serial adapter's buffering (the FTDI latency timer is 16 ms by
default). Those delays are roughly constant per stream; align.py takes per-stream offsets.
Nothing is ever written to the serial port.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
sys.path.insert(0, str(HERE.parents[1] / "hardware" / "ultrasonic"))
import camlib as C  # noqa: E402
import read_ultrasonic as R  # noqa: E402

T0 = time.perf_counter()


def now() -> float:
    return time.perf_counter() - T0


class CameraStream(threading.Thread):
    def __init__(self, name: str, cfg: dict, out: Path, stop: threading.Event):
        super().__init__(daemon=True)
        self.name_, self.cfg, self.out, self.stop = name, cfg, out / name, stop
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows: list[tuple[int, float, str]] = []
        self.q: queue.Queue = queue.Queue(maxsize=120)
        self.dropped = 0
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.ready = threading.Event()
        self.latest = None  # last frame, for the preview window

    def _write(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            path, frame = item
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def run(self) -> None:
        self.cap = C.open_camera(self.cfg["device"], self.cfg["width"], self.cfg["height"])
        self.writer.start()
        for _ in range(10):  # let exposure settle before recording
            ok, frame = self.cap.read()
            if ok:
                self.latest = frame
        self.ready.set()
        i = 0
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            t = now()
            if not ok:
                continue
            self.latest = frame
            fname = f"{i:06d}.jpg"
            try:
                self.q.put_nowait((self.out / fname, frame))
            except queue.Full:
                self.dropped += 1
                continue
            self.rows.append((i, t, f"{self.name_}/{fname}"))
            i += 1
        self.cap.release()
        self.q.put(None)
        self.writer.join()


class UltrasonicStream(threading.Thread):
    """Opened only after the cameras are streaming. Lines that queued up before (the FTDI
    driver keeps them even after reset_input_buffer) would all arrive at once and get a
    wrong arrival time, so the backlog is discarded: lines are dropped until one arrives
    after a quiet gap of >= 60 ms (the firmware sends one line every ~100 ms)."""

    def __init__(self, port: str, stop: threading.Event):
        super().__init__(daemon=True)
        self.port, self.stop = port, stop
        self.rows: list[tuple[int, float, str, str]] = []
        self.discarded = 0
        self.ready = threading.Event()

    def _drain(self, ser) -> None:
        ser.reset_input_buffer()
        last = now()
        while not self.stop.is_set():
            raw = ser.readline()
            t = now()
            if raw and t - last >= 0.060 and self.discarded > 0:
                return  # first fresh line: the caller re-reads from the next one
            if raw:
                self.discarded += 1
            last = t

    def run(self) -> None:
        import serial
        ser = serial.Serial(self.port, 115200, timeout=0.2)
        self.ser = ser
        self._drain(ser)
        self.ready.set()
        i = 0
        while not self.stop.is_set():
            raw = self.ser.readline()
            t = now()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            ok, v = R.parse(line)
            if not ok:
                continue
            self.rows.append((i, t, "" if v is None else f"{v:.1f}", line))
            i += 1
        self.ser.close()


WIN = "record"


def _font(size: int):
    from PIL import ImageFont
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


FONT_BIG, FONT_SMALL = _font(64), _font(30)


def _text(img, lines: list[tuple[str, int, tuple[int, int, int]]], x: int = 30, y: int = 30):
    """Draw (Chinese) text lines with PIL; lines = [(text, size_flag 0/1, BGR colour)]."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, big, col in lines:
        font = FONT_BIG if big else FONT_SMALL
        rgb = (col[2], col[1], col[0])
        if font is None:
            d.text((x, y), text, fill=rgb)
            y += 30
            continue
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            d.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        d.text((x, y), text, font=font, fill=rgb)
        y += (80 if big else 42)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _show(cams, lines, border=None) -> bool:
    """Preview of both cameras with status text; returns False if Q was pressed."""
    tiles = []
    for c in cams:
        f = c.latest
        tiles.append(np.full((360, 640, 3), 60, np.uint8) if f is None else cv2.resize(f, (640, 360)))
    view = np.hstack(tiles)
    view = np.vstack([view, np.full((180, view.shape[1], 3), 30, np.uint8)])
    if border is not None:
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), border, 14)
    view = _text(view, lines, 30, 375)
    cv2.imshow(WIN, view)
    return (cv2.waitKey(30) & 0xFF) not in (ord("q"), 27)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="recording name (folder under real2sim/data/recordings/)")
    ap.add_argument("--seconds", type=float, help="stop after this long (default: until Ctrl+C)")
    ap.add_argument("--port", help="ultrasonic serial port (default: auto-detect by listening)")
    ap.add_argument("--note", default="", help="free text stored in meta.json")
    args = ap.parse_args()

    rig = C.load_rig(C.HERE / "rig.yaml")
    out = C.DATA_DIR / "recordings" / args.name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists; pick another name")
    out.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    port = args.port or R.pick_port(115200)
    print(f"ultrasonic on {port}; opening cameras (this can take a minute) ...", flush=True)
    cams = [CameraStream(n, rig["cameras"][n], out, stop) for n in ("top", "side")]
    us = UltrasonicStream(port, stop)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 540)
    for c in cams:
        c.start()
    t_open = time.time()
    while not all(c.ready.is_set() for c in cams):
        _show(cams, [("相机启动中，请稍候 …", 1, (200, 200, 200)),
                     (f"已等待 {time.time() - t_open:4.0f} 秒（通常需要约 1 分钟），先不要动方块", 0, (200, 200, 200))])
    us.start()
    while not us.ready.is_set():
        _show(cams, [("超声波准备中 …", 1, (200, 200, 200))])
    t_cd = time.time()
    while time.time() - t_cd < 3.0:
        _show(cams, [(f"准备开始：{3 - int(time.time() - t_cd)}", 1, (0, 200, 255)),
                     ("倒计时结束后开始动作", 0, (200, 200, 200))])
    started = datetime.now(timezone.utc).isoformat()
    t_start = now()
    print("RECORDING - move the cube now. Q in the window or Ctrl+C to stop.", flush=True)
    try:
        while args.seconds is None or now() - t_start < args.seconds:
            el = now() - t_start
            last = us.rows[-1][2] if us.rows else "-"
            left = "" if args.seconds is None else f"，还剩 {args.seconds - el:4.1f} 秒"
            if not _show(cams, [(f"● 正在录制  {el:5.1f} 秒", 1, (60, 60, 255)),
                                (f"现在可以动方块{left}    超声波 {last or '无回波'} mm    按 Q 结束", 0,
                                 (255, 255, 255))], border=(40, 40, 230)):
                break
    except KeyboardInterrupt:
        pass
    stop.set()
    t_stop = now()
    for _ in range(3):
        _show(cams, [("录制结束，正在保存 …", 1, (120, 220, 120))])
    for th in (*cams, us):
        th.join(timeout=10)
    cv2.destroyAllWindows()
    print()

    # keep only what arrived between the end of the countdown and the stop (frames before
    # it are still on disk but not listed)
    for c in cams:
        c.rows = [r for r in c.rows if t_start <= r[1] <= t_stop]
    us.rows = [r for r in us.rows if t_start <= r[1] <= t_stop]
    for c in cams:
        with open(out / f"{c.name_}_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t", "file"])
            w.writerows((i, f"{t:.6f}", p) for i, t, p in c.rows)
    with open(out / "ultrasonic.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "t", "dist_mm", "raw"])
        w.writerows((i, f"{t:.6f}", d, raw) for i, t, d, raw in us.rows)
    meta = {"started_utc": started, "clock": "time.perf_counter, s since process start; t = arrival on host",
            "recording_start_t": t_start, "stopped_t": t_stop, "port": port,
            "cameras": {c.name_: {**rig["cameras"][c.name_], "frames": len(c.rows), "dropped": c.dropped}
                        for c in cams},
            "ultrasonic_samples": len(us.rows), "ultrasonic_backlog_discarded": us.discarded, "note": args.note}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}: top {len(cams[0].rows)} frames, side {len(cams[1].rows)} frames, "
          f"ultrasonic {len(us.rows)} samples (dropped frames: top {cams[0].dropped}, side {cams[1].dropped})")


if __name__ == "__main__":
    main()
