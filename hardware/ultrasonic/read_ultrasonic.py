"""Read the HC-SR04 / ESP32 sensor and report whether a cube is in front of it.

    python hardware/ultrasonic/read_ultrasonic.py            # auto-pick the serial port
    python hardware/ultrasonic/read_ultrasonic.py COM5
    python hardware/ultrasonic/read_ultrasonic.py COM5 --raw # just print every reading

Start it with NOTHING in front of the sensor: the first --calib-s seconds measure
the background distance (table edge, wall, or "no echo"). Afterwards a reading
that is clearly shorter than the background, and stable for a few readings, is
reported as "cube at <d> mm".

Firmware: hardware/ultrasonic/hcsr04_esp32/hcsr04_esp32.ino (prints "dist_mm,<v>").
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque

import serial
from serial.tools import list_ports


def pick_port() -> str:
    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit("No serial port found -- is the ESP32 plugged in (and its USB driver installed)?")
    # Only auto-pick the usual ESP32 USB-serial chips (Silicon Labs CP210x, WCH CH34x,
    # Espressif native USB). Never fall back to e.g. the FTDI servo bus adapter.
    esp_vids = {0x10C4, 0x1A86, 0x303A}
    for p in ports:
        if p.vid in esp_vids:
            return p.device
    names = ", ".join(f"{p.device} ({p.description})" for p in ports)
    raise SystemExit(f"No ESP32-like serial port found (have: {names}). "
                     "Pass the port explicitly if you are sure, e.g. COM5.")


def readings(ser: serial.Serial):
    """Yield distances in mm (None = no echo) from the firmware's 'dist_mm,<v>' lines."""
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("dist_mm,"):
            continue
        try:
            v = float(line.split(",", 1)[1])
        except ValueError:
            continue
        yield None if v < 0 else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="serial port, e.g. COM5 (default: auto)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--raw", action="store_true", help="print every reading, no detection")
    ap.add_argument("--calib-s", type=float, default=3.0, help="background calibration time")
    ap.add_argument("--margin-mm", type=float, default=30.0,
                    help="how much closer than the background counts as an object")
    ap.add_argument("--max-mm", type=float, default=400.0,
                    help="ignore objects farther than this (a 36 mm cube gets hard to see beyond ~30-40 cm)")
    ap.add_argument("--stable", type=int, default=3, help="consecutive readings needed to report a cube")
    args = ap.parse_args()

    port = args.port or pick_port()
    print(f"opening {port} @ {args.baud}")
    with serial.Serial(port, args.baud, timeout=1.0) as ser:
        time.sleep(1.5)  # opening the port resets most ESP32 boards
        ser.reset_input_buffer()
        src = readings(ser)

        if args.raw:
            for d in src:
                print("no echo" if d is None else f"{d:7.1f} mm", flush=True)
            return

        print(f"calibrating background for {args.calib_s:.0f} s -- keep the area in front of the sensor empty ...")
        bg_vals, n_none = [], 0
        t0 = time.time()
        for d in src:
            if d is None:
                n_none += 1
            else:
                bg_vals.append(d)
            if time.time() - t0 > args.calib_s:
                break
        if not bg_vals and n_none == 0:
            raise SystemExit("No data from the sensor -- check the firmware is flashed and the baud rate.")
        background = statistics.median(bg_vals) if len(bg_vals) > n_none else None
        print("background:", "no echo (open space)" if background is None else f"{background:.0f} mm")
        limit = args.max_mm if background is None else min(args.max_mm, background - args.margin_mm)
        print(f"an object closer than {limit:.0f} mm will be reported. Ctrl+C to stop.\n")

        recent: deque[float | None] = deque(maxlen=args.stable)
        state = None
        for d in src:
            recent.append(d if (d is not None and d < limit) else None)
            if len(recent) == recent.maxlen and all(v is not None for v in recent):
                dist = statistics.median(recent)  # type: ignore[arg-type]
                new_state = "cube"
                msg = f"cube at {dist:6.1f} mm"
            elif all(v is None for v in recent):
                new_state, msg = "empty", "nothing in front"
            else:
                continue  # in between: keep the last state, avoids flicker
            if new_state != state or new_state == "cube":
                print(msg, flush=True)
            state = new_state


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
