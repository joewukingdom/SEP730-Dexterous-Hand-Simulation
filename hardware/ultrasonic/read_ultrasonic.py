"""Read the HC-SR04 / ESP32 sensor and report whether a cube is in front of it.

    python hardware/ultrasonic/read_ultrasonic.py            # auto-pick the serial port
    python hardware/ultrasonic/read_ultrasonic.py COM5
    python hardware/ultrasonic/read_ultrasonic.py COM5 --raw # just print every reading

Start it with NOTHING in front of the sensor: the first --calib-s seconds measure
the background distance (table edge, wall, or "no echo"). Afterwards a reading
that is clearly shorter than the background, and stable for a few readings, is
reported as "cube at <d> mm".

Firmware (either works):
  ORCA Ultra Sonic/src/main.cpp        (ESP32-S3, prints "Distance: 12.3 cm" / "Out of range / no echo")
  hardware/ultrasonic/hcsr04_esp32/    (prints "dist_mm,<v>")

On Joe's bench the sensor is on COM4 (FTDI) and the hand's servo adapter is on COM3. The port
is found by LISTENING for distance lines; nothing is ever written to any port.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque

import serial
from serial.tools import list_ports


def parse(line: str) -> tuple[bool, float | None]:
    """(is_reading, distance_mm or None for no echo)."""
    line = line.strip()
    if line.startswith("dist_mm,"):
        try:
            v = float(line.split(",", 1)[1])
        except ValueError:
            return False, None
        return True, (None if v < 0 else v)
    if line.startswith("Distance:"):
        try:
            return True, float(line.split(":", 1)[1].split()[0]) * 10.0
        except (ValueError, IndexError):
            return False, None
    if line.lower().startswith("out of range"):
        return True, None
    return False, None


def pick_port(baud: int, listen_s: float = 2.5) -> str:
    """The first port that prints distance lines (read-only probe)."""
    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit("No serial port found -- is the ESP32 plugged in (and its USB driver installed)?")
    for p in ports:
        try:
            with serial.Serial(p.device, baud, timeout=0.3) as ser:
                t0 = time.time()
                while time.time() - t0 < listen_s:
                    if parse(ser.readline().decode(errors="ignore"))[0]:
                        return p.device
        except serial.SerialException:
            continue  # busy (e.g. a serial monitor) or not openable
    names = ", ".join(f"{p.device} ({p.description})" for p in ports)
    raise SystemExit(f"No port printing distance lines (have: {names}). Is a serial monitor still open? "
                     "Or pass the port explicitly, e.g. COM4.")


def readings(ser: serial.Serial):
    """Yield distances in mm (None = no echo)."""
    while True:
        ok, v = parse(ser.readline().decode(errors="ignore"))
        if ok:
            yield v


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

    port = args.port or pick_port(args.baud)
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
