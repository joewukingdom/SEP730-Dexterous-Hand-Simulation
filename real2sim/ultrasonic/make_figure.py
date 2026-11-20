"""Figure for the ultrasonic distance experiment (docs/figures/real2sim/ultrasonic_positions.jpg).

For each position: side-camera crop with the transducer face and the cube's near face
marked, and the three numbers (camera, real sensor, sim). Needs real2sim/data/us_check/.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

import twin_check as W

C = W.C
# (session, cube near-face pixel in the side image, sim reading mm or None)
POSITIONS = [("pos1", 1322, 20.0), ("pos2", 1245, 26.8), ("pos4", 1193, 42.9), ("pos3", 1128, 60.7),
             ("pos6", 1153, 55.9), ("pos5", 1075, 77.3)]  # sim = calibrated reading; pos6 = held-out
TRANSDUCER_PX = (1338, 515)


def gap_mm(cams, edge_px: float) -> float:
    def at(uv):
        o, d = W.ray(cams, "side", uv)
        return o + (0.0 - o[1]) / d[1] * d
    return float((at((edge_px, TRANSDUCER_PX[1]))[0] - at(TRANSDUCER_PX)[0]) * 1000)


def main() -> None:
    cams = W.cameras()
    tiles = []
    for sess, edge, sim in POSITIONS:
        i = sess.removeprefix("pos")
        folder = C.DATA_DIR / "us_check" / sess
        img = cv2.imread(str(folder / "side.png"))
        real = json.loads((folder / "real_reading.json").read_text())["median_mm"]
        cam = gap_mm(cams, edge)
        y = TRANSDUCER_PX[1]
        cv2.line(img, (TRANSDUCER_PX[0], y - 70), (TRANSDUCER_PX[0], y + 70), (0, 0, 255), 3)
        cv2.line(img, (int(edge), y - 70), (int(edge), y + 70), (255, 0, 0), 3)
        cv2.arrowedLine(img, (TRANSDUCER_PX[0], y), (int(edge), y), (0, 160, 0), 3, tipLength=0.15)
        crop = img[400:680, 940:1440].copy()
        crop = cv2.resize(crop, (800, 448))
        lines = [f"position {i}", f"camera  {cam:5.1f} mm", f"real    {real:5.1f} mm",
                 f"sim     {'  -  ' if sim is None else f'{sim:5.1f}'} mm"]
        for k, s in enumerate(lines):
            cv2.putText(crop, s, (16, 44 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
            cv2.putText(crop, s, (16, 44 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        tiles.append(crop)
    out = C.REPO_ROOT / "docs" / "figures" / "real2sim" / "ultrasonic_positions.jpg"
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % 3)
    rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
