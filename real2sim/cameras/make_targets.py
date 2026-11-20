"""Generate the printable targets.

    python real2sim/cameras/make_targets.py               # US Letter (default)
    python real2sim/cameras/make_targets.py --paper a4

Writes to real2sim/cameras/targets/:
  board_<paper>.pdf  ChArUco board from rig.yaml, landscape. Print at 100% / "actual
                   size" (NOT "fit to page"), glue onto a flat rigid backing, then measure
                   a square with calipers and put the real size into rig.yaml.
  cube_tags.pdf    Six AprilTag 36h11 tags (ids 0-5) for the 36 mm cube, with cut lines.
  cube_tags.json   Which tag goes on which cube face (cube body frame of the sim).
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import camlib as C

DPI = 600
MM = DPI / 25.4  # pixels per mm
OUT = C.HERE / "targets"
PAPERS_MM = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}  # portrait (w, h)

# Sim cube body frame: red face is local +z (see scene_right_cube_orientation.xml).
CUBE_FACES = {0: "+z (red face)", 1: "-z (opposite red)", 2: "+x", 3: "-x", 4: "+y", 5: "-y"}


def font(size_mm: float) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=int(size_mm * MM))
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def page(w_mm: float, h_mm: float) -> Image.Image:
    return Image.new("L", (int(round(w_mm * MM)), int(round(h_mm * MM))), 255)


def save_pdf(img: Image.Image, path) -> None:
    img.convert("RGB").save(path, "PDF", resolution=DPI)


def make_board_pdf(board_cfg: dict, paper: str = "letter") -> None:
    pw, ph = PAPERS_MM[paper]
    img = page(ph, pw)  # landscape
    board = C.board_image(board_cfg, MM)
    bh, bw = board.shape
    if bw > img.width - 12 * MM or bh > img.height - 30 * MM:
        raise SystemExit(f"board {bw / MM:.0f}x{bh / MM:.0f} mm does not fit on {paper}; reduce square_mm in rig.yaml")
    x0 = (img.width - bw) // 2
    y0 = (img.height - bh) // 2 - int(6 * MM)
    img.paste(Image.fromarray(board), (x0, y0))
    d = ImageDraw.Draw(img)
    # 100 mm check bar: if it does not measure 100 mm the printer scaled the page
    bx, by = x0, y0 + bh + int(6 * MM)
    d.rectangle([bx, by, bx + int(100 * MM), by + int(2 * MM)], fill=0)
    txt = (f"ChArUco {board_cfg['squares_x']}x{board_cfg['squares_y']}  square {board_cfg['square_mm']} mm  "
           f"marker {board_cfg['marker_mm']} mm  {board_cfg['dictionary']}   "
           f"bar above = 100 mm (print at 100%, measure, put the real square size in rig.yaml)")
    d.text((bx, by + int(4 * MM)), txt, fill=0, font=font(3.2))
    save_pdf(img, OUT / f"board_{paper}.pdf")


def make_cube_tags(face_mm: float, tag_mm: float, paper: str = "letter") -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = page(*PAPERS_MM[paper])
    d = ImageDraw.Draw(img)
    face_px, tag_px = int(round(face_mm * MM)), int(round(tag_mm * MM))
    gap = int(25 * MM)
    left = (img.width - (3 * face_px + 2 * gap)) // 2
    top = int(40 * MM)
    for i in range(6):
        r, c = divmod(i, 3)
        fx, fy = left + c * (face_px + gap), top + r * (face_px + gap + int(10 * MM))
        # cut line = face outline (dashed grey)
        for k in range(0, face_px, int(2 * MM)):
            for (ax, ay, bx2, by2) in [(fx + k, fy, fx + k + int(MM), fy), (fx + k, fy + face_px, fx + k + int(MM), fy + face_px),
                                       (fx, fy + k, fx, fy + k + int(MM)), (fx + face_px, fy + k, fx + face_px, fy + k + int(MM))]:
                d.line([ax, ay, bx2, by2], fill=140, width=2)
        tag = cv2.aruco.generateImageMarker(dictionary, i, tag_px, borderBits=1)
        off = (face_px - tag_px) // 2
        img.paste(Image.fromarray(tag), (fx + off, fy + off))
        d.text((fx, fy + face_px + int(2 * MM)), f"id {i}: face {CUBE_FACES[i]}", fill=0, font=font(3.0))
    d.text((left, int(15 * MM)),
           f"AprilTag 36h11, tag {tag_mm} mm (black square), face {face_mm} mm. Print at 100%. "
           f"Tag top edge = text side up.", fill=0, font=font(3.2))
    save_pdf(img, OUT / "cube_tags.pdf")
    (OUT / "cube_tags.json").write_text(json.dumps({
        "family": "DICT_APRILTAG_36h11", "tag_mm": tag_mm, "face_mm": face_mm,
        "faces": {str(k): v for k, v in CUBE_FACES.items()},
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default=C.HERE / "rig.yaml")
    ap.add_argument("--cube-mm", type=float, default=36.0, help="cube edge (sim cube: 36 mm)")
    ap.add_argument("--tag-mm", type=float, default=28.0, help="AprilTag black-square size on each face")
    ap.add_argument("--paper", choices=sorted(PAPERS_MM), default="letter")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    make_board_pdf(C.load_rig(args.rig)["board"], args.paper)
    make_cube_tags(args.cube_mm, args.tag_mm, args.paper)
    print(f"wrote {OUT / f'board_{args.paper}.pdf'}, {OUT / 'cube_tags.pdf'}, {OUT / 'cube_tags.json'}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
