"""Turn the JPEG frame folders of the recordings into one MP4 per camera, so the recordings
fit in git (the raw frames are ~21 GB).

    python real2sim/recording/make_videos.py                 # every recording that has frames
    python real2sim/recording/make_videos.py run08 static01  # only these
    python real2sim/recording/make_videos.py --crf 23

For real2sim/data/recordings/<run>/: writes top.mp4 / side.mp4 (H.264, 1920x1080, yuv420p).
Capture times: <cam>_frames.csv maps file names to `t`. `top_file` / `side_file` in the datasets
are file numbers: top/000142.jpg is frame 142 of top.mp4 (video frame N = file N; the
folder also holds the frames from before / after the recording window, which are not in the
CSVs). The video's own 30 fps clock is only for playback. To get a frame back:

    cap = cv2.VideoCapture("top.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, 142); ok, img = cap.read()

H.264 CRF 28 (preset slow): on run08, AprilTag corners from the decoded video differ from the
JPEG ones by a median 1 px (max 2 px) and 2 of 47 tags were missed. The original JPEGs stay
on the bench PC.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cameras"))
import camlib as C  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def encode(folder: Path, out: Path, crf: int) -> int:
    """Frames named 000000.jpg, 000001.jpg ... (contiguous) -> video frame N = file N."""
    files = sorted(folder.glob("*.jpg"))
    nums = [int(f.stem) for f in files]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise SystemExit(f"{folder}: frame numbers are not contiguous")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-framerate", "30", "-start_number", str(nums[0]),
           "-i", str(folder / "%06d.jpg"), "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    recs = C.DATA_DIR / "recordings"
    names = a.runs or sorted(p.name for p in recs.iterdir() if p.is_dir())
    for name in names:
        rec = recs / name
        for cam in ("top", "side"):
            if not (rec / cam).is_dir() or not any((rec / cam).glob("*.jpg")):
                continue
            out = rec / f"{cam}.mp4"
            n = encode(rec / cam, out, a.crf)
            print(f"{name}/{cam}.mp4: {n} frames, {out.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
