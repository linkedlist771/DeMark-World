"""Opt-in LaMa restoration restricted to reviewed boxes and frame intervals."""

import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.demark_world.timed_regions import compile_regions

# Allow long local videos while bounding malformed-input mux/verification hangs.
FFMPEG_POSTPROCESS_TIMEOUT_SECONDS = 3600


def clean_timed_frame(frame, frame_index, regions, cleaner):
    """Keep every pixel outside the currently active masks unchanged."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for region in regions:
        if region.active(frame_index):
            x1, y1, x2, y2 = region.bbox
            mask[y1:y2, x1:x2] = 255
    if not mask.any():
        return frame
    # Video decoding/encoding uses BGR; the existing LaMa cleaner uses RGB.
    restored = cleaner.clean(frame[:, :, ::-1].copy(), mask)
    if restored.shape != frame.shape or restored.dtype != np.uint8:
        raise ValueError("Cleaner returned an invalid frame")
    result = frame.copy()
    result[mask > 0] = restored[:, :, ::-1][mask > 0]
    return result


def _count_decoded_frames(capture: cv2.VideoCapture) -> int:
    """Count successful decodes with bounded memory, ignoring container estimates."""
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            return count
        count += 1


def clean_timed_video(input_path, output_path, regions, cleaner, progress_callback=None):
    """Stream an MP4 output without loading detector weights or overwriting files.

    Start frames are inclusive, end frames exclusive. Audio is preserved through
    AAC muxing; encoding, decoding, and mux failures never publish a partial file.
    An initial decode pass supplies exact interval bounds. Muxing and final
    verification each have a one-hour timeout; timeout failures are propagated.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Timed-region output must use the MP4 container")
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise FileExistsError("Choose a new output path; existing files are preserved")
    capture = cv2.VideoCapture(str(input_path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open the input video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input video has invalid metadata")
        total = _count_decoded_frames(capture)
        if total == 0:
            raise ValueError("Input video contains no decodable frames")
        compiled = compile_regions(regions, width, height, total)
        # Reopen rather than relying on backend-dependent seek-to-zero behavior.
        capture.release()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Cannot reopen the input video for processing")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".demark-timed-", dir=output_path.parent) as work:
            work = Path(work)
            video, merged = work / "video.mp4", work / "merged.mp4"
            with (work / "encoder.log").open("w+b") as errors:
                process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "slow",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        str(video),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                )
                try:
                    count = 0
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        process.stdin.write(
                            clean_timed_frame(frame, count, compiled, cleaner).tobytes()
                        )
                        count += 1
                        if progress_callback and count % 10 == 0:
                            progress_callback(min(90, int(count / total * 90)))
                    if count != total:
                        raise ValueError(f"Decoded {count} frames; expected {total}")
                    process.stdin.close()
                    if process.wait() != 0:
                        errors.seek(0)
                        raise RuntimeError(errors.read()[-4000:].decode(errors="replace"))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-i",
                    str(input_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    str(merged),
                ],
                check=True,
                capture_output=True,
                timeout=FFMPEG_POSTPROCESS_TIMEOUT_SECONDS,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(merged),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-f",
                    "null",
                    "-",
                ],
                check=True,
                capture_output=True,
                timeout=FFMPEG_POSTPROCESS_TIMEOUT_SECONDS,
            )
            # Same-filesystem hard link is atomic and refuses a destination that
            # appeared while processing. Temporary files are removed afterward.
            os.link(merged, output_path)
        if progress_callback:
            progress_callback(100)
    finally:
        capture.release()
