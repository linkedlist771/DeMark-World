import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.demark_world.timed_cleaning import clean_timed_frame, clean_timed_video
from src.demark_world.timed_regions import compile_regions


class Cleaner:
    def __init__(self):
        """Record how often restoration is requested."""
        self.calls = 0

    def clean(self, image, mask):
        """Return altered RGB pixels to exercise mask compositing."""
        self.calls += 1
        # Deliberately alter everything; compositing must protect the rest.
        return np.full_like(image, [200, 80, 20])


def test_masks_apply_only_inside_their_independent_frame_ranges():
    """Check exact boundaries and protect pixels outside each active rectangle."""
    frame = np.full((80, 120, 3), [10, 40, 70], dtype=np.uint8)
    regions = compile_regions(
        [
            {"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 2, "end_frame": 4},
            {"bbox": [0.7, 0.6, 0.9, 0.9], "start_frame": 5, "end_frame": 7},
        ],
        120,
        80,
        8,
    )
    cleaner = Cleaner()
    for index in range(8):
        output = clean_timed_frame(frame, index, regions, cleaner)
        if index in [0, 1, 4, 7]:
            assert np.array_equal(output, frame)
        else:
            active = regions[0] if index in [2, 3] else regions[1]
            x1, y1, x2, y2 = active.bbox
            expected = frame.copy()
            expected[y1:y2, x1:x2] = [20, 80, 200]
            assert np.array_equal(output, expected)
    assert cleaner.calls == 4


@pytest.mark.parametrize("start,end", [(-1, 10), (5, 5), (10, 9), (0, 31), (1.5, 10), (True, 10)])
def test_invalid_times_are_rejected(start, end):
    """Reject reversed, empty, non-integer and out-of-video frame intervals."""
    with pytest.raises(ValueError):
        compile_regions(
            [{"bbox": [0.1, 0.1, 0.2, 0.2], "start_frame": start, "end_frame": end}], 100, 100, 30
        )


@pytest.mark.parametrize("bbox", [[0, 0, 2, 1], [0, 0, 0, 1], [0, 0, float("nan"), 1]])
def test_invalid_boxes_are_rejected(bbox):
    """Reject empty, non-finite and out-of-frame rectangles."""
    with pytest.raises(ValueError):
        compile_regions([bbox], 100, 100, 30)


def test_legacy_boxes_cover_the_full_video_and_unchecked_suggestions_are_rejected():
    """Retain full-video boxes but reject suggestions lacking identity attestation."""
    region = compile_regions([[0.1, 0.1, 0.2, 0.2]], 100, 100, 30)[0]
    assert region.active(0) and region.active(29) and not region.active(30)
    with pytest.raises(ValueError, match="identity"):
        compile_regions([{"bbox": [0.1, 0.1, 0.2, 0.2], "source": "suggested"}], 100, 100, 30)


@pytest.fixture
def video(tmp_path):
    """Generate a 20-frame synthetic MP4 with an audio track, without models."""
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for the media integration checks")
    path = tmp_path / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=120x80:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_streamed_output_preserves_audio_frame_count_and_source(video, tmp_path):
    """Exercise the public API and fully decode both output tracks."""
    original = hashlib.sha256(video.read_bytes()).hexdigest()
    output = tmp_path / "cleaned.mp4"
    progress = []
    cleaner = Cleaner()
    from src.demark_world.core import DeMarkWorld

    engine = DeMarkWorld(automatic_detection=False, cleaner=cleaner)
    engine.run(
        video,
        output,
        watermark_regions=[{"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 2, "end_frame": 4}],
        progress_callback=progress.append,
    )
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 20
    capture.release()
    assert cleaner.calls == 2 and progress[-1] == 100
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original
    assert not list(tmp_path.glob(".demark-timed-*"))


def test_failure_never_publishes_output_or_leaves_encoder_files(video, tmp_path):
    """Keep the source and remove partial files after cleaner failure."""

    class FailingCleaner:
        def clean(self, image, mask):
            """Simulate an inference failure before output publication."""
            raise RuntimeError("inference failed")

    output = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError, match="inference failed"):
        clean_timed_video(video, output, [[0.1, 0.1, 0.3, 0.3]], FailingCleaner())
    assert video.exists() and not output.exists()
    assert not list(tmp_path.glob(".demark-timed-*"))


def test_existing_output_is_preserved(video, tmp_path):
    """Refuse to replace a pre-existing output."""
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep this output")
    with pytest.raises(FileExistsError):
        clean_timed_video(video, output, [[0.1, 0.1, 0.3, 0.3]], Cleaner())
    assert output.read_bytes() == b"keep this output"


def test_engine_does_not_load_detector_or_optional_backends_for_injected_cleaner(monkeypatch):
    """Disable detector construction and require explicit regions for this mode."""
    from src.demark_world import core

    def forbidden(*args, **kwargs):
        """Fail if disabled automatic detection tries to construct a detector."""
        raise AssertionError("Detector weights must not load")

    monkeypatch.setattr(core, "DeMarkWorldDetector", forbidden)
    engine = core.DeMarkWorld(automatic_detection=False, cleaner=Cleaner())
    assert engine.detector is None
    with pytest.raises(ValueError, match="watermark_regions"):
        engine.run("unused.mp4", "unused-output.mp4")


@pytest.mark.parametrize("coordinate", [None, [], {}, "not a number"])
def test_non_numeric_coordinates_raise_clear_validation_errors(coordinate):
    """Malformed coordinate types all follow the documented ValueError contract."""
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        compile_regions([[0, 0, coordinate, 1]], 100, 100, 30)


def test_numeric_strings_remain_supported():
    """Normalization preserves conversion of valid numeric strings."""
    region = compile_regions([["0.1", "0.1", "0.2", "0.2"]], 100, 100, 30)[0]
    assert region.bbox == (10, 10, 20, 20)


@pytest.mark.parametrize("reported_count", [0, 10, 100, float("nan")])
def test_interval_validation_uses_decoded_frames_not_metadata(
    video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_count: float,
) -> None:
    """The final real frame remains selectable despite missing or incorrect metadata."""
    from src.demark_world import timed_cleaning

    real_capture = cv2.VideoCapture

    class MetadataCapture:
        """Expose inaccurate frame metadata without changing actual decoding."""

        def __init__(self, path: str) -> None:
            """Wrap a real decoder for the synthetic input."""
            self.capture = real_capture(path)

        def get(self, prop: int) -> float:
            """Substitute only the unreliable frame-count property."""
            return reported_count if prop == cv2.CAP_PROP_FRAME_COUNT else self.capture.get(prop)

        def isOpened(self) -> bool:
            """Report whether the underlying decoder opened."""
            return self.capture.isOpened()

        def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
            """Decode a real frame unchanged."""
            return self.capture.read()

        def release(self) -> None:
            """Close the real decoder."""
            self.capture.release()

    monkeypatch.setattr(timed_cleaning.cv2, "VideoCapture", MetadataCapture)
    cleaner = Cleaner()
    output = tmp_path / "metadata-mismatch.mp4"
    clean_timed_video(
        video,
        output,
        [
            {"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 19, "end_frame": 20},
        ],
        cleaner,
    )
    capture = real_capture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 20 and cleaner.calls == 1


def test_interval_past_decoded_end_is_rejected_before_encoding(video, tmp_path, monkeypatch):
    """Intervals cannot refer to frames beyond the decoded input."""
    from src.demark_world import timed_cleaning

    def forbidden_encoder(*args, **kwargs):
        """Fail if invalid input reaches encoding."""
        raise AssertionError("Encoder must not start for an invalid interval")

    monkeypatch.setattr(timed_cleaning.subprocess, "Popen", forbidden_encoder)
    with pytest.raises(ValueError, match="valid start and stop"):
        clean_timed_video(
            video,
            tmp_path / "invalid.mp4",
            [
                {"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 19, "end_frame": 21},
            ],
            Cleaner(),
        )
    assert not list(tmp_path.glob(".demark-timed-*"))


@pytest.mark.parametrize("stage", ["mux", "verify"])
def test_postprocessing_timeouts_preserve_source_and_remove_partial_files(
    video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Both finishing stages have deadlines and clean up if that deadline expires."""
    from src.demark_world import timed_cleaning

    real_run = subprocess.run
    original = hashlib.sha256(video.read_bytes()).hexdigest()
    calls = []

    def run_with_timeout(
        args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        """Inject a deadline expiry only at the requested finishing stage."""
        assert kwargs["timeout"] == timed_cleaning.FFMPEG_POSTPROCESS_TIMEOUT_SECONDS
        current_stage = "verify" if "-xerror" in args else "mux"
        calls.append(current_stage)
        if current_stage == stage:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return real_run(args, **kwargs)

    monkeypatch.setattr(timed_cleaning.subprocess, "run", run_with_timeout)
    output = tmp_path / "timeout.mp4"
    with pytest.raises(subprocess.TimeoutExpired):
        clean_timed_video(video, output, [[0.1, 0.1, 0.3, 0.3]], Cleaner())
    assert calls == (["mux"] if stage == "mux" else ["mux", "verify"])
    assert not output.exists() and not list(tmp_path.glob(".demark-timed-*"))
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original
