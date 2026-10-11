# Opt-in timed watermark regions

The default detector workflow remains available. When a watermark appears intermittently, or the detector selects captions, supply reviewed boxes with individual start/stop frames instead. Only those pixels during those intervals are passed to LaMa. No detector weights are loaded when `automatic_detection=False`.

```python
from pathlib import Path
from src.demark_world.core import DeMarkWorld

engine = DeMarkWorld(automatic_detection=False)
engine.run(
    Path("input.mp4"),
    Path("cleaned.mp4"),
    watermark_regions=[
        {"bbox": [0.78, 0.04, 0.97, 0.13], "start_frame": 300, "end_frame": 360},
        {"bbox": [0.78, 0.04, 0.97, 0.13], "start_frame": 720, "end_frame": 780},
    ],
)
```

Coordinates are normalized `[left, top, right, bottom]`. Frame numbers are zero-based: start inclusive, end exclusive. A bare four-coordinate box applies throughout the video. Invalid boxes, invalid intervals, and automatic suggestions lacking an identity-check marker are rejected before encoding. This API does not itself detect or verify branding; callers remain responsible for supplying reviewed regions.

This first contribution supports LaMa and MP4 output. It streams frames, preserves pixels outside active masks before encoding, muxes available audio, fully decodes the result, and publishes only on success. Existing outputs and input files are preserved. Video encoding is lossy, so pixel preservation before encoding does not mean a bit-identical output. Constant-frame-rate inputs are the tested case; variable-frame-rate timing needs further work. CUDA/E2FGVI timed processing and UI controls are outside this PR.

## Validation and finishing failures

The input is decoded once to determine its actual frame count before interval validation, then reopened for restoration. This adds one decode pass with bounded memory and avoids relying on backend frame-count estimates. The second pass must decode the same number of frames or processing fails without publication.

Coordinate conversion errors raise `ValueError` with a clear message. Frames at `end_frame` are excluded; an interval extending beyond the decoded input is rejected before encoding.

Audio muxing and final output verification each have a 3,600-second timeout. If either stage exceeds its deadline, `subprocess.TimeoutExpired` propagates, the encoder has already been stopped, temporary files are removed, and no output is published. A one-hour finishing deadline does not bound model inference or the initial decode pass.

## Tests

```bash
uv run pytest tests/test_timed_regions.py tests/test_placeholder.py
```

The media integration tests require FFmpeg on PATH and use synthetic fixtures. They test timed masks, RGB/BGR compositing, audio, decoded frame counts despite inaccurate metadata, source preservation, malformed coordinates, both finishing-stage timeouts, failed inference cleanup, and existing-output protection. They inject a cleaner, so no model downloads are required. These injected-cleaner tests verify the processing contract, not model quality. Actual LaMa inference and full CUDA/E2FGVI coverage are outside this test suite.
