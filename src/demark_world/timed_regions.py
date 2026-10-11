"""Validated boxes with exact, zero-based frame ranges.

The end frame is exclusive. Legacy rectangles still mean the entire video.
"""

import math
from dataclasses import dataclass


def region_boxes(regions, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Validate normalized rectangles and convert them to enclosing pixel bounds.

    Raise ValueError for malformed, non-numeric, non-finite or out-of-frame
    coordinates. Numeric strings retain their float-conversion behavior.
    """
    if not isinstance(regions, list | tuple) or not regions:
        raise ValueError("Draw a box around the watermark before processing.")
    boxes = []
    for region in regions:
        if not isinstance(region, list | tuple) or len(region) != 4:
            raise ValueError("Each watermark area must contain four coordinates.")
        try:
            x1, y1, x2, y2 = map(float, region)
        except (TypeError, ValueError) as e:
            raise ValueError("Watermark coordinates must be numbers.") from e
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (
            0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        ):
            raise ValueError("Watermark areas must be inside the video frame.")
        boxes.append(
            (
                math.floor(x1 * width),
                math.floor(y1 * height),
                math.ceil(x2 * width),
                math.ceil(y2 * height),
            )
        )
    return boxes


@dataclass(frozen=True)
class TimedRegion:
    """Immutable pixel rectangle active over an end-exclusive frame interval."""

    bbox: tuple[int, int, int, int]
    start_frame: int
    end_frame: int

    def active(self, frame: int) -> bool:
        """Return whether a zero-based frame lies within this end-exclusive interval."""
        return self.start_frame <= frame < self.end_frame


def normalize_regions(regions, total_frames: int) -> list[dict]:
    """Validate frame intervals while retaining labels and verification metadata.

    Bare rectangles cover total_frames. Dictionary regions use inclusive
    start_frame and exclusive end_frame; both must be integers within the
    decoded input. Verification markers describe caller-supplied checks.
    """
    if not isinstance(regions, list | tuple) or not regions:
        raise ValueError("Select at least one watermark area.")
    normalized = []
    for index, item in enumerate(regions):
        if isinstance(item, dict):
            box = item.get("bbox")
            start = item.get("start_frame", 0)
            end = item.get("end_frame", total_frames)
        else:
            box, start, end = item, 0, total_frames
        region_boxes([box], 1000, 1000)
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not (
                isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= total_frames
            )
        ):
            raise ValueError("Every area needs a valid start and stop frame inside the video.")
        normalized.append(
            {
                "id": str(item.get("id", f"area-{index}"))
                if isinstance(item, dict)
                else f"area-{index}",
                "bbox": list(map(float, box)),
                "start_frame": start,
                "end_frame": end,
                "label": str(item.get("label", "Watermark"))[:100]
                if isinstance(item, dict)
                else "Watermark",
                "source": "suggested"
                if isinstance(item, dict) and item.get("source") == "suggested"
                else "manual",
            }
        )
        if isinstance(item, dict) and item.get("verification") == "named-ocr-v1":
            normalized[-1]["verification"] = "named-ocr-v1"
        if isinstance(item, dict) and item.get("timing_verified") is True:
            normalized[-1]["timing_verified"] = True
    return normalized


def compile_regions(regions, width: int, height: int, total_frames: int) -> list[TimedRegion]:
    """Compile validated rectangles and reject suggestions without identity checks.

    This validates the caller-provided marker, not the branding itself.
    Callers must supply reviewed regions and the actual decoded frame count.
    """
    regions = normalize_regions(regions, total_frames)
    if any(r["source"] == "suggested" and r.get("verification") != "named-ocr-v1" for r in regions):
        raise ValueError(
            "These old automatic suggestions have no watermark identity check. "
            "Rescan for the specific watermark or edit the boxes manually."
        )
    return [
        TimedRegion(region_boxes([r["bbox"]], width, height)[0], r["start_frame"], r["end_frame"])
        for r in regions
    ]
