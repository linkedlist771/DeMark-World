from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.demark_world.configs import WATER_MARK_DETECT_YOLO_WEIGHTS, WATER_MARK_DETECT_YOLO_WEIGHTS_REMOTE_URL
from src.demark_world.utils.devices_utils import get_device
from src.demark_world.utils.download_utils import ensure_model_downloaded
from src.demark_world.utils.video_utils import VideoLoader

# based on the sora tempalte to detect the whole, and then got the icon part area.


class DeMarkWorldDetector:
    def __init__(self):
        ensure_model_downloaded(WATER_MARK_DETECT_YOLO_WEIGHTS, WATER_MARK_DETECT_YOLO_WEIGHTS_REMOTE_URL)
        logger.debug(f"Begin to load yolo water mark detet model.")
        self.model = YOLO(WATER_MARK_DETECT_YOLO_WEIGHTS)
        self.model.to(str(get_device()))
        self.model.eval()
        logger.debug(f"Yolo water mark detet model loaded from {WATER_MARK_DETECT_YOLO_WEIGHTS}.")

    def _parse_detect_results(self, result) -> dict:
        """Parse YOLO detection result into a standardized dictionary format."""
        if len(result.boxes) == 0:
            return {"detected": False, "bbox": None, "confidence": None, "center": None}

        box = result.boxes[0]
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        confidence = float(box.conf[0].cpu().numpy())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return {
            "detected": True,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": confidence,
            "center": (int(center_x), int(center_y)),
        }

    def detect(self, input_image: np.ndarray) -> dict:
        results = self.model(input_image, verbose=False)
        result = results[0]
        return self._parse_detect_results(result)

    def detect_batch(self, input_images: List[np.ndarray], batch_size: int) -> List[dict]:
        if not input_images:
            return []

        all_results = []

        for i in range(0, len(input_images), batch_size):
            batch_images = input_images[i : i + batch_size]
            batch_results = self.model(batch_images, verbose=False)

            for result in batch_results:
                parsed_result = self._parse_detect_results(result)
                all_results.append(parsed_result)

        return all_results


if __name__ == "__main__":
    pass
