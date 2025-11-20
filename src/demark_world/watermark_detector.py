from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO
from loguru import logger
from src.demark_world.configs import WATER_MARK_DETECT_YOLO_WEIGHTS
from src.demark_world.utils.devices_utils import get_device

# from src.demark_world.utils.download_utils import download_detector_weights
from src.demark_world.utils.video_utils import VideoLoader

# based on the sora tempalte to detect the whole, and then got the icon part area.


class SoraWaterMarkDetector:
    def __init__(self):
        # download_detector_weights()
        logger.debug(f"Begin to load yolo water mark detet model.")
        self.model = YOLO(WATER_MARK_DETECT_YOLO_WEIGHTS)
        self.model.to(str(get_device()))
        self.model.eval()
        logger.debug(f"Yolo water mark detet model loaded from {WATER_MARK_DETECT_YOLO_WEIGHTS}.")

        self.model.eval()

    def detect(self, input_image: np.ndarray):
        # import cv2
        # # cv2.imshow("input_image", input_image)
        # cv2.imwrite("input_image.png", input_image)
        # raise RuntimeError()

        results = self.model.predict(source=input_image, conf=0.05, verbose=False, stream=False)
        # logger.error(f"input_image.shape:{input_image.shape}\nresults: {results}")

        result = results[0]

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


if __name__ == "__main__":
    pass
