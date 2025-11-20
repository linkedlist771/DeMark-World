from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from src.demark_world.configs import DEFAULT_WATERMARK_REMOVE_MODEL
from src.demark_world.iopaint.const import DEFAULT_MODEL_DIR
from src.demark_world.iopaint.download import cli_download_model, scan_models
from src.demark_world.iopaint.model_manager import ModelManager
from src.demark_world.iopaint.schema import InpaintRequest
from src.demark_world.utils.devices_utils import get_device

# This codebase is from https://github.com/Sanster/IOPaint#, thanks for their amazing work!


class LamaCleaner:
    def __init__(self):
        self.model = DEFAULT_WATERMARK_REMOVE_MODEL
        self.device = get_device()

        scanned_models = scan_models()
        if self.model not in [it.name for it in scanned_models]:
            logger.info(f"{self.model} not found in {DEFAULT_MODEL_DIR}, try to downloading")
            cli_download_model(self.model)
        self.model_manager = ModelManager(name=self.model, device=self.device)
        self.inpaint_request = InpaintRequest()

    def clean(self, input_image: np.array, watermark_mask: np.array) -> np.array:
        inpaint_result = self.model_manager(input_image, watermark_mask, self.inpaint_request)
        inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
        return inpaint_result
