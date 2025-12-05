import hashlib
import json
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from src.demark_world.configs import (
    WATER_MARK_DETECT_YOLO_WEIGHTS,
    WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON,
)


def ensure_model_downloaded(model_path: Path, url: str, force_download: bool = False):
    if not model_path.exists() or force_download:
        logger.debug(f"Downloading model from {url}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = model_path.with_suffix(".tmp")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(temp_file, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            if model_path.exists():
                model_path.unlink()
            temp_file.rename(model_path)
            logger.success(f"✓ Model downloaded: {model_path}")
            return True
        except requests.exceptions.RequestException as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Download failed: {e}")
    else:
        logger.debug(f"Model already exists: {model_path}")


def generate_sha256_hash(file_path: Path) -> str:
    """Generate SHA256 hash for a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _get_local_hash() -> str | None:
    """Get local model hash from JSON file, generate if not exists."""
    if WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON.exists():
        with WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON.open("r") as f:
            return json.load(f).get("sha256")

    if WATER_MARK_DETECT_YOLO_WEIGHTS.exists():
        logger.info(f"Generating SHA256 hash for {WATER_MARK_DETECT_YOLO_WEIGHTS}")
        return generate_sha256_hash(WATER_MARK_DETECT_YOLO_WEIGHTS)

    return None



def download_detector_weights(force_download: bool = False):
    raise NotImplementedError("Downloading detector weights is not implemented yet.")
