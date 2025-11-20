from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
print(ROOT)

RESOURCES_DIR = ROOT / "resources"
WATER_MARK_TEMPLATE_IMAGE_PATH = RESOURCES_DIR / "watermark_template.png"

CHECKPOINT_DIR = RESOURCES_DIR / "checkpoint"
WATER_MARK_DETECT_YOLO_WEIGHTS = CHECKPOINT_DIR / "watermark-best.pt"
WATER_MARK_DETECT_YOLO_WEIGHTS_REMOTE_URL = "https://github.com/linkedlist771/DeMark-World/releases/download/V0.0.1/watermark-best.pt"
WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON = RESOURCES_DIR / "model_version.json

CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
SPYNET_CHECKPOINT_PATH = CHECKPOINT_DIR / "spynet_20210409-c6c1bd09.pth"
# release_model/E2FGVI-HQ-CVPR22.pth
E2FGVI_HQ_CHECKPOINT_PATH = CHECKPOINT_DIR / "E2FGVI-HQ-CVPR22.pth"
E2FGVI_HQ_CHECKPOINT_REMOTE_URL = "https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/E2FGVI-HQ-CVPR22.pth"


OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


DEFAULT_WATERMARK_REMOVE_MODEL = "lama"

WORKING_DIR = ROOT / "working_dir"
WORKING_DIR.mkdir(exist_ok=True, parents=True)

LOGS_PATH = ROOT / "logs"
LOGS_PATH.mkdir(exist_ok=True, parents=True)

DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True, parents=True)

SQLITE_PATH = DATA_PATH / "db.sqlite3"

if __name__ == "__main__":
    from loguru import logger

    # for val in globals():
    # logger.debug(f"val: {val}")
    # print the globals one by one
    logger.debug(globals())
    # for k, v in globals().items():
    # logger.debug(f"{k}:  {v}")
