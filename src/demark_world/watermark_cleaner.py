from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np

from src.demark_world.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner
from src.demark_world.cleaner.lama_cleaner import LamaCleaner
from src.demark_world.schemas import CleanerType


class WaterMarkCleaner:
    def __new__(cls, cleaner_type: CleanerType):
        match cleaner_type:
            case CleanerType.LAMA:
                return LamaCleaner()
            case CleanerType.E2FGVI_HQ:
                return E2FGVIHDCleaner()
            case _:
                raise ValueError(f"Invalid cleaner type: {cleaner_type}")
