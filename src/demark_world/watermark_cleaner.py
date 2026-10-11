from src.demark_world.schemas import CleanerType
from src.demark_world.utils.devices_utils import is_bf16_supported


class WaterMarkCleaner:
    def __new__(
        cls,
        cleaner_type: CleanerType,
        enable_torch_compile: bool = True,
        use_bf16: bool = is_bf16_supported(),
    ):
        """Construct only the requested backend without importing unrelated backends."""
        if cleaner_type == CleanerType.LAMA:
            from src.demark_world.cleaner.lama_cleaner import LamaCleaner

            return LamaCleaner()
        elif cleaner_type == CleanerType.E2FGVI_HQ:
            from src.demark_world.cleaner.e2fgvi_hq_cleaner import (
                E2FGVIHDCleaner,
                E2FGVIHDConfig,
            )

            e2fgvi_hq_config = E2FGVIHDConfig(
                enable_torch_compile=enable_torch_compile, use_bf16=use_bf16
            )
            return E2FGVIHDCleaner(config=e2fgvi_hq_config)
        else:
            raise ValueError(f"Invalid cleaner type: {cleaner_type}")
