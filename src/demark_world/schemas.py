from enum import StrEnum


class CleanerType(StrEnum):
    """Supported video object-removal cleaner implementations."""

    LAMA = "lama"
    E2FGVI_HQ = "e2fgvi_hq"
