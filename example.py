from pathlib import Path
from src.demark_world.core import DeMarkWorld
from src.demark_world.schemas import CleanerType

if __name__ == "__main__":
    input_video = Path("resources/demo.mp4")
    output_video = Path("outputs/cleaned.mp4")

    # Option 1: LaMa (Fast)
    demarker = DeMarkWorld(cleaner_type=CleanerType.LAMA)
    
    # Option 2: E2FGVI_HQ (High Quality + Time Consistent)
    # demarker = DeMarkWorld(cleaner_type=CleanerType.E2FGVI_HQ)

    demarker.run(input_video, output_video)