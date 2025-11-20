from pathlib import Path

import pytest
import torch

from src.demark_world.iopaint.model_manager import ModelManager
from src.demark_world.iopaint.schema import HDStrategy
from src.demark_world.iopaint.tests.utils import assert_equal, check_device, current_dir, get_config

model_name = "timbrooks/instruct-pix2pix"


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("disable_nsfw", [True, False])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_instruct_pix2pix(device, disable_nsfw, cpu_offload):
    sd_steps = check_device(device)
    model = ModelManager(
        name=model_name,
        device=torch.device(device),
        disable_nsfw=disable_nsfw,
        sd_cpu_textencoder=False,
        cpu_offload=cpu_offload,
    )
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="What if it were snowing?",
        sd_steps=sd_steps,
    )

    name = f"device_{device}_disnsfw_{disable_nsfw}_cpu_offload_{cpu_offload}"

    assert_equal(
        model,
        cfg,
        f"instruct_pix2pix_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.3,
    )
