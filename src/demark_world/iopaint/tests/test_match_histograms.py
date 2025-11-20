import pytest
import torch

from src.demark_world.iopaint.model_manager import ModelManager
from src.demark_world.iopaint.schema import HDStrategy, SDSampler
from src.demark_world.iopaint.tests.utils import assert_equal, check_device, current_dir, get_config


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_sd_match_histograms(device, sampler):
    sd_steps = check_device(device)

    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=7.5,
        sd_lcm_lora=False,
        sd_match_histograms=True,
        sd_sampler=sampler,
    )

    assert_equal(
        model,
        cfg,
        f"runway_sd_1_5_device_{device}_match_histograms.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
