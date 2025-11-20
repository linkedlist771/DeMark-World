from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from src.demark_world.configs import E2FGVI_HQ_CHECKPOINT_PATH, E2FGVI_HQ_CHECKPOINT_REMOTE_URL
from src.demark_world.models.model.e2fgvi_hq import InpaintGenerator
from src.demark_world.utils.devices_utils import get_device
from src.demark_world.utils.download_utils import ensure_model_downloaded
from src.demark_world.utils.video_utils import merge_frames_with_overlap


def get_ref_index(
    frame_idx: int, neighbor_ids: List[int], length: int, ref_length: int, num_ref: int
) -> List[int]:
    # TODO: optimize the code later.
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
        end_idx = min(length, frame_idx + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


def numpy_to_tensor(frames_np, masks_np):
    """
    Convert numpy arrays to tensors
    frames_np: (T, H, W, 3) uint8 [0, 255]
    masks_np: (T, H, W) uint8 [0, 255]
    Returns: frames tensor (1, T, 3, H, W) [-1, 1], masks tensor (1, T, 1, H, W) [0, 1]
    """
    # Frames: (T, H, W, 3) -> (T, 3, H, W) -> (1, T, 3, H, W)
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).float()
    frames_tensor = frames_tensor / 255.0 * 2 - 1  # Normalize to [-1, 1]

    # Masks: (T, H, W) -> (T, 1, H, W) -> (1, T, 1, H, W)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(1).unsqueeze(0).float()
    masks_tensor = masks_tensor / 255.0  # Normalize to [0, 1]

    return frames_tensor, masks_tensor


# MODEL_DIR = Path("release_model")
# CKPT_PATH = MODEL_DIR / "E2FGVI-HQ-CVPR22.pth"

# TODO: RuntimeError: MPS: Unsupported Border padding mode
# mps doesn't work here.....
device = get_device()
if device.type == "mps":
    logger.warning(
        f"E2FGVI_HQ Cleaner doesn't support MPS, using CPU instead. But it is very very slow!!"
    )
    device = torch.device("cpu")


class E2FGVIHDConfig(BaseModel):
    ref_length: int = 10
    num_ref: int = -1
    neighbor_stride: int = 5
    chunk_size_ratio: float = 0.2  # TODO: this can be adjust as the VRAM
    overlap_ratio: int = 0.05


class E2FGVIHDCleaner:
    def __init__(
        self,
        ckpt_path: Path = E2FGVI_HQ_CHECKPOINT_PATH,
        config: E2FGVIHDConfig = E2FGVIHDConfig(),
    ):
        ensure_model_downloaded(ckpt_path, E2FGVI_HQ_CHECKPOINT_REMOTE_URL)
        self.model = InpaintGenerator().to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.config = config

    def process_frames_chunk(
        self,
        chunk_length: int,
        neighbor_stride: int,
        imgs_chunk: torch.Tensor,
        masks_chunk: torch.Tensor,
        binary_masks_chunk: np.ndarray,
        frames_np_chunk: np.ndarray,
        h: int,
        w: int,
    ) -> List[np.ndarray]:
        comp_frames_chunk = [None] * chunk_length

        for f in tqdm(
            range(0, chunk_length, neighbor_stride),
            desc=f"  Frame progress",
            position=1,
            leave=False,
        ):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(chunk_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f,
                neighbor_ids,
                chunk_length,
                self.config.ref_length,
                self.config.num_ref,
            )
            selected_imgs = imgs_chunk[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_chunk[:1, neighbor_ids + ref_ids, :, :, :]

            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                    :, :, :, : h + h_pad, :
                ]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                    :, :, :, :, : w + w_pad
                ]
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(np.uint8) * binary_masks_chunk[
                        idx
                    ] + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                    if comp_frames_chunk[idx] is None:
                        comp_frames_chunk[idx] = img
                    else:
                        comp_frames_chunk[idx] = (
                            comp_frames_chunk[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )

        return comp_frames_chunk

    def clean(self, frames: np.ndarray, masks: np.ndarray) -> List[np.ndarray]:
        video_length = len(frames)
        chunk_size = int(self.config.chunk_size_ratio * video_length)
        overlap_size = int(self.config.overlap_ratio * video_length)
        num_chunks = int(np.ceil(video_length / (chunk_size - overlap_size)))
        h, w = frames[0].shape[:2]
        # Convert to tensors
        imgs_all, masks_all = numpy_to_tensor(frames, masks)
        # Prepare binary masks for compositing
        binary_masks = np.expand_dims(masks > 0, axis=-1).astype(np.uint8)  # (T, H, W, 1)
        comp_frames = [None] * video_length
        logger.debug(
            f"Processing {video_length} frames in {num_chunks} chunks (chunk_size={chunk_size}, overlap={overlap_size})"
        )

        for chunk_idx in tqdm(range(num_chunks), desc="Chunk", position=0, leave=True):
            start_idx = chunk_idx * (chunk_size - overlap_size)
            end_idx = min(start_idx + chunk_size, video_length)
            actual_chunk_size = end_idx - start_idx
            # logger.debug(f'\nProcessing chunk {chunk_idx + 1}/{num_chunks}: frames {start_idx}-{end_idx}')
            # Extract chunk data
            imgs_chunk = imgs_all[:, start_idx:end_idx, :, :, :].to(device)
            masks_chunk = masks_all[:, start_idx:end_idx, :, :, :].to(device)
            frames_np_chunk = frames[start_idx:end_idx]
            binary_masks_chunk = binary_masks[start_idx:end_idx]
            # Process chunk
            comp_frames_chunk = self.process_frames_chunk(
                actual_chunk_size,
                self.config.neighbor_stride,
                imgs_chunk,
                masks_chunk,
                binary_masks_chunk,
                frames_np_chunk,
                h,
                w,
            )
            # Merge results with blending in overlap region
            comp_frames = merge_frames_with_overlap(
                result_frames=comp_frames,
                chunk_frames=comp_frames_chunk,
                start_idx=start_idx,
                overlap_size=overlap_size,
                is_first_chunk=(chunk_idx == 0),
            )
            # Clear GPU memory
            del imgs_chunk, masks_chunk, comp_frames_chunk
            try:
                torch.cuda.empty_cache()
            except:
                pass
        return comp_frames


if __name__ == "__main__":
    #       --frames examples/extract_frame_and_mask_frames.npy \
    #   --masks examples/extract_frame_and_mask_masks.npy \
    import os

    import cv2

    frames_path = Path("examples/extract_frame_and_mask_frames.npy")
    masks_path = Path("examples/extract_frame_and_mask_masks.npy")
    frames_np = np.load(frames_path)
    masks_np = np.load(masks_path)
    # Convert BGR to RGB if frames are saved in BGR format
    frames_np = frames_np[:, :, :, ::-1].copy()
    cleaner = E2FGVIHDCleaner()
    comp_frames = cleaner.clean(frames_np, masks_np)

    # Save the result as video
    fps = 30
    output_video_path = "results/output.mp4"
    h, w = frames_np[0].shape[:2]

    os.makedirs("results", exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in comp_frames:
        # Convert RGB to BGR for OpenCV
        writer.write(frame.astype(np.uint8)[:, :, ::-1])
    writer.release()
    logger.info(f"Video saved to: {output_video_path}")
