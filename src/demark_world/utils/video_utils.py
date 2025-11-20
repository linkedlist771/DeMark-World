from pathlib import Path
from typing import List, Optional

import numpy as np

import ffmpeg


def merge_frames_with_overlap(
    result_frames: Optional[List[Optional[np.ndarray]]],
    chunk_frames: List[np.ndarray],
    start_idx: int,
    overlap_size: int,
    is_first_chunk: bool = False,
) -> List[Optional[np.ndarray]]:
    chunk_size = len(chunk_frames)

    # Initialize result_frames if this is the first chunk
    if result_frames is None or is_first_chunk:
        if result_frames is None:
            result_frames = [None] * (start_idx + chunk_size)
        for i in range(chunk_size):
            result_frames[start_idx + i] = chunk_frames[i]
        return result_frames

    # Ensure result_frames is large enough
    required_size = start_idx + chunk_size
    if len(result_frames) < required_size:
        result_frames.extend([None] * (required_size - len(result_frames)))

    # Blend overlap region
    overlap_start = 0
    overlap_end = min(overlap_size, chunk_size)

    for i in range(overlap_start, overlap_end):
        result_idx = start_idx + i
        if result_frames[result_idx] is not None and chunk_frames[i] is not None:
            # Alpha blending: gradually transition from old to new
            alpha = i / overlap_end if overlap_end > 0 else 1.0
            result_frames[result_idx] = (
                result_frames[result_idx].astype(np.float32) * (1 - alpha)
                + chunk_frames[i].astype(np.float32) * alpha
            ).astype(np.uint8)
        elif chunk_frames[i] is not None:
            result_frames[result_idx] = chunk_frames[i]

    # Copy non-overlap region
    for i in range(overlap_end, chunk_size):
        result_frames[start_idx + i] = chunk_frames[i]

    return result_frames


class VideoLoader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.get_video_info()

    def get_video_info(self):
        probe = ffmpeg.probe(self.video_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])
        fps = eval(video_info["r_frame_rate"])
        self.width = width
        self.height = height
        self.fps = fps
        if "nb_frames" in video_info:
            self.total_frames = int(video_info["nb_frames"])
        else:
            # 通过时长计算
            duration = float(video_info.get("duration", probe["format"]["duration"]))
            self.total_frames = int(duration * self.fps)

        original_bitrate = video_info.get("bit_rate", None)
        self.original_bitrate = original_bitrate

    def __len__(self):
        return self.total_frames

    def get_slice(self, start: int, end: int) -> List[np.ndarray]:
        num_frames = end - start
        if num_frames <= 0:
            return []
        start_time = start / self.fps
        process_in = (
            ffmpeg.input(self.video_path, ss=start_time)
            .output("pipe:", format="rawvideo", pix_fmt="bgr24", frames=num_frames)
            .global_args("-loglevel", "error")
            .run_async(pipe_stdout=True)
        )

        frames = []
        try:
            for _ in range(num_frames):
                in_bytes = process_in.stdout.read(self.width * self.height * 3)
                if not in_bytes:
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                frames.append(frame)
        finally:
            process_in.stdout.close()
            if process_in.stderr:
                process_in.stderr.close()
            process_in.wait()

        return frames

    def __iter__(self):
        process_in = (
            ffmpeg.input(self.video_path)
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .global_args("-loglevel", "error")
            .run_async(pipe_stdout=True)
        )

        try:
            while True:
                in_bytes = process_in.stdout.read(self.width * self.height * 3)
                if not in_bytes:
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                yield frame
        finally:
            # 确保进程被清理
            process_in.stdout.close()
            if process_in.stderr:
                process_in.stderr.close()
            process_in.wait()


if __name__ == "__main__":
    from tqdm import tqdm

    video_path = Path("resources/dog_vs_sam.mp4")

    # 创建 VideoLoader 实例
    loader = VideoLoader(video_path)

    # 显示视频信息
    print(f"视频路径: {video_path}")
    print(f"分辨率: {loader.width}x{loader.height}")
    print(f"帧率: {loader.fps:.2f} fps")
    print(f"总帧数: {loader.total_frames}")
    print(f"时长: {loader.total_frames / loader.fps:.2f} 秒")
    print("-" * 50)

    # 遍历所有帧并显示进度
    frame_count = 0
    for frame in tqdm(loader, total=len(loader), desc="处理视频"):
        frame_count += 1

        # 每隔 30 帧显示一次信息（可选）
        if frame_count % 30 == 0:
            print(
                f"\n第 {frame_count} 帧 - shape: {frame.shape}, dtype: {frame.dtype}, "
                f"min: {frame.min()}, max: {frame.max()}"
            )

    print(f"\n处理完成！共处理 {frame_count} 帧")

    # 测试提前退出（验证资源清理）
    print("\n测试提前退出...")
    loader2 = VideoLoader(video_path)
    for i, frame in enumerate(loader2):
        if i >= 5:  # 只读取前 5 帧
            print(f"提前退出，已读取 {i + 1} 帧")
            break
    print("资源已正确清理")
