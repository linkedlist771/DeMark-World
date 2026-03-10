# 🌍 DeMark-World

**A Universal Framework for AI Video Watermark Removal**

**DeMark-World** is the evolution of [SoraWatermarkCleaner](https://github.com/linkedlist771/SoraWatermarkCleaner). While its predecessor focused on Sora, DeMark-World is a "ready-to-roll" universal framework designed to detect and remove unwanted watermarks from *any* AI-generated video (Sora, Runway, Pika, Kling, etc.) with high fidelity and temporal consistency.
> Prompt: a cat running in a forest.





<table>
  <tr>
    <td width="20%">
      <strong>Sora2</strong>
    </td>
    <td width="80%">
      <video src="https://github.com/user-attachments/assets/a92b99fc-a716-4f96-9963-feb85491e84b" 
             width="100%" controls></video>
    </td>
  </tr>
  <tr>
    <td>
      <strong>Google Veo3.1</strong>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/de8b7f45-22ac-4871-b59b-9b5837f25432" 
             width="100%" controls></video>
    </td>
  </tr>
  <tr>
    <td>
      <strong>Ruanway Gen4</strong>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/435d4888-539b-4670-b364-2a1ac7e211c9" 
             width="100%" controls></video>
    </td>
  </tr>
</table>




## ✨ Features

Moving beyond just Sora v2 in [SoraWatermarkCleaner](https://github.com/linkedlist771/SoraWatermarkCleaner), DeMark-World is engineered to handle watermarks from the latest generation of video models, including **Google Gemini/Veo**, **Runway Gen-3/Gen-4**, **Pika**, **Kling**, and **Luma Dream Machine**.

| Feature | Details |
|---|---|
| **Universal detection** | Works with Sora, Runway, Pika, Kling, Veo, and more |
| **2 inpainting backends** | LaMa (fast) · E2FGVI-HQ (temporally consistent) |
| **Batch YOLO detection** | Processes frames in GPU batches (`--batch-size 4`) for ~2× detection throughput |
| **BF16 inference** | Auto-enabled on supported GPUs (Ampere+) via `torch.cuda.is_bf16_supported()` |
| **TorchCompile** | `torch.compile` with artifact caching for persistent speedup across runs |
| **4 interfaces** | Web UI · Python API · CLI · REST Server |
| **Docker support** | GPU-enabled `docker-compose.yaml` included |

## 🛠️ Installation

**Prerequisites**: You must have [FFmpeg](https://ffmpeg.org/) installed and added to your system PATH.

We use **[uv](https://github.com/astral-sh/uv)** for project management. It is significantly faster and more reliable than pip/poetry.

1. **Clone the repository**

   Bash

   ```
   git clone https://github.com/linkedlist771/DeMark-World.git
   cd DeMark-World
   ```

2. **Install environment**

   ```
   # This creates the virtual environment and installs all dependencies
   uv sync
   ```

3. **Activate environment**

   ```
   # Linux/MacOS
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

> **Note on Models**: Detector weights and Inpainting models will be downloaded automatically to the cache directory upon the first run.



## 🚀 Quick Start

### 1. Interactive Web UI (Recommended)

The easiest way to use DeMark-World is via the modern Streamlit interface.

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

![image](assets/webui.png)

### 2. CLI

Process an entire folder of videos from the command line with a rich progress display:

```bash
# Basic usage (LaMa, fast)
demark-world -i /path/to/videos -o /path/to/output

# High quality, time-consistent (E2FGVI-HQ)
demark-world -i /path/to/videos -o /path/to/output --model e2fgvi_hq

# Max performance on CUDA (BF16 + TorchCompile, auto-enabled on supported GPUs)
demark-world -i /path/to/videos -o /path/to/output --model e2fgvi_hq --torch-compile

# Adjust detection batch size
demark-world -i /path/to/videos -o /path/to/output --batch-size 8

# Suppress internal progress bars
demark-world -i /path/to/videos -o /path/to/output --quiet
```

### 3. Python API

You can integrate DeMark-World into your own pipelines easily.

```python
from pathlib import Path
from src.demark_world.core import DeMarkWorld
from src.demark_world.schemas import CleanerType

if __name__ == "__main__":
    input_video = Path("resources/Veo3_Cat_Running_In_Forest_Video.mp4")
    output_video: Path = Path("outputs/cleaned.mp4")

    # Option 1: LaMa (Fast)
    demarker = DeMarkWorld(cleaner_type=CleanerType.LAMA)

    # Option 2: E2FGVI_HQ (High Quality + Time Consistent)
    # BF16 and batch detection are auto-configured based on your hardware
    # demarker = DeMarkWorld(cleaner_type=CleanerType.E2FGVI_HQ)

    demarker.run(input_video, output_video)
```

### 4. REST Server

Start the API server for queue-based async processing:

```bash
python start_server.py --host 0.0.0.0 --port 5344
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit_remove_task` | Submit a video; accepts `?cleaner_type=lama\|e2fgvi_hq` |
| `GET` | `/get_results?remove_task_id=<id>` | Poll task status and progress |
| `GET` | `/get_queue_status` | View queue depth and all task states |
| `GET` | `/download/<task_id>` | Download the processed video |

```bash
# Submit a task (LAMA, default)
curl -X POST http://localhost:5344/submit_remove_task \
  -F "video=@input.mp4"

# Submit with E2FGVI_HQ
curl -X POST "http://localhost:5344/submit_remove_task?cleaner_type=e2fgvi_hq" \
  -F "video=@input.mp4"

# Check queue
curl http://localhost:5344/get_queue_status
```

### 5. Docker

```bash
docker compose up
```

The server will be available at `http://localhost:5344`. GPU passthrough is configured automatically.



## 🧠 How It Works

DeMark-World operates in a two-stage pipeline:

1. **Detection** — A YOLO model scans every frame (in GPU batches) to locate the watermark bounding box. Missed detections are imputed using interval-averaged bounding boxes from neighboring frames.
2. **Inpainting** — The detected region is inpainted using the selected backend:
   - **LaMa**: per-frame, fast, suitable for static watermarks.
   - **E2FGVI-HQ**: processes overlapping temporal segments for smooth, flicker-free results.

### ⚡ Performance

| Optimization | How it's applied |
|---|---|
| **Batch YOLO detection** | Frames are accumulated into batches (`DEFAULT_DETECT_BATCH_SIZE=4`) before GPU inference |
| **BF16 inference** | Auto-detected via `torch.cuda.is_bf16_supported()` — enabled by default on Ampere (RTX 30xx/40xx, A100, H100) and newer |
| **TorchCompile** | `torch.compile(mode="default")` with artifact caching in `~/.cache/torch_compile/e2fgvi_hq/` — first run compiles once, subsequent runs load from cache |



## 📜 License



Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.



## 🖊️ Citation

If you find this project helpful in your research or work, please cite:

```bash
@misc{DeMark-World2025,
  author = {linkedlist771},
  title = {DeMark-World},
  year = {2025},
  url = {https://github.com/linkedlist771/DeMark-World}
}
```

## 🙏 Acknowledgments

- For the incredible implementation of SOTA inpainting models.
- For the YOLO object detection framework.

---

💝 If you find this project helpful, please consider starring the repo!
