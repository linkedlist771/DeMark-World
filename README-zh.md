# 🌍 DeMark-World

[English](README.md) | 中文

**AI视频水印移除的通用框架**

**DeMark-World** 是 [SoraWatermarkCleaner](https://github.com/linkedlist771/SoraWatermarkCleaner) 的进化版本。它的前身专注于 Sora，而 DeMark-World 是一个"开箱即用"的通用框架，旨在以高保真度和时间一致性检测并移除*任何* AI 生成视频（Sora、Runway、Pika、Kling 等）中的水印。

> 提示词：一只猫在森林中奔跑。

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

## ✨ 特性

DeMark-World 超越了 [SoraWatermarkCleaner](https://github.com/linkedlist771/SoraWatermarkCleaner) 中仅针对 Sora v2 的功能，它专门设计用于处理最新一代视频模型的水印，包括 **Google Gemini/Veo**、**Runway Gen-3/Gen-4**、**Pika**、**Kling** 和 **Luma Dream Machine**。

## 🛠️ 安装

**前置要求**：您必须安装 [FFmpeg](https://ffmpeg.org/) 并将其添加到系统 PATH 中。

我们使用 **[uv](https://github.com/astral-sh/uv)** 进行项目管理。它比 pip/poetry 更快、更可靠。

1. **克隆仓库**

   ```bash
   git clone https://github.com/linkedlist771/DeMark-World.git
   cd DeMark-World
安装环境

BASH
复制
# 这将创建虚拟环境并安装所有依赖项
uv sync
激活环境

BASH
复制
# Linux/MacOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
模型说明：检测器权重和修复模型将在首次运行时自动下载到缓存目录。
🚀 快速开始

1. 交互式 Web 界面（推荐）

使用 DeMark-World 最简单的方式是通过现代化的 Streamlit 界面。

BASH
复制
streamlit run app.py
然后在浏览器中打开 http://localhost:8501。



2. Python API

您可以轻松地将 DeMark-World 集成到自己的流程中。

PYTHON
复制
from pathlib import Path
from src.demark_world.core import DeMarkWorld
from src.demark_world.schemas import CleanerType

if __name__ == "__main__":
    input_video = Path("resources/Veo3_Cat_Running_In_Forest_Video.mp4")
    output_video: Path = Path("outputs/cleaned.mp4")

    # 选项 1: LaMa（快速）
    demarker = DeMarkWorld(cleaner_type=CleanerType.LAMA)
    
    # 选项 2: E2FGVI_HQ（高质量 + 时间一致性）
    # demarker = DeMarkWorld(cleaner_type=CleanerType.E2FGVI_HQ)
    
    demarker.run(input_video, output_video)
🧠 工作原理

DeMark-World 采用两阶段流程运行，其工作方式与 SoraWatermarkCleaner 相同。

📜 许可证

根据 Apache 2.0 许可证分发。更多信息请参见 LICENSE。

🖊️ 引用

如果您在研究或工作中发现此项目有帮助，请引用：

BIBTEX
复制
@misc{DeMark-World2025,
  author = {linkedlist771},
  title = {DeMark-World},
  year = {2025},
  url = {https://github.com/linkedlist771/DeMark-World}
}
🙏 致谢

感谢提供的 SOTA 修复模型的出色实现。
感谢 YOLO 目标检测框架。
💝 如果您觉得这个项目有帮助，请考虑给仓库点个星！
