import shutil
import tempfile
import time
from pathlib import Path
import streamlit as st

from src.demark_world.core import DeMarkWorld
from src.demark_world.schemas import CleanerType

st.set_page_config(
    page_title="DeMark-World | Universal Watermark Remover",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(120deg, #1E88E5, #00E676);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
        }
        
        .sub-title {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-top: -10px;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* 卡片样式 */
        .card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }

        /* 按钮增强 */
        .stButton>button {
            border-radius: 20px;
            font-weight: 600;
        }
    </style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """初始化 Session State"""
    if "sora_wm" not in st.session_state:
        st.session_state.sora_wm = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None


def sidebar_config():
    """侧边栏配置区域"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/video-trimming.png", width=64)
        st.title("⚙️ Configuration")

        st.markdown("### Model Selection")

        model_type = st.selectbox(
            "Choose Inpainting Engine:",
            options=[CleanerType.LAMA, CleanerType.E2FGVI_HQ],
            format_func=lambda x: {
                CleanerType.LAMA: "🚀 LAMA (Fast & Efficient)",
                CleanerType.E2FGVI_HQ: "💎 E2FGVI-HQ (High Fidelity)",
            }[x],
        )

        # 模型介绍信息
        if model_type == CleanerType.LAMA:
            st.info(
                "**LAMA**: Best for quick previews and simple watermarks. Uses Large Mask Inpainting technology."
            )
        else:
            st.info(
                "**E2FGVI-HQ**: Best for complex backgrounds. Uses flow-based propagation for temporal consistency (GPU Recommended)."
            )

        st.markdown("---")
        st.markdown("### About DeMark-World")
        st.caption(
            "A universal framework designed to detect and remove unwanted watermarks "
            "from AI-generated videos (Sora, Runway, Pika, etc.) with temporal consistency."
        )
        st.markdown("---")
        st.caption("v1.0.0 | Built with Streamlit")

        return model_type


def main():
    init_session_state()

    # --- 头部 Hero 区域 ---
    st.markdown('<p class="main-title">DeMark-World 🌍</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Ready-to-roll Universal Framework for AI Video Watermark Removal</p>',
        unsafe_allow_html=True,
    )

    # --- 侧边栏 & 模型加载逻辑 ---
    selected_model = sidebar_config()

    # 如果模型发生变化或未加载，则重新加载
    if st.session_state.sora_wm is None or st.session_state.get("current_model") != selected_model:
        with st.spinner(f"⚙️ Initializing {selected_model.value.upper()} engine..."):
            try:
                st.session_state.sora_wm = DeMarkWorld(cleaner_type=selected_model)
                st.session_state.current_model = selected_model
                st.toast(f"Engine switched to {selected_model.value.upper()}", icon="✅")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

    # --- 主功能区 (Tabs) ---
    tab_single, tab_batch = st.tabs(["🎬 Single Video Studio", "📦 Batch Processing Center"])

    # === Tab 1: 单视频处理 ===
    with tab_single:
        st.markdown("<br>", unsafe_allow_html=True)

        # 文件上传区
        uploaded_file = st.file_uploader(
            "Drag and drop your video here",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV",
        )

        if uploaded_file:
            # 重置状态逻辑
            if (
                "current_file_name" not in st.session_state
                or st.session_state.current_file_name != uploaded_file.name
            ):
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.processed_video_data = None
                st.session_state.processed_video_name = None

            # 视频展示区
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📥 Input Source")
                st.video(uploaded_file)
                st.caption(
                    f"Filename: {uploaded_file.name} | Size: {uploaded_file.size / (1024 * 1024):.2f} MB"
                )

            with col2:
                st.markdown("### ✨ DeMarked Output")
                if st.session_state.get("processed_video_data"):
                    st.video(st.session_state.processed_video_data)
                    st.download_button(
                        label="⬇️ Download Clean Video",
                        data=st.session_state.processed_video_data,
                        file_name=st.session_state.processed_video_name,
                        mime="video/mp4",
                        use_container_width=True,
                        type="primary",
                    )
                else:
                    st.markdown(
                        """
                        <div style="height: 200px; border: 2px dashed #ccc; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #888;">
                            Waiting for processing...
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 处理按钮栏
            st.markdown("---")
            col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

            with col_btn_2:
                if st.button("🪄 Start Magic Removal", type="primary", use_container_width=True):
                    process_single_video(uploaded_file)

    # === Tab 2: 批量处理 ===
    with tab_batch:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "📂 **Batch Mode**: Upload multiple files or drag an entire folder to process them sequentially."
        )

        uploaded_files = st.file_uploader(
            "Select videos for batch processing",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            key="batch_uploader",
        )

        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} files**")

            with st.expander("📋 Review File List"):
                for f in uploaded_files:
                    st.text(f"- {f.name}")

            if st.button("🚀 Process All Videos", type="primary"):
                process_batch_videos(uploaded_files)

            # 显示批量下载结果
            if "batch_results" in st.session_state and st.session_state.batch_results:
                st.markdown("### ✅ Completed Tasks")
                for res in st.session_state.batch_results:
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.text(f"📄 {res['name']}")
                    with col_b:
                        st.download_button(
                            "⬇️",
                            data=res["data"],
                            file_name=res["name"],
                            mime="video/mp4",
                            key=f"dl_{res['name']}",
                        )


# --- 处理逻辑函数 ---


def process_single_video(uploaded_file):
    """处理单个视频的逻辑"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / uploaded_file.name
        output_path = tmp_path / f"demarked_{uploaded_file.name}"

        # 写入临时文件
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # UI 进度条组件
        progress_bar = st.progress(0)
        status_container = st.empty()

        try:

            def update_progress(p):
                progress_bar.progress(p / 100)
                if p < 30:
                    status_container.info("🔍 Analyzing watermark patterns...")
                elif p < 80:
                    status_container.info("🧹 Inpainting and restoring frames...")
                else:
                    status_container.info("🎵 Re-encoding audio track...")

            # 运行核心模型
            st.session_state.sora_wm.run(input_path, output_path, progress_callback=update_progress)

            # 读取结果
            with open(output_path, "rb") as f:
                video_data = f.read()

            # 更新状态
            st.session_state.processed_video_data = video_data
            st.session_state.processed_video_name = f"demarked_{uploaded_file.name}"

            progress_bar.progress(100)
            status_container.success("✅ Processing Complete!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


def process_batch_videos(uploaded_files):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()
        output_folder.mkdir()

        results = []

        main_progress = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)

        try:
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(
                    f"**Processing file {idx + 1}/{total_files}:** `{uploaded_file.name}`"
                )

                file_path = input_folder / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                output_file_path = output_folder / f"demarked_{uploaded_file.name}"

                def batch_step_progress(p):
                    pass

                st.session_state.sora_wm.run(
                    file_path, output_file_path, progress_callback=batch_step_progress
                )

                with open(output_file_path, "rb") as f:
                    results.append({"name": f"demarked_{uploaded_file.name}", "data": f.read()})

                main_progress.progress((idx + 1) / total_files)

            st.session_state.batch_results = results
            status_text.success(f"🎉 All {total_files} videos processed successfully!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")


if __name__ == "__main__":
    main()
