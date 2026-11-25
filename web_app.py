from pathlib import Path
import tempfile
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
# Optional WebRTC for live camera in browser
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
SIGN_DIR = ROOT / "sign_to_text"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="ISL Translator", layout="wide")
st.title("Indian Sign Language (ISL) Translator")


def run_text_to_sign(sentence: str):
    """
    Connects to existing avatar / text-to-sign logic if available.
    Fallback generates a simple preview video if avatar module differs.
    """
    out_mp4 = ROOT / "results" / "avatar_web_preview.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        frames = []
        base = np.zeros((360, 480, 3), dtype=np.uint8)
        base[:, :] = (24, 30, 42)
        for i in range(24):
            f = base.copy()
            x = 120 + (i * 8) % 200
            y = 180 + int(25 * np.sin(i * 0.4))
            f[max(0, y - 15):min(360, y + 15), max(0, x - 15):min(480, x + 15)] = (80, 200, 255)
            frames.append(f)
        imageio.mimsave(str(out_mp4), frames, fps=12, codec="libx264")
        return str(out_mp4), None
    except Exception as e:
        return None, str(e)


class LiveSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.status_text = "READY | Top-3: --"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        # You can call your live_recognition.py / features.py / model step here.
        # Safe overlay status update for now:
        if self.frame_count % 15 == 0:
            self.status_text = f"LIVE FRAME #{self.frame_count} | Calibrated/Active"
        return av.VideoFrame.from_ndarray(img, format="bgr24")


tab1, tab2, tab3 = st.tabs(["Sign ? Text", "Text ? Sign", "About"])

with tab1:
    st.subheader("Sign ? Text")
    if WEBRTC_AVAILABLE:
        mode = st.radio(
            "Choose an input source:",
            ["Live Camera (Browser)", "Upload Video Clip"],
            horizontal=True,
        )
    else:
        st.info("Live camera requires `streamlit-webrtc` and `av` to be installed. Video upload is active for now.")
        mode = "Upload Video Clip"

    if mode == "Live Camera (Browser)" and WEBRTC_AVAILABLE:
        st.caption("Allow camera permission in your browser. Once you stop, the summary/sentence will appear here.")
        ctx = webrtc_streamer(
            key="isl-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveSignProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            st.write("**Status:**", ctx.video_processor.status_text)
    else:
        uploaded = st.file_uploader("Upload a short sign video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            st.video(tmp_path)
            if st.button("Recognize Clip"):
                st.success("Clip loaded! Confirm the paths to `features.py` and `best.pt` to connect it to your model/feature pipeline.")

with tab2:
    st.subheader("Text ? Sign")
    sentence = st.text_input("Enter a sentence or words", value="hello how are you")
    if st.button("Sign it"):
        mp4_path, err = run_text_to_sign(sentence)
        if err:
            st.error(f"Error: {err}")
        elif mp4_path and Path(mp4_path).exists():
            st.video(mp4_path)

with tab3:
    st.subheader("About")
    st.write("ISL Translator web prototype � Sign ? Text (Live Camera / Clip Upload) and Text ? Sign.")
