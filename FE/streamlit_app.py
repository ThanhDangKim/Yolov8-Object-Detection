import streamlit as st
import requests
import base64
import tempfile
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
import av  

# ==== BACKEND URL ====
NGROK_BACKEND_URL = 'https://1601ccc832df.ngrok-free.app/'

# ==== PAGE CONFIG ====
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== MAIN TITLE ====
st.title("Object Detection And Tracking using YOLOv8 via Colab API")

# ==== LEFT SIDEBAR ====
st.sidebar.header("ML Model Config")
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
tracker = st.sidebar.selectbox("Choose Tracker (optional)", ["", "bytetrack", "botsort"])

# ==== RIGHT SIDEBAR (SOURCE SELECTION) ====
st.sidebar.header("Image/Video Config")
if "previous_option" not in st.session_state:
    st.session_state.previous_option = None
option = st.sidebar.radio("Select Source", ["Image", "Video", "Webcam", "YouTube"])

if option != st.session_state.previous_option:
    for key in ["media_type", "img_result", "video_result_path", "video_result_bytes", "webcam_frames", "youtube_frames"]:
        st.session_state.pop(key, None)
    st.session_state.previous_option = option
    st.rerun()

# ==== MAIN CONTENT ====

# --- IMAGE ---
if option == "Image":
    source_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    with col1:
        if source_img is not None:
            uploaded_image = Image.open(source_img)
            img_width, img_height = uploaded_image.size
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.sidebar.button("Detect Objects"):
            if source_img is not None:
                img_byte_arr = io.BytesIO()
                uploaded_image.save(img_byte_arr, format="JPEG") 
                img_byte_arr.seek(0) 

                res = requests.post(
                    f"{NGROK_BACKEND_URL}/detect/image",
                    files={"image": ("image.jpg", img_byte_arr, "image/jpeg")},
                    data={"conf": confidence, "tracker": tracker or ""}
                )
                if res.status_code == 200:
                    result_b64 = res.json().get("result", "")
                    img_bytes = base64.b64decode(result_b64)
                    img_result = Image.open(io.BytesIO(img_bytes))
                    img_resized = img_result.resize((img_width, img_height))
                    st.session_state.img_result = (img_bytes, img_resized)
                    st.session_state.media_type = "image"
                else:
                    st.error("❌ Detection failed.")
            else:
                st.warning("Please upload an image before detecting.")
    
        if st.session_state.get("media_type") == "image":
            img_bytes, img_result = st.session_state.img_result
            st.image(img_result, caption="Detected Image", use_column_width=True)
            st.download_button(
                label="📥 Download Detected Image",
                data=img_bytes,
                file_name="detected_image.jpg",
                mime="image/jpeg"
            )

# --- VIDEO ---
elif option == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"]) 

    if st.sidebar.button("Detect Video") and uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        with open(tfile.name, "rb") as f:
            res = requests.post(
                f"{NGROK_BACKEND_URL}/detect/video",
                files={"video": ("video.mp4", f, "video/mp4")},
                data={"conf": confidence, "tracker": tracker or ""}
            )

        if res.status_code == 200:
            st.success("✅ Detection completed.")
            # Lưu file kết quả vào tạm để hiển thị
            result_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(result_video_path, "wb") as f:
                f.write(res.content)

            with open(result_video_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.session_state.video_result_path = result_video_path
            st.session_state.video_result_bytes = video_bytes
            st.session_state.media_type = "video"
        else:
            st.error("❌ Video processing failed.")

    elif st.session_state.get("media_type") is None:
        st.warning("Please upload a video before detecting.")
        
    if st.session_state.get("media_type") == "video":
        st.video(st.session_state.video_result_bytes)
        st.download_button(
            label="📥 Download Processed Video",
            data=st.session_state.video_result_bytes,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )

# --- WEBCAM ---
elif option == "Webcam":
    st.subheader("📷 Live Webcam Detection (Real-time)")

    # Khởi tạo biến lưu ảnh chụp trong session_state
    if "captured_frame" not in st.session_state:
        st.session_state.captured_frame = None

    # Video processor: KHÔNG gọi st.* trong đây
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.conf = confidence
            self.tracker = tracker or ""
            self.last_sent_time = 0.0
            self.last_frame = None  # lưu frame detect gần nhất

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Giảm số frame gửi (1 frame / 0.3s)
            now = time.time()
            if now - self.last_sent_time < 0.3:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            self.last_sent_time = now

            try:
                _, img_encoded = cv2.imencode('.jpg', img)
                files = {"frame": img_encoded.tobytes()}
                data = {"conf": float(self.conf), "tracker": self.tracker}

                res = requests.post(f"{NGROK_BACKEND_URL}/detect/frame",
                                    files=files, data=data, timeout=5)
                if res.status_code == 200:
                    nparr = np.frombuffer(res.content, np.uint8)
                    processed = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.last_frame = processed
                    return av.VideoFrame.from_ndarray(processed, format="bgr24")
                else:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            except Exception as e:
                print("Error sending frame to backend:", e)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Khởi tạo webcam
    webrtc_ctx = webrtc_streamer(
        key="webcam-detect",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Nút capture ở sidebar
    if st.sidebar.button("📸 Capture latest processed frame"):
        if webrtc_ctx and webrtc_ctx.video_processor:
            last = getattr(webrtc_ctx.video_processor, "last_frame", None)
            if last is not None:
                st.session_state.captured_frame = last
            else:
                st.sidebar.warning("No processed frame yet. Wait a moment.")
        else:
            st.sidebar.warning("Webcam not started or processor not ready yet.")
    
    st.sidebar.markdown(
        "<small>The captured image will appear below the webcam video.</small>",
        unsafe_allow_html=True
    )
    
    # Hiển thị ảnh đã chụp + nút download bên dưới webcam
    if st.session_state.captured_frame is not None:
        st.image(st.session_state.captured_frame, channels="BGR", use_column_width=True)
        _, buffer = cv2.imencode(".jpg", st.session_state.captured_frame)
        st.download_button(
            label="📥 Download Last Captured Frame",
            data=buffer.tobytes(),
            file_name="captured_frame.jpg",
            mime="image/jpeg"
        )


# --- YOUTUBE ---
elif option == "YouTube":
    url = st.sidebar.text_input("Paste YouTube URL:")

    if st.sidebar.button("Detect from YouTube"):
        if not url.strip():
            st.error("❌ Please enter a YouTube URL before detecting.")
        else:
            try:
                res = requests.post(
                    f"{NGROK_BACKEND_URL}/detect/youtube",
                    json={"url": url, "conf": confidence, "tracker": tracker or ""}
                )

                if res.status_code == 200:
                    st.success("✅ YouTube processed successfully.")

                    result_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    with open(result_video_path, "wb") as f:
                        f.write(res.content)

                    with open(result_video_path, "rb") as vid_file:
                        video_bytes = vid_file.read()

                    st.session_state.youtube_result_path = result_video_path
                    st.session_state.youtube_result_bytes = video_bytes
                    st.session_state.media_type = "youtube"
                else:
                    err = res.json().get("error", res.text)
                    st.error(err)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.get("media_type") == "youtube":
        st.video(st.session_state.youtube_result_bytes)
        st.download_button(
            label="📥 Download Processed Video",
            data=st.session_state.youtube_result_bytes,
            file_name="youtube_processed.mp4",
            mime="video/mp4"
        )
