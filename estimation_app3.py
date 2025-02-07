import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import threading
import queue

# Constants
DEMO_IMAGE = 'stand.jpg'
DEMO_VIDEO = 'demo.mp4'

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Model Configurations
width = 368
height = 368
inWidth = width
inHeight = height
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Page Title and Sidebar
st.set_page_config(page_title="Human Pose Estimation", page_icon="🕺", layout="wide")

# Add custom CSS for background and design
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #56CCF2, #2F80ED); 
            font-family: 'Arial', sans-serif;
        }
        .main {
            color: white;
            padding: 30px;
            text-align: center;
        }
        .sidebar {
            background-color: #2F80ED;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #56CCF2;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #2F80ED;
        }
        .expander .stExpanderHeader {
            font-size: 20px;
            color: #56CCF2;
            background-color: #2F80ED;
            padding: 10px;
        }
        .expander .stExpanderContent {
            background-color: #f4f4f4;
            padding: 15px;
        }
        h1 {
            font-size: 50px;
            font-weight: bold;
        }
        h2 {
            font-size: 30px;
            font-weight: normal;
        }
        p {
            font-size: 18px;
        }
        .stImage {
            border-radius: 10px;
            border: 5px solid #56CCF2;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the options below to interact with the app:")

st.sidebar.subheader("Upload an Image or Video")
file_buffer = st.sidebar.file_uploader(
    "Upload an image or video (up to 1GB supported)", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)
thres = st.sidebar.slider("Detection Threshold", min_value=0.05, value=0.15, max_value=0.5, step=0.05)
frame_skip = st.sidebar.slider("Process every Nth frame (1 = all frames)", min_value=1, max_value=10, value=3)

# Pose Detection Function
@st.cache
def poseDetector(frame):
    try:
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        # Preprocess the image for the model
        input_blob = cv2.dnn.blobFromImage(
            frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False
        )
        net.setInput(input_blob)

        # Forward pass
        out = net.forward()
        out = out[:, :19, :, :]  # Only keep the first 19 parts

        # Initialize list of points
        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)

            x = int((frameWidth * point[0]) / out.shape[3])
            y = int((frameHeight * point[1]) / out.shape[2])

            # Append keypoints with confidence > threshold
            points.append((x, y) if conf > thres else None)

        # Draw skeleton
        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.circle(frame, points[idFrom], 5, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, points[idTo], 5, (0, 0, 255), cv2.FILLED)

        return frame

    except Exception as e:
        st.error(f"Error in pose detection: {e}")
        return frame

# Video Processing with Optimization
def process_video(video_path, output_queue, frame_skip, thres):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_skip == 0:
            frame = cv2.resize(frame, (640, 480))  # Downscale for faster processing
            output = poseDetector(frame)
            output_queue.put(output)
    cap.release()
    output_queue.put(None)  # Signal the end of the video

# Header Section
st.markdown("""
    <div class="main">
        <h1>🕺 Human Pose Estimation</h1>
        <p>Upload an image or video to estimate and visualize human pose.</p>
    </div>
""", unsafe_allow_html=True)

# Processing Image or Video
if file_buffer is not None:
    file_type = file_buffer.name.split(".")[-1].lower()

    if file_type in ["jpg", "jpeg", "png"]:  # Image Processing
        image = np.array(Image.open(file_buffer))
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Pose Estimation for Image
        output = poseDetector(image)

        # Results Section for Image
        with st.expander("View Pose Estimation Results"):
            st.subheader("Pose Estimation Output")
            st.image(output, caption="Pose Estimated", use_container_width=True)

    elif file_type in ["mp4", "avi", "mov"]:  # Video Processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_buffer.read())

        st.subheader("Uploaded Video")
        output_queue = queue.Queue()
        threading.Thread(
            target=process_video,
            args=(tfile.name, output_queue, frame_skip, thres),
            daemon=True
        ).start()

        stframe = st.empty()
        while True:
            frame = output_queue.get()
            if frame is None:
                break
            if frame is not None:  # Ensure the frame is valid
                stframe.image(frame, channels="BGR", caption="Pose Estimated from Video", use_container_width=True)

        # Results Section for Video
        with st.expander("View Pose Estimation Results"):
            st.subheader("Pose Estimation Output")
            if frame is not None:  # Check if frame is not None before displaying
                st.image(frame, channels="BGR", caption="Pose Estimated from Video", use_container_width=True)
            else:
                st.warning("No valid frame to display.")
else:
    st.subheader("Demo")
    st.image(DEMO_IMAGE, caption="Demo Image", use_container_width=True)
