## **Human Pose Estimation System Using MediaPipe and OpenCV**  

## **📌 Overview**  
This project implements **Human Pose Estimation** using **MediaPipe Pose**, **OpenCV**, and **Streamlit**. It detects **17 keypoints** on the human body from images and videos, creating a skeleton overlay to track movement. The system can process real-time webcam feeds and pre-recorded videos efficiently using multi-threading and frame skipping techniques.  

## **🚀 Features**  
✅ Pose estimation on **images and videos**  
✅ **Real-time webcam support**  
✅ **Frame skipping** for optimized performance  
✅ **Confidence score visualization**  
✅ **Streamlit UI** for an interactive web-based interface  
✅ Uses **COCO topology** (17 keypoints)  

## **🛠️ System Setup**  
### **1. Install Dependencies**  
Ensure you have **Python 3.10** installed. Then, install the required libraries:  
```bash
pip install opencv-python mediapipe numpy matplotlib streamlit
```

### **2. Run the Streamlit UI**  
Launch the interactive web interface where users can upload images/videos:  
```bash
streamlit run estimation_app.py
```

## **🖼️ Pose Estimation on Images and Videos**
- **Images:** The system processes a single image, extracts keypoints, and overlays them.  
- **Videos:** Uses OpenCV to process frames, detect poses in each frame, and display the result dynamically.

## **📝 How It Works?**
### **1. Preprocessing**
- Resizes input images/videos to a suitable resolution.
- Normalizes pixel values for consistent model inference.
- Implements frame skipping to reduce computational load in real-time applications.
  
### **2. Feature Extraction**
- Detects 17 keypoints (e.g., shoulders, elbows, knees) using MediaPipe Pose.
- Outputs (x, y, confidence) coordinates for each detected keypoint.

### **3. Output Display**
- Draws skeleton overlays on detected keypoints.
- Visualizes confidence scores to assess detection accuracy.
- Displays results on the command line or via Streamlit UI.

## **📊 Results & Observations**
The system accurately detects and tracks human body poses in both images and videos. The confidence score helps assess detection quality, while real-time frame skipping improves performance without significant accuracy loss. The results can be used for:

- Fitness tracking
- Rehabilitation monitoring
- Sports analytics

## **📌 COCO Topology**
The model follows the COCO (Common Objects in Context) topology, detecting the following 17 keypoints:
- 👤 Head & Face → Nose, Eyes, Ears
- 🦾 Upper Body → Shoulders, Elbows, Wrists
- 🦵 Lower Body → Hips, Knees, Ankles

## **📌 Output Display, Testing, and Optimization**
- The pose-estimated image highlights detected keypoints with markers, forming a skeletal representation.
- For videos, each frame is processed to detect and overlay keypoints in real-time, ensuring smooth visualization.
- Testing was performed on multiple images and videos, confirming that keypoints are accurately mapped even in varying lighting conditions.
- Optimization techniques, including frame skipping and multi-threading, improve real-time performance.

## **🛠️ Future Improvements**
- 3D Pose Estimation for better motion analysis
- Custom keypoint tracking for specific use cases
- Integration with ML models for pose classification

## **📜 License**
This project is open-source under the MIT License.

```bash
Now you can just copy and paste it into your README.md file! 🚀
```


