## Real-Time Face Detection and Dataset Collection System using OpenCV <br>
This project implements a real-time face detection and structured dataset generation system using OpenCV’s Haar Cascade algorithm. The application captures live webcam input, detects facial regions, extracts grayscale face data, and stores multiple samples per user in an organized directory structure. The system simulates a basic login mechanism by detecting facial presence and providing instant visual feedback. The project highlights practical implementation of classical computer vision techniques including feature-based detection, frame-by-frame video processing, image matrix manipulation, and automated dataset creation. It lays the groundwork for extending the system into a full face recognition and AI-based authentication pipeline.

---

## 🚀 Extended Features (Face Recognition Added)

This project has been extended beyond basic face detection to implement a complete face recognition pipeline using OpenCV’s LBPH (Local Binary Pattern Histogram) algorithm.

The system now supports:

- Real-time face detection using Haar Cascade
- Automatic dataset generation per user
- Structured dataset folder creation
- Model training using LBPH Face Recognizer
- Live face recognition from webcam
- Confidence-based authentication logic
- Unknown user detection

## 🟢 Phase 1 — Face Detection & Dataset Collection

The application captures live webcam frames and detects faces using Haar Cascade.

### Workflow:

1. Capture video from webcam  
2. Convert frames to grayscale  
3. Detect faces using `detectMultiScale()`  
4. Crop detected face region  
5. Save face images into user-specific folders  

Each user has a separate directory inside `dataset/`.
This dataset is later used for training the recognition model.

---

## 🟡 Phase 2 — Model Training (LBPH Algorithm)

The training script:

- Reads all images from the dataset
- Assigns numeric labels to each person
- Converts images to grayscale
- Trains an LBPH face recognizer
- Saves the trained model as `trained_model.yml`

### Why LBPH?

LBPH (Local Binary Pattern Histogram):

- Extracts local texture features from face images
- Converts pixel intensity patterns into histograms
- Compares histogram similarity during recognition
- Lightweight and efficient for small to medium datasets

---

## 🔵 Phase 3 — Real-Time Face Recognition

The recognition system:

1. Loads the trained model
2. Detects face from live webcam feed
3. Predicts identity


---

## 🧠 Technical Concepts Covered

- Image as NumPy matrix
- Grayscale image processing
- Haar Cascade face detection
- Region extraction & cropping
- Dataset generation
- Label encoding
- Feature extraction (LBPH)
- Model training & persistence
- Confidence threshold tuning
- Real-time prediction pipeline

---
