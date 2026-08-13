# 🧑‍💻 Real-Time Face Detection & Recognition using OpenCV

A real-time face detection and dataset generation system built with OpenCV's Haar Cascade algorithm — extended into a full **face recognition pipeline** using LBPH (Local Binary Pattern Histogram). Captures live webcam input, detects facial regions, builds a per-user dataset, trains a recognizer, and identifies faces live with confidence-based authentication logic.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 🚀 Overview

This started as a basic face-detection + dataset-collection tool and was extended into a complete pipeline:

- 🟢 Real-time face detection using Haar Cascade
- 🟢 Automatic, structured per-user dataset generation
- 🟡 Model training using the LBPH Face Recognizer
- 🔵 Live face recognition from webcam feed
- 🔒 Confidence-based authentication logic
- ❓ Unknown-user detection

It simulates a basic login mechanism — detecting facial presence and giving instant visual feedback — and lays the groundwork for a full AI-based face-authentication system.

---

## 🟢 Phase 1 — Face Detection & Dataset Collection

The application captures live webcam frames and detects faces using Haar Cascade.

**Workflow:**
1. Capture video from webcam
2. Convert frames to grayscale
3. Detect faces using `detectMultiScale()`
4. Crop the detected face region
5. Save face images into user-specific folders inside `dataset/`

Each user gets their own directory — this dataset is used later to train the recognition model.

---

## 🟡 Phase 2 — Model Training (LBPH Algorithm)

`train_model.py`:
- Reads all images from the dataset
- Assigns a numeric label to each person
- Converts images to grayscale
- Trains an LBPH face recognizer
- Saves the trained model as `trained_model.yml`

**Why LBPH?**
- Extracts local texture features from face images
- Converts pixel-intensity patterns into histograms
- Compares histogram similarity during recognition
- Lightweight and efficient for small-to-medium datasets

---

## 🔵 Phase 3 — Real-Time Face Recognition

`recognize_face.py`:
1. Loads the trained model (`trained_model.yml`)
2. Detects faces from the live webcam feed
3. Predicts identity with a confidence score, flagging unrecognized faces as unknown

---

## 🛠️ Tech Stack

- Python
- OpenCV (Haar Cascade + LBPH Face Recognizer)
- NumPy

---

## 📂 Project Structure

```
real-time-face-detection-dataset-opencv/
│
├── main.py               # Phase 1 — face detection & dataset collection
├── train_model.py         # Phase 2 — LBPH model training
├── recognize_face.py       # Phase 3 — real-time face recognition
│
├── dataset/                # Auto-generated, per-user face image folders
├── trained_model.yml       # Saved trained LBPH model (generated after training)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Usage

**1️⃣ Clone the repository**
```
git clone https://github.com/GouravK1107/real-time-face-detection-dataset-opencv.git
cd real-time-face-detection-dataset-opencv
```

**2️⃣ Create a virtual environment**
```
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux / macOS
```

**3️⃣ Install dependencies**
```
pip install opencv-contrib-python numpy
```
> Note: `opencv-contrib-python` (not plain `opencv-python`) is required — the LBPH recognizer lives in the `cv2.face` module, which only ships with the contrib build.

**4️⃣ Collect a dataset** (run once per new user)
```
python main.py
```

**5️⃣ Train the recognition model**
```
python train_model.py
```

**6️⃣ Run real-time recognition**
```
python recognize_face.py
```

---

## 🧠 Technical Concepts Covered

- Image as a NumPy matrix
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

## 💡 Future Improvements

- 🧬 Swap Haar Cascade for a DNN-based detector (higher accuracy in varied lighting)
- 🔐 Turn confidence threshold into a proper liveness/anti-spoofing check
- 📊 Add a simple dashboard showing recognition logs
- 🌐 Wrap into a Flask/Streamlit app for a proper login demo
- 👥 Multi-face recognition in a single frame

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
Fork → create a branch → commit → push → open a pull request.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 👨‍💻 Author

**Gourav R**
Backend Developer | Applied AI Developer — exploring Computer Vision & AI systems

GitHub: https://github.com/GouravK1107
Portfolio: https://gouravk1107.github.io/my-portfolio/

---

Made with ❤️ and a lot of grayscale frames.
