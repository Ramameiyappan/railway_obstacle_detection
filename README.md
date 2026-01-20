# Railway Obstacle Detection System

An AI-powered railway safety application that detects obstacles on railway tracks using YOLO models and provides visual and audio alerts through a Streamlit web interface.
It uses two model one is segmentation model  and other is object detetcion model

---

## 🌐 Live Demo (Streamlit Deployment)

👉 **Streamlit App Link:**  
🔗 https://railwayobstacledetection.streamlit.app/

---

## 📌 Features

- Railway track detection using YOLO
- Obstacle detection on railway tracks
- Obstacle-on-track overlap logic
- Side-by-side original and annotated image display
- Audio alert generation for detected obstacles
- Clean and user-friendly Streamlit UI

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **YOLO (Ultralytics)**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **gTTS (Text-to-Speech)**

---

## 📂 Project Structure

```
railway_obstacle_detection/
│
├── app.py
├── requirements.txt
├── package.txt
├── README.md
│
├── model/
│   ├── track.pt
│   └── obstacle.pt
│
└── utils/
    ├── detector.py
    └── audio.py
```

---

## ▶️ How to Run Locally

### 1️⃣ Create and activate virtual environment

```bash
python -m venv railway
railway\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit application

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

- Python version used: **3.10**
- `runtime.txt` content:
  ```
  python-3.10
  ```
- YOLO model files are lightweight (~6 MB) and included in the repository.

---

## 🔊 Audio Alerts

- Audio alerts are generated using **gTTS**
- Internet connection is required for first-time audio generation
- Audio is cached for faster playback on repeated detections

---

This project is intended for educational and research purposes.
