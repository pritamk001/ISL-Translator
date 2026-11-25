# 🤟 ISL Translator — Bi-Directional Sign Language Studio

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> Bi-directional Indian Sign Language (ISL) translation prototype — bridging **Sign → Text** recognition and **Text → Sign** avatar synthesis.

---

## 💡 Problem Statement
Communication gaps between sign language users and non-signing individuals create everyday barriers across education, healthcare, and workplaces. **ISL Translator** provides a bi-directional web studio supporting real-time camera/video **Sign → Text** recognition alongside **Text → Sign** avatar generation across 17 signs.

## ✨ Key Features

- 🎥 **Live Sign → Text Recognition** — Real-time camera and landmark recognition with rest-pose recalibration (`r`), mirror toggle, backspace/clear, and speech output (`s`)
- 🎯 **Confidence & Personal Calibration** — Adjustable minimum confidence threshold (`20%–90%`) and optional personal profile toggle (`models/profile.npz`)
- 🧠 **Top-3 Predictions & Sentence Builder** — Live movement indicator, sentence buffer, and top-3 sign probability breakdown
- 🧍 **Text → Sign Avatar** — Type custom text/sentences or click quick phrase chips (`Hello`, `Thank you`, `How are you`, `Good morning`, `Good night`)
- 📈 **Model Benchmarks** — Comparative evaluation across Baseline, GRU, and Transformer models

## 🛠️ Tech Stack

- **Transformer & GRU** — Sequence modeling for sign recognition (Transformer: **77.0%** test accuracy, **95.9%** Top-3, **1.95 ms** latency)
- **Python & NumPy** — Landmark/feature sequence processing and prototype calibration
- **Streamlit & WebRTC (`streamlit_webrtc`, `av`, `imageio`)** — Interactive web studio and live media streaming
- **HTML / CSS / JavaScript (`templates/index.html`)** — Studio UI with Live Sign → Text, Text → Sign Avatar, and Benchmarks tabs

## 📊 Model Performance

Test set evaluation (**122 samples, 17 signs**):

- **Baseline** — Accuracy **45.9%** | Top-3 **85.2%** | Params **76,177** | Latency **0.39 ms** | Train Time **1 s**
- **GRU** — Accuracy **73.8%** | Top-3 **95.1%** | Params **151,288** | Latency **6.64 ms** | Train Time **61 s**
- **Transformer ✅** — Accuracy **77.0%** | Top-3 **95.9%** | Params **83,544** | Latency **1.95 ms** | Train Time **54 s**

- **Personal Calibration (9 signs, 44 samples):** Boosts accuracy from **52.0% → 87.0%** at `k = 3` shots.

## 📸 Screenshots

### Live Sign → Text &  Text → Sign Avatar
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/8a406573-385b-4052-a407-ee0b15060fb9" />

<!-- <img width="1100" alt="Live Sign to Text" src="PASTE_IMAGE_URL_HERE" /> -->

### System Architecture & Workflow
![Uploading image.png…]()

<!-- <img width="1100" alt="System Architecture" src="PASTE_IMAGE_URL_HERE" /> -->

### Model Benchmarks
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/e18de3f8-2ed4-40d1-af9e-5454cfe61fe7" />

<!-- <img width="1100" alt="Model Benchmarks" src="PASTE_IMAGE_URL_HERE" /> -->

---

## ⚙️ Run Locally

```bash
git clone [https://github.com/pritamk001/ISL-Translator.git](https://github.com/pritamk001/ISL-Translator.git)
cd ISL-Translator

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
python server.py
