# ISL Translator

Bi-directional Indian Sign Language (ISL) translation prototype supporting **Sign → Text** recognition and **Text → Sign** avatar synthesis across 17 signs.

## Features
- **Live Sign → Text Recognition:** Real-time camera and landmark recognition with rest-pose recalibration, confidence threshold control, top-3 predictions, and sentence building.
- **Personal Profile Calibration:** Optional calibrated profile (`profile.npz`) toggle alongside base model inference.
- **Text → Sign Avatar:** Type custom text/sentences or use quick phrase chips to generate sign output.
- **Model Benchmarks:** Comparison across Baseline, GRU, and Transformer architectures.

## Project Structure
```text
isl-translator/
├── models/              # Saved model weights / calibration profile (when present)
├── sign_to_text/        # Landmark extraction, features, training, adaptation & live recognition
├── templates/           # Web studio UI (index.html)
├── server.py            # Backend / streaming server
├── web_app.py           # Web / app entry point
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependencies (if applicable)
└── README.md
```

## Setup & Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pritamk001/ISL-Translator.git](https://github.com/pritamk001/ISL-Translator.git)
   cd ISL-Translator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   Check `server.py` or `web_app.py` depending on how you start the service, e.g.:
   ```bash
   python server.py
   ```
   *(Or run `web_app.py` / Streamlit command as configured in your setup.)*

## Model Benchmarks (122 Test Samples, 17 Signs)
| Model | Params | Accuracy | Top-3 | Latency | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 76,177 | 45.9% | 85.2% | 0.39 ms | 1 s |
| GRU | 151,288 | 73.8% | 95.1% | 6.64 ms | 61 s |
| **Transformer** | **83,544** | **77.0%** | **95.9%** | **1.95 ms** | **54 s** |

- **Personal Calibration (9 signs, 44 samples):** `52.0% → 87.0%` at `k = 3` shots.