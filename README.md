# Signala

Real-time ASL gesture recognition built to run offline on a Raspberry Pi Zero 2 W.

Framed as an embedded systems engineering project — the focus is on model optimization,
latency, and hardware constraints, not just accuracy.

---

## Phase 1 Results (PC Baseline)

| Metric | Value |
|---------------------------------------------------------|
| Signs recognized | A, B, C, L, Y |
| Dataset | 500 samples, 100 per sign |
| Model | Dense NN — Input(63) → Dense(64) → Dense(32) → Dense(5) |
| Test accuracy | 100.00% |
| Test loss | 0.0305 |
| Framework | TensorFlow 2.21.0, Python 3.11 |

---

## Stack

- Python 3.11
- MediaPipe Tasks API — hand landmark detection
- TensorFlow / TFLite — training and embedded deployment
- OpenCV — camera input and frame rendering
- Target hardware: Raspberry Pi Zero 2 W

---

## Project Structure

Signala_Proj/
├── data/                   # Collected landmark datasets
├── docs/                   # Metrics logs and notes
├── nn_data_collection.py   # Webcam-based landmark data collector
├── model_training.py       # Neural network training script
├── hand_tracking_test.py   # Live hand tracking with FPS display
└── requirements.txt

---

## Setup

```bash
pip install -r requirements.txt
```

The MediaPipe hand landmark model (`hand_landmarker.task`) is downloaded
automatically on first run of `hand_tracking_test.py`.

---

## Roadmap

- [x] Hand landmark detection (MediaPipe Tasks API)
- [x] Data collection pipeline
- [x] Neural network training
- [ ] TFLite model conversion
- [ ] Raspberry Pi Zero 2 W deployment
- [ ] Latency and memory optimization
- [ ] Expanded sign vocabulary