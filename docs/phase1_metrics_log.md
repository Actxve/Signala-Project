Metric                          Value
---------------------------------------------------------
Dataset             | 500 samples, 5 signs, 100 each
Model               | 3-layer Dense NN (64-->32-->5)
Test Accuracy       | 100.0%
Test Loss           | 0.0305
Model Format        | .keras
Hardware            | 12th Gen Intel CORE I7-12700H, Intel Iris Xe Graphics, 16 GB RAM
Training Epochs     | 50
Batch Size          | 32
Framework           | TensorFlow 2.21.1, Python 3.11


## Hand Tracking Baseline (PC, pre-classification)

Metric                          Value 
---------------------------------------------------------
Date                | 25 April 2026 
FPS                 | 21.1 – 21.8 (stable)
Known issues        | Landmarks drop at high hand speed; fist poses poorly detected

## Next Targets (Raspberry Pi Zero 2 W)

Metric                          Target
---------------------------------------------------------
Inference latency   | < 200ms
FPS                 | ≥ 15
Model size          | < 5MB
Model format        | TFLite (post-conversion)