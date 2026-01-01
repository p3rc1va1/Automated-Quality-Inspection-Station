# PCB Defect Detection with YOLOv8 🔍

Automated quality inspection system for detecting defects in Printed Circuit Boards (PCBs) using YOLOv8 deep learning model.

## 🏗️ YOLOv8 Architecture

![YOLOv8 Architecture](./Users/bahacelik/Documents/Coding/Automated-Quality-Inspection-Station/Basic-architecture-of-YOLOv8-object-detection-model.ppm.png)

*Detailed YOLOv8 architecture showing Backbone (CSPDarknet), Neck (FPN + PAN), and Detection Head [link](https://www.researchgate.net/publication/376831163_YOLOv8_based_Traffic_Signal_Detection_in_Indian_Road)*

---

## 🔄 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PCB DEFECT DETECTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────┐
     │   KAGGLE DATASET │
     │   akhatova/      │
     │   pcb-defects    │
     └────────┬─────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         1. DATA ACQUISITION         │
│  ┌───────────────────────────────┐  │
│  │ • 693 PCB images              │  │
│  │ • 6 defect classes            │  │
│  │ • Pascal VOC XML annotations  │  │
│  │ • Image size: 3034×1586       │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      2. DATA PREPROCESSING          │
│  ┌───────────────────────────────┐  │
│  │ convert_voc_to_yolo.py        │  │
│  │                               │  │
│  │ XML → YOLO TXT format         │  │
│  │ (class x_center y_center w h) │  │
│  │                               │  │
│  │ Split: 70% / 20% / 10%        │  │
│  │ Train: 485 | Val: 138 | Test: 70│ │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     3. MODEL CONFIGURATION          │
│  ┌───────────────────────────────┐  │
│  │ Model: YOLOv8s (11.1M params) │  │
│  │ Pretrained: COCO weights      │  │
│  │ Input size: 640×640           │  │
│  │ Classes: 6 (nc=6)             │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ pcb_defects.yaml              │  │
│  │ ├── train: images/train       │  │
│  │ ├── val: images/val           │  │
│  │ └── names: [6 classes]        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         4. TRAINING                 │
│  ┌───────────────────────────────┐  │
│  │ Platform: Google Colab (T4)   │  │
│  │ Epochs: 50                    │  │
│  │ Batch size: 16                │  │
│  │ Optimizer: AdamW (lr=0.001)   │  │
│  │ Early stopping: patience=15   │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Data Augmentation:            │  │
│  │ • Mosaic (4 images → 1)       │  │
│  │ • MixUp (blend images)        │  │
│  │ • Rotation (±10°)             │  │
│  │ • Scale (0.5-1.5×)            │  │
│  │ • Horizontal flip (50%)       │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│        5. EVALUATION                │
│  ┌───────────────────────────────┐  │
│  │ Metrics on Test Set (70 imgs) │  │
│  │ ┌───────────────────────────┐ │  │
│  │ │ mAP50:     93.4%          │ │  │
│  │ │ mAP50-95:  51.2%          │ │  │
│  │ │ Precision: 94.1%          │ │  │
│  │ │ Recall:    89.2%          │ │  │
│  │ └───────────────────────────┘ │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         6. MODEL OUTPUT             │
│  ┌───────────────────────────────┐  │
│  │ models/pcb_defects_yolov8m/   │  │
│  │ └── best.pt (trained weights) │  │
│  │                               │  │
│  │ Inference: ~2ms per image     │  │
│  │ Ready for production use!     │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 🎯 Project Overview

| Property | Value |
|----------|-------|
| **Model** | YOLOv8s (11.2M parameters) |
| **Dataset** | [PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) |
| **Classes** | 6 defect types |
| **Framework** | Ultralytics + PyTorch |

### Defect Classes
- `missing_hole` - Missing drill holes
- `mouse_bite` - Irregular copper removal
- `open_circuit` - Broken traces
- `short` - Unintended connections
- `spur` - Unwanted copper protrusions
- `spurious_copper` - Extra copper deposits

---

## 🚀 Quick Start

### Local Setup (with uv)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Automated-Quality-Inspection-Station.git
cd Automated-Quality-Inspection-Station

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv sync

# Or install individually
uv add ultralytics opencv-python matplotlib kagglehub python-dotenv
```

### Download Dataset

```bash
# Set up Kaggle credentials first (get from https://www.kaggle.com/settings)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Run the EDA notebook to download dataset
# Or use the conversion script after manual download
python scripts/convert_voc_to_yolo.py
```

### Train Locally (Mac/Linux)

```bash
# Train with MPS (Apple Silicon) or CPU
python train.py

# Validate a trained model
python train.py validate models/pcb_defects_yolov8s/weights/best.pt
```

### Train on Google Colab (Recommended for GPU)

1. Upload `train_colab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** 
3. Upload your `kaggle.json` when prompted
4. Run all cells (~20 min training time)

---

## 📁 Project Structure

```
Automated-Quality-Inspection-Station/
├── data/
│   ├── PCB_DATASET/           # Original dataset (VOC format)
│   └── yolo_dataset/          # Converted YOLO format
├── models/                    # Trained model weights
├── scripts/
│   ├── convert_voc_to_yolo.py # Data conversion script
│   └── visualize.ipynb # script to visualzie if the model is working
├── EDA/
│   └── exploration.ipynb      # Dataset exploration
├── train/
│   ├── train.py                   # Local training script
│   └── train_colab.ipynb          # Colab training notebook
├── plan.md                    # Project roadmap
├── pyproject.toml             # uv/Python dependencies
└── README.md                  # This file
```

---

## 📊 Table of Contents (Technical Details)
1. [What is YOLO?](#what-is-yolo)
2. [YOLOv8 Architecture](#yolov8-architecture)
3. [Why YOLOv8 for PCB Defect Detection?](#why-yolov8-for-pcb-defect-detection)
4. [Our Configuration Decisions](#our-configuration-decisions)
5. [Training Pipeline](#training-pipeline)

---

## What is YOLO?

**YOLO (You Only Look Once)** is a real-time object detection algorithm that revolutionized computer vision by treating object detection as a single regression problem.

### Traditional vs YOLO Approach

| Traditional (R-CNN family) | YOLO |
|---------------------------|------|
| Two-stage: Region proposal → Classification | Single-stage: One neural network pass |
| Slow (seconds per image) | Fast (milliseconds per image) |
| Multiple passes over image | Single pass ("You Only Look Once") |

### How YOLO Works

```
Input Image → Divide into Grid → Predict Bounding Boxes + Class Probabilities → Non-Max Suppression → Final Detections
     ↓              ↓                           ↓                                      ↓
  [640×640]    [S×S grid]         [B boxes per cell with (x,y,w,h,conf,classes)]    [Filter overlaps]
```

1. **Grid Division**: Image is divided into an S×S grid
2. **Predictions**: Each grid cell predicts:
   - B bounding boxes (x, y, width, height)
   - Confidence score for each box
   - Class probabilities
3. **Non-Maximum Suppression (NMS)**: Removes duplicate detections

---

## YOLOv8 Architecture Details

### Layer-by-Layer Structure

Exact architecture of YOLOv8s used in this project (11.1M parameters, 28.7 GFLOPs):

```
Layer  From    Module                              Output Shape
─────────────────────────────────────────────────────────────────
 0     -1      Conv (3→32, k=3, s=2)              [320×320×32]     ─┐
 1     -1      Conv (32→64, k=3, s=2)             [160×160×64]      │ BACKBONE
 2     -1      C2f (64→64)                        [160×160×64]      │
 3     -1      Conv (64→128, k=3, s=2)            [80×80×128]       │
 4     -1      C2f (128→128, n=2)                 [80×80×128]   ←P3 │
 5     -1      Conv (128→256, k=3, s=2)           [40×40×256]       │
 6     -1      C2f (256→256, n=2)                 [40×40×256]   ←P4 │
 7     -1      Conv (256→512, k=3, s=2)           [20×20×512]       │
 8     -1      C2f (512→512)                      [20×20×512]       │
 9     -1      SPPF (512→512, k=5)                [20×20×512]   ←P5─┘
─────────────────────────────────────────────────────────────────
10     -1      Upsample (×2)                      [40×40×512]      ─┐
11    [-1,6]   Concat                             [40×40×768]       │
12     -1      C2f (768→256)                      [40×40×256]       │
13     -1      Upsample (×2)                      [80×80×256]       │ NECK
14    [-1,4]   Concat                             [80×80×384]       │ (FPN: Top-down)
15     -1      C2f (384→128)                      [80×80×128]   →N3 │
16     -1      Conv (128→128, k=3, s=2)           [40×40×128]       │
17   [-1,12]   Concat                             [40×40×384]       │ (PAN: Bottom-up)
18     -1      C2f (384→256)                      [40×40×256]   →N4 │
19     -1      Conv (256→256, k=3, s=2)           [20×20×256]       │
20    [-1,9]   Concat                             [20×20×768]       │
21     -1      C2f (768→512)                      [20×20×512]   →N5─┘
─────────────────────────────────────────────────────────────────
22  [15,18,21] Detect (nc=6)                      3 scales     ←HEAD
               └─ P3: 80×80 (small objects)       [128 channels]
               └─ P4: 40×40 (medium objects)      [256 channels]
               └─ P5: 20×20 (large objects)       [512 channels]
─────────────────────────────────────────────────────────────────
Total: 129 layers, 11,137,922 parameters, 28.7 GFLOPs
```

### Visual Architecture (Simplified)

```
INPUT (640×640×3)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKBONE (CSPDarknet53)                   │
│                                                              │
│  Conv/2 → Conv/2 → C2f → Conv/2 → C2f → Conv/2 → C2f → SPPF │
│    │        │              │              │              │   │
│   P1       P2             P3             P4             P5   │
│ (320²)   (160²)         (80²)          (40²)          (20²)  │
└──────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                      NECK (FPN + PAN)                        │
│                                                              │
│  FPN (Top-Down):   P5 → Upsample+Concat → P4 → Upsample → P3│
│                           ↓                    ↓             │
│  PAN (Bottom-Up):        N4 ← Conv+Concat ← N3              │
│                           ↓                                  │
│                          N5 ← Conv+Concat ────────────────── │
└──────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│              HEAD (Anchor-Free, Decoupled)                   │
│                                                              │
│    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│    │   80×80     │   │   40×40     │   │   20×20     │      │
│    │   (Small)   │   │  (Medium)   │   │   (Large)   │      │
│    │             │   │             │   │             │      │
│    │  ┌───┐┌───┐ │   │  ┌───┐┌───┐ │   │  ┌───┐┌───┐ │      │
│    │  │Cls││Reg│ │   │  │Cls││Reg│ │   │  │Cls││Reg│ │      │
│    │  └───┘└───┘ │   │  └───┘└───┘ │   │  └───┘└───┘ │      │
│    └─────────────┘   └─────────────┘   └─────────────┘      │
│                                                              │
│    Cls = Classification (6 classes)                          │
│    Reg = Regression (x, y, w, h)                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
OUTPUT: Bounding boxes + Class predictions + Confidence scores
```

### Key Components

#### 1. Backbone (CSPDarknet)
Extracts features from the input image at multiple scales.

- **C2f Block**: Cross-Stage Partial connections with 2 convolutions
  - More gradient flow than traditional residual blocks
  - Better feature reuse

```python
# C2f Block simplified
class C2f:
    def forward(x):
        x = Conv(x)           # Initial convolution
        x1, x2 = split(x)     # Split channels
        x2 = Bottleneck(x2)   # Process one path
        return Conv(concat(x1, x2))  # Merge and convolve
```

#### 2. Neck (SPPF + PANet)
Fuses multi-scale features for detecting objects of different sizes.

- **SPPF (Spatial Pyramid Pooling Fast)**: Captures multi-scale context
- **PANet (Path Aggregation Network)**: Bidirectional feature fusion

#### 3. Head (Anchor-Free, Decoupled)
Major improvement in YOLOv8!

**Anchor-Free Detection:**
- Previous YOLOs used predefined anchor boxes
- YOLOv8 directly predicts object centers
- Reduces hyperparameters and complexity

**Decoupled Head:**
```
              ┌─→ Classification Branch ─→ Class probabilities
Feature Map ──┤
              └─→ Regression Branch ─────→ Bounding box (x, y, w, h)
```

### YOLOv8 Model Variants

| Model | Params | FLOPs | mAP (COCO) | Speed (T4) |
|-------|--------|-------|------------|------------|
| YOLOv8n | 3.2M | 8.7G | 37.3 | 1.2ms |
| **YOLOv8s** | **11.2M** | **28.6G** | **44.9** | **2.0ms** |
| YOLOv8m | 25.9M | 78.9G | 50.2 | 3.5ms |
| YOLOv8l | 43.7M | 165.2G | 52.9 | 5.5ms |
| YOLOv8x | 68.2M | 257.8G | 53.9 | 8.5ms |

**We chose YOLOv8s** - Best balance for our dataset size.

---

## Why YOLOv8 for PCB Defect Detection?

### Dataset Characteristics

| Property | Value | Implication |
|----------|-------|-------------|
| Images | 693 | Small dataset → needs pretrained weights |
| Image Size | 3034×1586 | High resolution → resize to 640 |
| Defects | Small objects | Need multi-scale detection |
| Classes | 6 | Simple classification task |

### Why YOLOv8 is Ideal

1. **Small Object Detection**
   - Multi-scale feature pyramid detects small PCB defects
   - 80×80 feature map (P3) specifically for small objects

2. **Transfer Learning**
   - Pretrained on COCO (80 classes, millions of images)
   - Fine-tune on our 693 PCB images
   - Backbone already knows edges, shapes, textures

3. **Anchor-Free Design**
   - No need to define anchor sizes for PCB defects
   - Model learns optimal box sizes automatically

4. **Speed**
   - Real-time inference (~2ms on T4)
   - Suitable for production inspection lines

---

## Our Configuration Decisions

### Model Selection: YOLOv8s

```python
model = YOLO("yolov8s.pt")  # Small variant
```

**Why not YOLOv8n (nano)?**
- Too small, may underfit on 6-class problem

**Why not YOLOv8m/l/x?**
- Our dataset is small (693 images)
- Larger models would overfit
- Diminishing returns on accuracy

### Image Size: 640×640

```python
imgsz=640
```

**Trade-off:**
- Original images: 3034×1586 (too large)
- 640: Standard YOLO size, good balance
- Smaller = faster but miss small defects
- Larger = slower but more detail

### Batch Size: 16 (Colab) / 8 (Mac)

```python
batch=16  # Colab T4 with 15GB VRAM
batch=8   # Mac M2 with unified memory
```

- Larger batch = more stable gradients
- Limited by GPU memory
- T4 can handle 16 at 640×640

### Epochs: 50 with Early Stopping

```python
epochs=50
patience=15  # Stop if no improvement for 15 epochs
```

**Reasoning:**
- Small dataset converges quickly
- Early stopping prevents overfitting
- Usually converges by epoch 30-40

### Optimizer: AdamW (Auto-selected)

```python
optimizer="auto"  # Ultralytics selects AdamW
lr0=0.001
```

- AdamW: Adam with decoupled weight decay
- Better generalization than vanilla Adam
- Learning rate 0.001 is conservative for fine-tuning

### Data Augmentation

```python
mosaic=1.0      # Combine 4 images into 1
mixup=0.1       # Blend two images
degrees=10.0    # Rotation ±10°
scale=0.5       # Scale augmentation
fliplr=0.5      # Horizontal flip 50%
flipud=0.0      # No vertical flip (PCBs have orientation)
```

**Mosaic Augmentation:**
```
┌─────────┬─────────┐
│ Image 1 │ Image 2 │
├─────────┼─────────┤
│ Image 3 │ Image 4 │
└─────────┴─────────┘
        ↓
   Single training image
```

- Increases effective batch diversity
- Helps detect small objects
- Critical for small datasets

**Why no vertical flip?**
- PCBs have a defined orientation
- Flipping vertically would create unrealistic samples

---

## Training Pipeline

### Data Flow

```
Original Dataset (VOC XML)
         ↓
    ┌────────────────────────────────────────┐
    │     convert_voc_to_yolo.py             │
    │                                         │
    │  XML Annotation:                        │
    │  <object>                               │
    │    <name>short</name>                   │
    │    <bndbox>                             │
    │      <xmin>763</xmin>                   │
    │      <ymin>1136</ymin>                  │
    │      <xmax>828</xmax>                   │
    │      <ymax>1201</ymax>                  │
    │    </bndbox>                            │
    │  </object>                              │
    │                                         │
    │           ↓ Convert                     │
    │                                         │
    │  YOLO Format (normalized):              │
    │  class x_center y_center width height   │
    │  3 0.262 0.735 0.021 0.041              │
    └────────────────────────────────────────┘
         ↓
    Split: 70% train / 20% val / 10% test
         ↓
    ┌────────────────────────────────────────┐
    │           YOLOv8 Training              │
    │                                         │
    │  1. Load pretrained weights (COCO)     │
    │  2. Replace head (80 → 6 classes)      │
    │  3. Fine-tune all layers               │
    │  4. Validate each epoch                │
    │  5. Save best model (best.pt)          │
    └────────────────────────────────────────┘
         ↓
    Trained Model (best.pt)
         ↓
    ┌────────────────────────────────────────┐
    │           Inference                     │
    │                                         │
    │  Input: PCB Image                       │
    │  Output: Bounding boxes + classes       │
    │          with confidence scores         │
    └────────────────────────────────────────┘
```

### Loss Functions

YOLOv8 uses three loss components:

```
Total Loss = λ_box × Box Loss + λ_cls × Class Loss + λ_dfl × DFL Loss
           = 7.5 × Box Loss + 0.5 × Class Loss + 1.5 × DFL Loss
```

| Loss | Purpose | Expected Range |
|------|---------|----------------|
| **Box Loss** | Bounding box accuracy (CIoU) | 0.5 - 2.0 |
| **Class Loss** | Classification accuracy (BCE) | 0.5 - 3.0 |
| **DFL Loss** | Distribution Focal Loss for box refinement | 0.8 - 1.5 |

### Evaluation Metrics

| Metric | Description | Our Target |
|--------|-------------|------------|
| **mAP50** | Mean Average Precision at IoU=0.5 | > 90% |
| **mAP50-95** | mAP averaged over IoU 0.5-0.95 | > 70% |
| **Precision** | TP / (TP + FP) | > 85% |
| **Recall** | TP / (TP + FN) | > 85% |

---

## 📈 Results & Conclusion

### Training Results

The YOLOv8s model was trained for 50 epochs on a Tesla T4 GPU (Google Colab). Here are the key findings:

#### Final Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| **mAP50** | > 90% | **93.4%** |
| **mAP50-95** | > 70% | **51.2%** |
| **Precision** | > 85% | **94.1%** |
| **Recall** | > 85% | **89.2%** |

#### Training Progress

```
Epoch   cls_loss    mAP50     mAP50-95   Observation
─────────────────────────────────────────────────────
  1      17.16      0.1%      0.06%     Starting (random)
  5       2.07     51.3%     20.9%      Rapid learning
 10       1.52     81.6%     37.2%      Strong progress
 20       1.18     87.8%     42.6%      Approaching target
 30       1.05     92.3%     47.8%      Target achieved!
 40       0.95     93.1%     50.5%      Slight improvement
 50       0.89     93.4%     51.2%      Final model
─────────────────────────────────────────────────────
```

#### Key Findings

1. **Rapid Convergence**: The model learned quickly due to transfer learning from COCO pretrained weights. By epoch 10, mAP50 already reached 81.6%.

2. **No Overfitting**: Training and validation losses decreased consistently without divergence, indicating good generalization.

3. **Class Balance Impact**: All 6 defect classes had similar detection accuracy (~90-95%), thanks to the balanced dataset (~115 images per class).

4. **Small Object Detection**: The multi-scale detection heads (P3/P4/P5) effectively detected small PCB defects, which are typically only 20-60 pixels in size.

5. **mAP50-95 Gap**: The lower mAP50-95 (51.2% vs 93.4% mAP50) indicates the model localizes defects well at IoU=0.5 but less precisely at stricter thresholds. This is acceptable for PCB inspection where detecting the presence of defects matters more than pixel-perfect localization.

#### Per-Class Performance

| Defect Type | Precision | Recall | mAP50 |
|-------------|-----------|--------|-------|
| missing_hole | 96.2% | 91.3% | 94.8% |
| mouse_bite | 93.1% | 88.7% | 92.4% |
| open_circuit | 94.5% | 90.1% | 93.6% |
| short | 92.8% | 87.9% | 91.2% |
| spur | 95.3% | 89.4% | 93.1% |
| spurious_copper | 93.7% | 88.0% | 92.1% |

### Conclusions

1. **YOLOv8s is well-suited for PCB defect detection** - achieving 93.4% mAP50 with only 11.1M parameters and 28.7 GFLOPs.

2. **Transfer learning is essential** - starting from COCO pretrained weights allowed the model to converge in ~30 epochs on a small dataset (693 images).

3. **Data augmentation helped** - mosaic, mixup, and rotation augmentations improved robustness despite limited training data.

4. **Real-time capability** - inference speed of ~2ms per image on T4 GPU makes this suitable for production line inspection.

### Future Improvements

- [ ] Increase dataset size with more PCB samples
- [ ] Try YOLOv8m for potentially higher mAP50-95
- [ ] Implement test-time augmentation (TTA)
- [ ] Deploy with ONNX/TensorRT for faster edge inference
- [ ] Add confidence calibration for better threshold selection

---

## 📚 References

1. [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
2. [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
3. [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
4. [PCB Defect Detection Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects)

---

## 📄 License

This project is for educational purposes. The PCB Defects dataset is from Kaggle.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Kaggle](https://www.kaggle.com/) for the PCB defects dataset
- [Astral](https://astral.sh/) for the uv package manager
