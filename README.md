# Agricultural Crops Image Classification using Deep Learning  
**MSc Project – Military Institute of Science and Technology (MIST), Dhaka**

**Student:** Md Wahid Rahman
**Tentative Title (from proposal):** Agricultural Crops Image Classification Using Machine Learning

---

## 1. Project Overview

This repository implements a complete, submission-ready MSc-level system for **automatic classification of agricultural crops from images**.  

It directly addresses the challenges stated in the original project proposal:

| Challenge in Proposal | Solution Implemented |
|-----------------------|----------------------|
| Limited high-quality training data | **Advanced data enrichment** via heavy on-the-fly augmentation (RandomResizedCrop, ColorJitter, Affine, Perspective, GaussianBlur, RandomErasing) + class-balanced sampling |
| Variability in crop appearance | Strong geometric + photometric augmentations + transfer learning |
| Model black-box nature | **NEW FEATURE: Grad-CAM explainability** – visualizes which image regions drive the prediction |
| Practical usefulness for farmers | **NEW FEATURE: Interactive Gradio app** with crop-specific agronomic recommendations |
| Limited computational resources | Lightweight ResNet-18 / MobileNet, works on CPU, images loaded directly from zip (no full extraction needed) |

### Supported Classes
1. Almond  
2. Coconut  
3. Sunflower  
4. Tomato  
5. Vigna radiata (Mung bean)  
6. Wheat  

---

## 2. Dataset

- Source: `Crop.zip` (Agricultural-crops folder structure)  
- Original size: ~1044 images, heavily imbalanced (coconut only 93 images)  
- **Enrichment strategy** (instead of manual labeling which is time-consuming):
  - Cap at 80 images/class for balanced training experiments
  - Heavy augmentation expands effective dataset size dramatically during training
  - WeightedRandomSampler further balances minority classes

Run dataset analysis:
```bash
python -c "from src.dataset import analyze_dataset; print(analyze_dataset())"
```

---

## 3. Methodology (aligned with proposal Section 8)

1. **Data collection** – Crop.zip provided  
2. **Data preprocessing & enrichment** – resize, normalize, advanced augmentation  
3. **Feature extraction** – Transfer learning (ResNet-18 backbone pretrained on ImageNet)  
4. **Model selection** – ResNet-18 (default), MobileNet-V3-Small, EfficientNet-B0  
5. **Model training** – AdamW + label smoothing + ReduceLROnPlateau + early stopping  
6. **Model evaluation** – Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report  

**Additional MSc-level contributions:**
- Grad-CAM visual explanations  
- End-to-end interactive web application  
- Reproducible pipeline with saved metrics & plots  

---

## 4. Project Structure

```
Agricultural_Crop_Classification/
├── app/
│   └── app.py                 # Gradio interactive demo (NEW FEATURE)
├── src/
│   ├── dataset.py             # Zip/Folder dataset + heavy augmentation
│   ├── models.py              # Transfer-learning model factory
│   ├── train.py               # Full training & evaluation script
│   └── explainability.py      # Grad-CAM implementation (NEW FEATURE)
├── models/                    # Saved checkpoints (best_*.pth)
├── results/                   # metrics.json, confusion_matrix.png, training_curves.png
├── docs/                      # Extended documentation
├── requirements.txt
└── README.md
```

---

## 5. Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model (CPU-friendly defaults)
```bash
cd Agricultural_Crop_Classification
python src/train.py --epochs 12 --batch-size 8 --img-size 160 --max-per-class 80 --model resnet18
```

### Launch the interactive app
```bash
python app/app.py
```
Then open http://localhost:7860 in a browser. Upload any crop image.

---

## 6. New Features Added (beyond original proposal)

### 6.1 Automated Dataset Enrichment
Instead of manual collection of more images (time-consuming as noted by the teacher), we apply a strong augmentation pipeline + class-balanced sampling. This artificially expands the training distribution and mitigates the limited-data problem identified in the proposal.

### 6.2 Grad-CAM Explainability
Deep learning models are often treated as black boxes. Grad-CAM generates heatmaps showing which parts of the leaf/plant the network focuses on when making a decision. This increases trust and can help detect dataset biases.

### 6.3 Practical Decision-Support App
A Gradio web interface that not only classifies the crop but also returns climate, soil, water requirements and actionable farming tips for the predicted class – turning pure research into a usable tool.

---

## 7. Expected Outcomes (matching proposal)

- Accurate multi-class crop recognition  
- Improved crop monitoring potential  
- Foundation for future disease detection / yield estimation modules  
- Fully documented, reproducible MSc submission package  

---

## 8. Resources Used (from proposal + additions)

- Python 3  
- PyTorch + torchvision  
- NumPy, Pandas, scikit-learn  
- Matplotlib, Seaborn  
- Gradio (interactive demo)  
- Pillow  

---