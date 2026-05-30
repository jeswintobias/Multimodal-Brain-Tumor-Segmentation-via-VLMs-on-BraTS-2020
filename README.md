# Multimodal-Brain-Tumor-Segmentation-via-VLMs-on-BraTS-2020


> **Text-Guided Brain Tumor Segmentation using Vision-Language Models with FLAIR MRI and Radiology Reports**

Multimodal deep learning framework for brain tumor segmentation that fuses FLAIR MRI images with radiology text descriptions. Combines a ResNet-34 Attention U-Net with CLIP-based cross-attention and FiLM conditioning — achieving **Dice 0.8589** on BraTS 2020, within 3.4% of the 3D multi-modal SOTA using only 2D FLAIR slices and text.

**Authors:** Kishore K (CS23B2016) · Jeswin Tobias J (CS23I1013) — Deep Learning Project, Jan–May 2026

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Lessons Learned](#lessons-learned)
- [Setup & Usage](#setup--usage)
- [References](#references)

---

## Overview

Brain tumor segmentation from MRI is critical for treatment planning and surgical guidance. While traditional methods rely solely on imaging, this project demonstrates that integrating radiology text descriptions via Vision-Language Models (VLMs) provides complementary semantic signal that measurably improves segmentation.

**Key contributions:**
- Multimodal fusion of FLAIR MRI + radiology reports via CLIP cross-attention and FiLM conditioning
- Diagnosed and resolved an all-ones prediction collapse caused by class imbalance
- Focal Loss + cosine LR warmup training pipeline that trains stably from the first epoch
- Dice **0.8589** / IoU **0.8188** using only 2D slices and text — no 3D volumes, no additional MRI modalities

---

## Datasets

### Image — BraTS 2020 FLAIR
[Kaggle: hussainnasirkhan/flair-brats2020](https://www.kaggle.com/datasets/hussainnasirkhan/flair-brats2020)

| Split | Volumes | Slices | Tumor | Non-Tumor |
|---|---|---|---|---|
| Train | 258 | 33,024 | 17,317 | 15,707 |
| Validation | 86 | 11,008 | 5,959 | 5,049 |

Each volume is 128 × 128 × 128 voxels. 4-channel BraTS masks (necrotic core, edema, enhancing tumor, background) are collapsed into a single binary whole-tumor mask via logical OR.

### Text — Text-BraTS 2020
[Google Drive](https://drive.google.com/file/d/17YKI4nwPW8qMKlg9k53dVax7F_1JCk9B/view?usp=drivesdk)

- 344 patients, one radiology-style `.txt` report per patient
- Each patient's 128 image slices share the same text embedding
- Reports describe tumor location, edema characteristics, and structural properties

### Preprocessing

| Step | Details |
|---|---|
| Image | Axial slices → resize to 224×224 → normalize [0,1] → 3-channel (RGB-equiv) |
| Mask | 4-channel BraTS → single binary whole-tumor via logical OR |
| Text | OpenCLIP (ViT-B/32) tokenization → 512-dim embeddings; default embedding for missing reports |
| Augmentation | Random horizontal/vertical flips, random rotation ±15° |

---

## Architecture

**Multimodal Attention U-Net** — 56,937,707 parameters

```
FLAIR MRI (224×224×3)
      │
  ResNet-34 Encoder (pretrained)
      │  ├─ L1: 64 × 112 × 112
      │  ├─ L2: 64 × 56 × 56
      │  ├─ L3: 128 × 28 × 28
      │  ├─ L4: 256 × 14 × 14
      │  └─ L5 (bottleneck): 512 × 7 × 7
      │                │
Radiology Text ──► OpenCLIP (ViT-B/32)
                        │
                   512-dim embedding
                        │
              ┌─────────┴──────────┐
         Cross-Attention         FiLM
         (Q=image, K/V=text)   γ(t)⊙x + β(t)
              └─────────┬──────────┘
                        │
            Attention U-Net Decoder
            (transposed conv upsampling)
            + Attention Gates on skip connections
            + Deep supervision at ¼ and ⅛ scale
                        │
            Binary Segmentation Mask
```

**Components:**

- **Image Encoder (ResNet-34):** Pretrained backbone extracting 5-level multi-scale features
- **Attention Gates:** Each skip connection applies a learned gate to suppress background and focus on tumor-relevant regions: `α = σ(Wψ · ReLU(Wg·g + Wx·x + b))`
- **CLIP Cross-Attention:** Text embeddings fused into bottleneck features via: `CrossAttn(Q,K,V) = softmax(QKᵀ/√dk)V`
- **FiLM Conditioning:** Channel-wise affine modulation of bottleneck by text: `FiLM(x) = γ(t) ⊙ x + β(t)`
- **Deep Supervision:** Auxiliary segmentation heads at ¼ and ⅛ resolution

---

## Training Strategy

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 6e-5 (reduced 10× from initial 6e-4) |
| LR scheduler | Cosine annealing with 3-epoch linear warmup |
| Batch size | 16 |
| Epochs | 25 |
| Precision | FP16 (PyTorch AMP) |
| Gradient clipping | Max norm 1.0 |
| Hardware | NVIDIA T4 GPU (Kaggle) |
| Seed | 42 |

**Loss Function:**

```
L_total = L_Dice + L_Focal + 0.3 · (L_DS4 + L_DS3)
```

Focal Loss: `L_Focal = -α(1-pt)^γ · log(pt)` with α=0.75, γ=2.0

BCE was replaced by Focal Loss to handle class imbalance — see [Lessons Learned](#lessons-learned).

---

## Results

### Final Metrics (Validation Set)

| Metric | Image-Only | Multimodal | Δ |
|---|---|---|---|
| Dice Coefficient | 0.8493 | **0.8589** | +0.0096 |
| IoU (Jaccard) | 0.8087 | **0.8188** | +0.0101 |
| Precision | 0.9107 | **0.9187** | +0.0080 |
| Recall | 0.8846 | **0.8875** | +0.0028 |
| Hausdorff Distance | 34.55 | **32.25** | −2.30 |
| BLEU-4 (text) | — | 0.7732 | — |
| ROUGE-L (text) | — | 0.9498 | — |

### Benchmark Comparison

| Method | Modalities | Dim | Dice (WT) |
|---|---|---|---|
| nnU-Net (BraTS 2020 winner) | 4 MRI modalities | 3D | 0.8895 |
| Typical Attention U-Net | 4 MRI modalities | 3D | 0.80–0.88 |
| **Ours (Multimodal)** | **FLAIR + Text** | **2D** | **0.8589** |
| Ours (Image-Only) | FLAIR only | 2D | 0.8493 |

Our model using only 2D FLAIR slices and text is within **3.4%** of the BraTS 2020 winner that uses all four 3D MRI modalities.

### Error Analysis

- Multimodal mean Dice: 0.7617 ± 0.3053
- Image-Only mean Dice: 0.7593 ± 0.3052
- High std driven by bimodal distribution: non-tumor slices score 0.0, tumor slices cluster at 0.85–0.95
- Primary failure cases: very small tumors (< 5% of image area) or boundary slices with minimal tumor extent

---

## Ablation Study

The ablation compares `use_text=False` (image-only) vs `use_text=True` (cross-attention + FiLM), trained identically for 25 epochs with the same hyperparameters and augmentation.

| Finding | Detail |
|---|---|
| Dice improvement | +0.96% |
| IoU improvement | +1.01% |
| Hausdorff improvement | −2.30 units (better boundary precision) |
| Precision vs Recall gain | +0.80% vs +0.28% — text guidance primarily reduces false positives |

---

## Lessons Learned

### Iteration 1 — Model Collapse (All-Ones Prediction)
**Symptom:** After 20 hours of training: Dice=0.087, Recall=1.0, Hausdorff=242.1  
**Root cause:** Standard Dice + BCE loss collapsed to predicting every pixel as tumor (local minimum satisfying BCE while keeping non-zero Dice)  
**Fix:** Replaced BCE with Focal Loss (α=0.75, γ=2.0) to downweight easy background pixels and focus on hard boundary pixels

### Iteration 2 — Learning Rate Too High
**Symptom:** Early training instability from LR=6e-4  
**Fix:** Reduced LR 10× to 6e-5; added 3-epoch linear warmup before cosine annealing to let pretrained ResNet-34 weights adapt gradually

### Iteration 3 — Text Branch Not Contributing
**Symptom:** Multimodal and image-only models produced identical outputs  
**Root cause:** The all-ones collapse masked the text branch's contribution entirely  
**Fix:** Once the base model was stabilized (Focal Loss + LR warmup), the text branch produced a measurable +0.96% Dice improvement

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision open-clip-torch scikit-learn scipy
```

Key libraries:
- PyTorch 2.x + CUDA 12.1
- OpenCLIP (ViT-B/32, `laion2b_s34b_b79k`)
- torchvision (ResNet-34 pretrained weights)
- scikit-learn (t-SNE)
- scipy (Hausdorff distance)

### Data Preparation

1. Download BraTS 2020 FLAIR from [Kaggle](https://www.kaggle.com/datasets/hussainnasirkhan/flair-brats2020)
2. Download Text-BraTS 2020 from [Google Drive](https://drive.google.com/file/d/17YKI4nwPW8qMKlg9k53dVax7F_1JCk9B/view?usp=drivesdk)
3. Match each patient's FLAIR volume to the corresponding `<patient_id>_flair_text.txt` file

### Training

Open and run the Jupyter notebook. All output cells are visible for reproducibility. Random seed is fixed at 42.

```python
# Multimodal model
model = MultimodalAttentionUNet(use_text=True)

# Image-only baseline
model = MultimodalAttentionUNet(use_text=False)
```

---

## Future Work

- Extend to 3D segmentation using volumetric convolutions
- Incorporate additional MRI modalities (T1, T1ce, T2)
- Explore learnable text prompts (prompt tuning) instead of fixed CLIP embeddings
- Apply HD95 (95th percentile Hausdorff) for fairer comparison with published benchmarks

---

## References

1. Menze et al. — "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)" — IEEE Trans. Medical Imaging, 2015
2. Isensee et al. — "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" — Nature Methods, 2021
3. Oktay et al. — "Attention U-Net: Learning Where to Look for the Pancreas" — MIDL, 2018
4. Radford et al. — "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" — ICML, 2021
5. Perez et al. — "FiLM: Visual Reasoning with a General Conditioning Layer" — AAAI, 2018
6. Lin et al. — "Focal Loss for Dense Object Detection" — ICCV, 2017
7. Bakas et al. — "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features" — Scientific Data, 2017
