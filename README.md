# FloodShift: SAR Normalization and Augmentation Interaction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/Sandeepssh/FloodShift/releases/tag/v1.0.0)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](manuscript/FloodShift_Manuscript.pdf)

Official code and model repository for:  
**"Investigating the Interaction Between SAR Normalization and Data Augmentation for Cross-Regional Generalization in Sentinel-1 Water Segmentation"**

**Author:** Sandeep Hosamani  
**Affiliation:** Independent Researcher, Vijayapura, Karnataka, India  
**E-mail:** [hosamanisandeep8@gmail.com](mailto:hosamanisandeep8@gmail.com) | **ORCID:** [0009-0002-4851-7183](https://orcid.org/0009-0002-4851-7183)  
**Manuscript:** [`manuscript/FloodShift_Manuscript.pdf`](manuscript/FloodShift_Manuscript.pdf)

---

## 📌 Overview

Deep learning models for Synthetic Aperture Radar (SAR) flood segmentation often suffer significant accuracy drops when evaluated on geographically unseen flood events. This study evaluates how **four SAR normalization methods** interact with **four augmentation regimes** across 10 controlled U-Net experiments on the 446 hand-labeled Sentinel-1 chips of the **Sen1Floods11** benchmark.

Models are trained on 5 familiar source regions, validated on 3 regions, and tested on both held-out in-distribution (ID) chips and 3 unseen out-of-distribution (OOD) events across three continents: **Bolivia** (vegetated Amazonian floodplain), **Mekong** (monsoonal delta), and **Spain** (semi-arid Mediterranean terrain).

---

## 🚀 Key Findings

1. **Crossover Interaction:** Normalization and augmentation do not act independently. Geometric flips and rotations rescue uncentered Min-Max scaling (**+5.11 percentage points OOD IoU**, $0.6889 \to 0.7400$) by regularizing boundaries, but **degrade** zero-centered Z-score models (**-2.41 percentage points OOD IoU**, $0.7771 \to 0.7530$) due to increased false positives on terrain shadows in Spain.
2. **Native Physical Decibels Generalize Best:** Preserving raw decibel values without augmentation (`E0_RawDB_NoAug`) achieves the highest overall unseen generalization (**0.7837 OOD IoU**, **0.8788 Dice**, **0.8965 Precision**).
3. **Floodplain Synergy:** Pairing 1st–99th percentile clipping with radiometric intensity augmentation (`E7_Percentile_Intensity`) achieves the best accuracy in vegetated Bolivia (**0.7313 IoU**), giving a **+7.24 percentage point** difference-in-differences interaction contrast over unclipped Z-score augmentation.
4. **Physical Drivers of Error:** Segmentation performance is governed by physical VV/VH water-to-land contrast ($\Delta\text{dB}$) and the percentage of dark land pixels mimicking water below $-15\text{ dB}$ (e.g., $5.67\%$ in Spain vs. $0.84\%$ in Mekong).

---

## 🔬 Core Preprocessing Formulations

For polarization band $B \in \{\text{VV}, \text{VH}\}$:

* **Raw Decibels (`raw_db`):**
  $$B_{\text{raw}} = B$$

* **Min-Max Scaling (`minmax_clip`):**
  $$B_{\text{mm}} = \frac{\min(\max(B, -30.0), 0.0) + 30.0}{30.0}$$

* **Z-Score Standardization (`zscore`):**
  $$B_z = \frac{B - \mu_B}{\sigma_B}$$

* **Percentile Winsorization (`percentile`):**
  $$B_{\text{pct}} = \frac{\min(\max(B, P_{01,B}), P_{99,B}) - \mu_B}{\sigma_B}$$

* **Radiometric Intensity Augmentation (`intensity`):**
  $$X_{\text{aug}} = X + \delta + \eta, \quad \delta \sim \mathcal{U}(-0.15, 0.15), \quad \eta \sim \mathcal{N}(0, 0.08^2)$$
  where $\delta$ is a shared scalar calibration shift and $\eta$ is independent per-pixel, per-channel noise.

* **Training Set Parameters (256 chips, leak-free):**
  - $\text{VV: } \mu = -10.14\text{ dB}, \; \sigma = 3.81\text{ dB}, \; P_{01} = -22.50\text{ dB}, \; P_{99} = -1.50\text{ dB}$
  - $\text{VH: } \mu = -16.93\text{ dB}, \; \sigma = 4.36\text{ dB}, \; P_{01} = -29.50\text{ dB}, \; P_{99} = -7.50\text{ dB}$

---

## 📊 Master Benchmark Results

*Evaluated on the 75 unseen OOD test chips (Bolivia, Mekong, Spain) and 43 held-out ID chips at threshold $\hat{y} > 0.5$.*

| Experiment ID | Normalization | Augmentation | ID IoU | Bolivia IoU | Mekong IoU | Spain IoU | OOD IoU | OOD Dice | OOD Prec | OOD Rec |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0_RawDB_NoAug** | `raw_db` | `none` | 0.4809 | 0.6898 | 0.8563 | **0.7174** | **0.7837** | **0.8788** | **0.8965** | 0.8617 |
| **E2_ZScore_NoAug** | `zscore` | `none` | **0.5131** | 0.6756 | **0.8656** | 0.6951 | 0.7771 | 0.8746 | 0.8647 | 0.8847 |
| **E3_Percentile_NoAug** | `percentile` | `none` | 0.4797 | 0.6955 | 0.8544 | 0.6988 | 0.7760 | 0.8739 | 0.8855 | 0.8625 |
| **E5_ZScore_Intensity** | `zscore` | `intensity` | 0.5112 | 0.6390 | 0.8541 | 0.6819 | 0.7614 | 0.8645 | 0.8659 | 0.8632 |
| **E7_Percentile_Intensity** | `percentile` | `intensity` | 0.4587 | **0.7313** | 0.8570 | 0.6335 | 0.7538 | 0.8596 | 0.8146 | 0.9098 |
| **E4_ZScore_Geometric**\* | `zscore` | `geometric` | 0.4288 | 0.7273 | 0.8642 | 0.6269 | 0.7530 | 0.8591 | 0.8162 | 0.9067 |
| **E8_ZScore_LeeFilter** | `zscore` | `none` ($5\times 5$ Lee) | 0.4513 | 0.7226 | 0.8384 | 0.6343 | 0.7447 | 0.8536 | 0.8112 | 0.9008 |
| **MinMax_Geometric**\* | `minmax_clip` | `geometric` | 0.4371 | 0.7015 | 0.8566 | 0.6122 | 0.7400 | 0.8506 | 0.8033 | 0.9038 |
| **E6_ZScore_Combined** | `zscore` | `combined` | 0.4534 | 0.7137 | 0.8535 | 0.5932 | 0.7311 | 0.8447 | 0.7754 | **0.9276** |
| **E1_MinMax_NoAug** | `minmax_clip` | `none` | 0.4031 | 0.6964 | 0.8080 | 0.5520 | 0.6889 | 0.8158 | 0.7539 | 0.8887 |

*\*Pilot runs trained on 299 familiar chips; their OOD metrics are strictly held-out and leak-free.*

---

## 🗺️ Dataset & Reproducibility Splits

The 446 hand-labeled Sentinel-1 chips ($512 \times 512$ pixels, 10 m resolution) are partitioned into four disjoint subsets:

* **Training Set (256 chips):** USA (59), India (58), Paraguay (57), Ghana (46), Sri Lanka (36).
* **Validation Set (72 chips):** Somalia (26), Pakistan (28), Nigeria (18) — used for checkpoint early stopping.
* **In-Distribution Test (43 chips):** Deterministic 1-in-7 systematic holdout from familiar regions.
* **Out-of-Distribution Test (75 chips):** Unseen geographical holdouts — Bolivia (15), Mekong (30), Spain (30).

All split file lists are provided in [`data_splits/`](data_splits/):
- [`train_files_clean.txt`](data_splits/train_files_clean.txt)
- [`val_files.txt`](data_splits/val_files.txt)
- [`id_test_files.txt`](data_splits/id_test_files.txt)
- [`ood_test_files.txt`](data_splits/ood_test_files.txt)
- [`all_splits.csv`](data_splits/all_splits.csv)

---

## 🖼️ Key Figures

| Figure | Description |
|:---:|:---|
| ![Fig1](figures/Fig1_Interaction_Regional_BarChart.png) | **Fig. 1:** Normalization $\times$ Augmentation crossover interaction and regional IoU comparisons. |
| ![Fig2](figures/Fig2_Regional_Histograms.png) | **Fig. 2:** Normalized VV backscatter density histograms for water and land in Bolivia, Mekong, and Spain. |
| ![Fig3](figures/Fig3_Convergence_Curves.png) | **Fig. 3:** 10-epoch validation IoU convergence trajectories for representative models. |
| ![Fig4](figures/Fig4_Visual_Predictions.png) | **Fig. 4:** Qualitative segmentation example on unseen test chip `Bolivia_103757_S1Hand.tif`. |

---

## 🛠️ Quickstart

### 1. Installation
```bash
git clone https://github.com/Sandeepssh/FloodShift.git
cd FloodShift
pip install -r requirements.txt
```

### 2. Dataset
Download the hand-labeled Sen1Floods11 (v1.1) dataset from [Cloud to Street](https://github.com/cloudtostreet/Sen1Floods11) and extract `S1Hand/` and `LabelHand/`.

### 3. Run Training
```bash
# Train E0 (Raw dB baseline - Best OOD Generalization)
python src/train.py --config experiments/configs/E0_RawDB_NoAug.json --data_dir /path/to/sen1floods11

# Train E7 (Percentile + Intensity - Best in Bolivia)
python src/train.py --config experiments/configs/E7_Percentile_Intensity.json --data_dir /path/to/sen1floods11
```

### 4. Interactive Notebook
Explore training curves, histogram analysis, and qualitative predictions:
```bash
jupyter notebook notebooks/sar_flood_experiments.ipynb
```

---

## 💾 Pretrained Model Weights

All 10 trained U-Net `.keras` models are tracked in [`experiments/weights/`](experiments/weights/) and downloadable as release assets on **[Release v1.0.0](https://github.com/Sandeepssh/FloodShift/releases/tag/v1.0.0)**:

* `E0_RawDB_NoAug.keras` (90.1 MB) — Highest overall OOD IoU (**0.7837**)
* `E1_MinMax_NoAug.keras` (90.1 MB) — Min-Max baseline without augmentation
* `E2_ZScore_NoAug.keras` (90.1 MB) — Z-score baseline without augmentation
* `E3_Percentile_NoAug.keras` (90.1 MB) — 1st–99th percentile baseline
* `E4_ZScore_Geometric.keras` (90.1 MB) — Z-score with geometric augmentation
* `E5_ZScore_Intensity.keras` (90.1 MB) — Z-score with intensity augmentation
* `E6_ZScore_Combined.keras` (90.1 MB) — Z-score with combined augmentation
* `E7_Percentile_Intensity.keras` (90.1 MB) — Highest Bolivia IoU (**0.7313**)
* `E8_ZScore_LeeFilter.keras` (90.1 MB) — $5\times 5$ Lee speckle-filtered baseline
* `MinMax_Geometric.keras` (90.1 MB) — Min-Max with geometric augmentation

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).  
Sentinel-1 satellite imagery courtesy of the European Space Agency (ESA) Copernicus Programme and Cloud to Street.
