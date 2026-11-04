# FloodShift: SAR Normalization and Augmentation Interaction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Official Code Repository for:** *Investigating the Interaction Between SAR Normalization and Data Augmentation for Cross-Regional Generalization in Sentinel-1 Water Segmentation*

**Author:** Sandeep Hosamani  
**Domain:** Synthetic Aperture Radar (SAR), Deep Learning, Domain Generalization, Flood Segmentation  

---

## 📌 Overview

Deep learning models for Synthetic Aperture Radar (SAR) surface-water segmentation frequently lose accuracy when deployed on geographically unseen flood events. While input normalization and data augmentation are standard preprocessing steps, their combined interaction under geographical distribution shift remains poorly understood. 

This repository contains the experimental code, saved weights, and evaluation metrics for **10 controlled U-Net experiments** trained on the hand-labeled **Sen1Floods11** dataset. We systematically evaluate four SAR normalization strategies (`raw_db`, `minmax_clip`, `zscore`, `percentile`) against four augmentation regimes (`none`, `geometric`, `intensity`, `combined`), plus a Lee speckle-filtering baseline, across In-Distribution (ID) and Out-of-Distribution (OOD) regions (Bolivia, Mekong, Spain).

## 🚀 Key Findings

1. **Crossover Interaction:** Normalization and augmentation do not act independently. Geometric augmentation rescues uncentered Min-Max models (+5.11 percentage points OOD IoU) but degrades well-calibrated Z-score models (-2.41 percentage points OOD IoU) by increasing false positives on directional terrain shadows.
2. **Best Overall Generalization:** Preserving native physical decibel values without geometric distortion (`E0_RawDB_NoAug`) yielded the highest overall unseen generalization (0.7837 OOD IoU).
3. **Synergy in Complex Terrain:** Clipping extreme backscatter outliers to the 1st–99th training percentiles is necessary for radiometric intensity augmentation to succeed, achieving the highest accuracy in the vegetated Bolivian floodplain (0.7313 IoU).
4. **Physical Drivers of Failure:** Regional segmentation accuracy is strictly governed by physical VV/VH water-to-land contrast ($\Delta$dB) and the proportion of dark land pixels mimicking water below -15 dB (e.g., 5.67% in Spain vs. 0.84% in Mekong).

## 📂 Repository Structure

```text
FloodShift-Research-Repo/
│
├── experiments/
│   ├── master_results_v2.csv              # Final 10-model quantitative results 
│   ├── regional_sar_contrast_fixed.csv    # 11-region physical backscatter stats
│   └── weights/                           # Saved U-Net model checkpoints (.keras)
│
├── figures/                               # High-resolution publication figures
│   ├── Fig1_Interaction_Regional_BarChart.png 
│   ├── Fig2_Regional_Histograms.png             
│   ├── Fig3_Convergence_Curves.png              
│   └── Fig4_Visual_Predictions.png              
│
├── notebooks/
│   └── sar_flood_experiments.ipynb        # Primary Jupyter Notebook (Train & Eval)
│
└── manuscript/
    └── FloodShift_ICTACT_Manuscript_3.docx # Publication-ready academic draft
```

## 💾 Dataset Access

Due to size constraints, the Sen1Floods11 satellite imagery is not hosted in this repository. 
To run the notebook, please download the original Sen1Floods11 (v1.1) dataset from the official repository:
👉 [Sen1Floods11 GitHub](https://github.com/cloudtostreet/Sen1Floods11)

Extract the `S1Hand` and `LabelHand` folders into your local directory and update the `s1_dir` and `label_dir` paths in the Jupyter Notebook.

## 🛠️ Requirements

The experiments were conducted using the following primary dependencies:
* `tensorflow == 2.13.0`
* `numpy == 1.24.0`
* `scipy == 1.11.0`
* `rasterio == 1.3.0`
* `pandas`, `matplotlib`, `scikit-learn`

To install the requirements, run:
```bash
pip install tensorflow numpy scipy rasterio pandas matplotlib scikit-learn
```

## 📜 Citation

If you use this code or our findings in your research, please cite the corresponding paper (Publication pending):

```bibtex
@article{hosamani2026floodshift,
  title={Investigating the Interaction Between SAR Normalization and Data Augmentation for Cross-Regional Generalization in Sentinel-1 Water Segmentation},
  author={Hosamani, Sandeep},
  journal={ICTACT Journal on Data Science and Machine Learning},
  year={2026}
}
```