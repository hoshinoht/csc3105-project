# LOS/NLOS UWB Wireless Signal Classification for Indoor Precise Positioning

**CSC3105 Data Analytics and AI -- Mini-Project**\
Lab P1, Group 4 | Singapore Institute of Technology / University of Glasgow

## Overview

This project addresses two-path LOS/NLOS classification and path-wise distance estimation for Ultra-Wideband (UWB) indoor positioning. Given a single channel impulse response (CIR) measurement, we extract the two dominant propagation paths, classify each as LOS or NLOS, and estimate their physical propagation distances.

We implement a dual-pipeline architecture:

- **Pipeline 1 (Feature-Engineered ML)**: 25 physics-motivated features per path, evaluated with Logistic Regression, SVM, Random Forest, Gradient Boosted Trees (GBT), and XGBoost.
- **Pipeline 2 (End-to-End DL)**: A hybrid CNN+Transformer that operates directly on the raw 1016-sample CIR waveform.

## Key Results

| Model | Accuracy | AUC | Pipeline |
|---|---|---|---|
| XGBoost | 0.9404 | 0.9834 | Feature-engineered (expanded 2-path) |
| Ensemble (Stacked) | 0.9415 | 0.9830 | Feature-engineered (expanded 2-path) |
| CNN+Transformer + Aug | 0.9421 | 0.9845 | Raw-CIR (original split) |

The ML and DL results come from separate pipelines and should be compared qualitatively.

## Project Structure

```
.
├── main.py                  # Full pipeline orchestration
├── evaluation.ipynb         # Jupyter notebook with evaluation
├── src/
│   ├── preprocessing.py     # Data cleaning, CIR normalization, scaling
│   ├── peak_detection.py    # Two-path extraction algorithm
│   ├── feature_engineering.py  # 25-feature per-path computation
│   ├── classification.py    # GridSearchCV training (LR, SVM, RF, GBT, XGB)
│   ├── dl_models.py         # CIRTransformerClassifier architecture
│   ├── dl_training.py       # CNN+Transformer training loop
│   ├── regression.py        # Distance estimation (Ridge, RF, GBT, XGB)
│   ├── ensemble.py          # Averaging and stacked generalization
│   ├── clustering.py        # K-Means and DBSCAN unsupervised analysis
│   ├── synthetic_data.py    # SMOTE and CIR waveform augmentation
│   └── visualization.py     # 24 diagnostic plots
├── dataset/                 # UWB LOS/NLOS dataset (7 CSV files, 42K samples)
├── plots/                   # Generated visualizations
├── Report/
│   ├── DAReport/            # Full project report (PDF + markdown draft)
│   └── ieee/                # IEEE conference paper (6-page, LaTeX)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Hardware acceleration is auto-detected (CUDA > Apple MPS > CPU) for the deep learning pipeline.

## Team

| Member | SIT ID | Glasgow ID |
|---|---|---|
| Po Haoting | 2401280 | 3070642P |
| Travis Neo Kuang Yi | 2401250 | 3070641N |
| Chiang Porntep | 2403352 | 3070566C |
| Nico Caleb Lim | 2401536 | 3070658L |
| Dui Ru En Joshua | 2402201 | 3070683D |

---

## Dataset Acknowledgement

The dataset used in this project is the **UWB LOS and NLOS Data Set** created using the [SNPN-UWB](http://www.log-a-tec.eu/mtc.html) board with the DecaWave [DWM1000](http://www.decawave.com/sites/default/files/resources/dwm1000-datasheet-v1.3.pdf) UWB radio module. It contains 42,000 samples (21,000 LOS + 21,000 NLOS) collected across 7 indoor environments at the Jozef Stefan Institute, Slovenia.

### Citation

If you use this dataset, please cite:

> Klemen Bregar, Andrej Hrovat, Mihael Mohorcic, ["NLOS Channel Detection with Multilayer Perceptron in Low-Rate Personal Area Networks for Indoor Localization Accuracy Improvement"](https://www.researchgate.net/publication/308986067_NLOS_Channel_Detection_with_Multilayer_Perceptron_in_Low-Rate_Personal_Area_Networks_for_Indoor_Localization_Accuracy_Improvement). Proceedings of the 8th Jozef Stefan International Postgraduate School Students' Conference, Ljubljana, Slovenia, May 31-June 1, 2016.

### Author and License

Author of the UWB LOS and NLOS Data Set: Klemen Bregar, **klemen.bregar@ijs.si**

Copyright (C) 2017 SensorLab, Jozef Stefan Institute http://sensorlab.ijs.si

### Funding Acknowledgement

The research leading to the dataset has received funding from the European Horizon 2020 Programme project eWINE under grant agreement No. 688116.
