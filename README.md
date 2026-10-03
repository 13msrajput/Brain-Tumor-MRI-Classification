<div align="center">

# Brain Tumor MRI Image Classification

**Deep Learning for Multi-Class Brain Tumor Detection from MRI Scans**

An end-to-end deep learning project that classifies brain MRI images into four categories — Glioma, Meningioma, Pituitary tumour and No Tumor — by comparing a custom CNN built from scratch against four ImageNet-pretrained models (transfer learning), and deploying the best model as an interactive Streamlit app.

**[Live App →](https://Brain-Tumor-MRI-Classification.streamlit.app)**

</div>

> **Disclaimer:** This project is for learning and demonstration only. It is **not** a medical device and must never replace the judgement of a qualified radiologist.

---

## Overview

This project works on 2,443 labelled brain MRI images to:

1. **Classify MRI scans** into four classes — Glioma, Meningioma, Pituitary and No Tumor — using deep learning.
2. **Compare five models** (one custom CNN and four pretrained networks) on accuracy, reliability and efficiency, and pick the best one using validation results.
3. **Serve predictions** through a Streamlit app where a user uploads an MRI image and instantly sees the predicted tumour type, the confidence and the probability of every class.

Possible real-world uses: AI-assisted diagnosis (a fast second reader for radiologists), early detection and patient triage, research and clinical-trial grouping by tumour type, and second-opinion tools for remote or under-resourced regions.

---

## Dataset

[Labeled MRI Brain Tumor Dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset) (Roboflow, CC BY 4.0) — 2,443 MRI images, all 640 × 640 pixels in RGB mode, already divided into train / validation / test folders.

| Split | Images | Share |
|---|---|---|
| Train | 1,695 | 69.4% |
| Validation | 502 | 20.5% |
| Test | 246 | 10.1% |

| Class | Images | % of Dataset | Test Images |
|---|---|---|---|
| Glioma | 805 | 32.95% | 80 |
| Pituitary | 610 | 24.97% | 54 |
| Meningioma | 545 | 22.31% | 63 |
| No Tumor | 483 | 19.77% | 49 |

---

## Data Quality Checks

Every image and label was validated before modelling:

| Check | Result |
|---|---|
| Missing values in the image table | 0 |
| Corrupted / unreadable images | 0 |
| Exact duplicate images (md5 fingerprint) | 0 |
| Folder name vs `_classes.csv` label mismatches | 0 |
| Distinct image sizes | 1 (640 × 640, RGB) |
| Class imbalance ratio (largest ÷ smallest class) | 1.67 (mild) |

**Result: all 2,443 images are usable with no cleaning required.**

### Overlap Between Splits (important finding)

The file names contain the ID of the original scan. About **19.3% of validation images (97 of 502)** and **23.2% of test images (57 of 246)** share a scan ID with a training image. These are most likely augmented copies of the same scan made by the dataset provider. The official split was kept, but the reported scores are probably somewhat optimistic compared with completely new patients (see [Limitations](#limitations)).

---

## Exploratory Data Analysis

Nine charts (univariate, bivariate and multivariate) were produced, each followed by written insights in the notebook. Key findings:

- **Mild class imbalance** — glioma is the largest class (805 images) and no tumour the smallest (483), so class weights and macro-averaged metrics are used.
- **Split distributions are similar** — class proportions are close in train, validation and test; meningioma and pituitary shift by a few percentage points.
- **Scans differ strongly in appearance** — images come from axial, sagittal and coronal views and different scan styles.
- **Brightness differs by class** — in the sampled images, glioma scans are the darkest on average, followed by meningioma, pituitary and no tumour (the brightest). This hints that a model could partly learn scan style instead of the tumour itself, so confusion matrices and misclassified images were studied, not only accuracy.
- **Average class images look like blurred brains** — differences between classes are not visible in a simple average, so the task needs a model that learns local patterns.

All charts are saved in the [`images/`](images) folder.

---

## Preprocessing and Augmentation

| Step | Action |
|---|---|
| 1 | Decoded every image to 3-channel RGB (pretrained models expect 3 channels) |
| 2 | Resized to **224 × 224** |
| 3 | Normalized pixel values to the **0–1** range |
| 4 | Built a `tf.data` pipeline: parallel loading, in-memory cache (images stored as `uint8`), shuffling for training only, batching (32) and prefetching |
| 5 | Applied **augmentation to training data only**: horizontal + vertical flips, rotation (up to ~18°), zoom (15%), shifts (10%), brightness (±20%) and contrast (±20%) |
| 6 | Computed **class weights** to counter the imbalance |

| Class | Class Weight |
|---|---|
| Glioma | 0.751 |
| Meningioma | 1.184 |
| No Tumor | 1.265 |
| Pituitary | 0.967 |

Validation and test images are never augmented, so evaluation stays fair.

---

## Models

### Custom CNN (from scratch)

Four convolutional blocks (32, 64, 128, 256 filters), each convolution followed by **Batch Normalization** and **ReLU**, each block ending with **Max Pooling** and **Dropout (0.25)**. After the blocks: **Global Average Pooling**, a Dense(256) layer with **Dropout (0.5)** and a 4-way **Softmax** output. About 1.23 M parameters.

### Transfer Learning

Four ImageNet-pretrained networks, each with a new head (Global Average Pooling → Batch Normalization → Dropout 0.4 → Dense 256 → Dropout 0.3 → Softmax). A small `Rescaling` layer inside every model converts the 0–1 pixels into the range the pretrained network was trained with. Training uses two phases:

| Phase | What is trained | Learning rate | Max epochs |
|---|---|---|---|
| 1 — Feature extraction | New head only (base frozen) | 1e-3 | 10 |
| 2 — Fine-tuning | Top 30 layers of the base + head (Batch Norm layers stay frozen) | 1e-5 | 10 |

Models: **MobileNetV2**, **ResNet50V2**, **InceptionV3**, **EfficientNetB0**.

### Training Control

All models use the same callbacks: **EarlyStopping** (patience 5, restores best weights), **ModelCheckpoint** (keeps only the best `.h5` file by validation loss) and **ReduceLROnPlateau**. The Adam optimizer and categorical cross-entropy loss are used throughout, with class weights.

---

## Model Comparison

All five models were trained and evaluated with the same data, callbacks and evaluation code. Rows are sorted by validation macro-F1 (the metric used for model selection).

| Model | Weights | Val Accuracy | Val F1 (Macro) | Test Accuracy | Test Precision | Test Recall | Test F1 (Macro) | Tumour Sensitivity |
|---|---|---|---|---|---|---|---|---|
| **ResNet50V2** | ImageNet | **0.8984** | **0.8959** | **0.8618** | **0.8696** | **0.8576** | **0.8546** | 0.9898 |
| InceptionV3 | ImageNet | 0.8825 | 0.8775 | 0.8415 | 0.8490 | 0.8418 | 0.8358 | 0.9848 |
| MobileNetV2 | ImageNet | 0.8466 | 0.8404 | 0.8252 | 0.8391 | 0.8259 | 0.8232 | **0.9949** |
| Custom CNN | Random | 0.7948 | 0.7618 | 0.7846 | 0.8101 | 0.7844 | 0.7530 | 0.9188 |
| EfficientNetB0 | ImageNet | 0.6773 | 0.6608 | 0.6992 | 0.7972 | 0.6786 | 0.6874 | **0.9949** |

**Tumour sensitivity** = of all scans that truly contain a tumour (glioma, meningioma or pituitary), the share that the model did **not** label as "No Tumor". In a medical setting, missing a tumour is the most dangerous mistake, so this matters as much as overall accuracy.

### Efficiency

| Model | Parameters (M) | File Size (MB) | Inference (ms / image) | Training Time (min) |
|---|---|---|---|---|
| ResNet50V2 | 24.10 | 206.4 | 3.04 | 10.2 |
| InceptionV3 | 22.34 | 129.2 | 2.85 | 10.6 |
| MobileNetV2 | 2.59 | 24.3 | **1.21** | 6.4 |
| Custom CNN | 1.23 | **14.2** | 2.16 | 11.1 |
| EfficientNetB0 | 4.38 | 31.0 | 2.60 | 7.9 |

### Selected Model

**ResNet50V2 was selected as the best model.** It has the highest validation F1 (0.8959), which is the selection metric, so the test set stays unbiased, and it also gives the best test accuracy (86.18%), precision, recall and F1.

- Pretrained models beat the custom CNN trained from scratch (78.46% test accuracy) — transfer learning clearly helps on this small dataset.
- **MobileNetV2** is the best lightweight alternative: about 8× smaller than ResNet50V2, the fastest at inference and has the highest tumour sensitivity (99.49%), at a cost of about 3.7 points of accuracy.
- **EfficientNetB0** performed worst on this dataset (69.92%); its best epoch was reached very early (epoch 2), so it is reported as-is.

### Per-Class Results (ResNet50V2, Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Glioma | 0.95 | 0.95 | 0.95 | 80 |
| Meningioma | 0.86 | 0.68 | 0.76 | 63 |
| No Tumor | 0.95 | 0.82 | 0.88 | 49 |
| Pituitary | 0.72 | 0.98 | 0.83 | 54 |
| **Overall accuracy** | | | **0.86** | 246 |

ResNet50V2 misclassified **34 of 246** test images. **Meningioma is the hardest class** for the pretrained models (recall 0.65–0.70), and in the custom CNN it is even weaker (recall 0.30). Confusion matrices and misclassified images for every model are in the notebook and in `images/`.

---

## Streamlit App

The app (`app.py`) lets a user:

- **Upload** a brain MRI image (JPG, JPEG or PNG, up to 10 MB)
- **See the predicted tumour type** with the model confidence
- **See the probability of every class** as a bar chart and a table
- **Switch between the saved models** from the sidebar
- **Read a low-confidence warning** (adjustable threshold) and a medical disclaimer
- **Open the model comparison table** produced during training

Invalid or unreadable files and oversized uploads are rejected with a clear message instead of crashing the app.

---

## Repository Structure

```
Brain-Tumor-MRI-Classification/
├── images/                                  # All charts saved by the notebook
├── models/
│   ├── best_model.h5                        # Model loaded by the app
│   ├── class_names.json                     # Class order used by the models
│   └── model_comparison.csv                 # Comparison table shown in the app
├── .gitignore
├── README.md
├── app.py                                   # Streamlit web application
├── requirements.txt
└── Brain_Tumor_MRI_Classification.ipynb     # Full project: EDA → preprocessing → models → evaluation → comparison
```

The dataset (`data/`) and the other trained models (`custom_cnn.h5`, `mobilenetv2.h5`, `resnet50v2.h5`, `inceptionv3.h5`, `efficientnetb0.h5`) are not stored in this repository because of GitHub's file-size limit. They are produced by running the notebook.

---

## Running Locally

```bash
git clone https://github.com/13msrajput/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
pip install -r requirements.txt
streamlit run app.py
```

The app needs `models/best_model.h5` and `models/class_names.json`. To retrain everything:

1. Download the dataset and place it in a folder named `data` next to the notebook, with `train/`, `valid/` and `test/` inside (each containing `glioma/`, `meningioma/`, `no_tumor/`, `pituitary/`).
2. Open `Brain_Tumor_MRI_Classification.ipynb` and choose *Run All*. A GPU is strongly recommended (for example free Google Colab) and an internet connection is needed the first time to download the ImageNet weights.
3. Set `QUICK_RUN = True` in the configuration cell for a very fast test of the whole notebook, then set it back to `False` for the real run.

---

## Limitations

- **Possible overlap between splits** — about 19% of validation and 23% of test images share a source-scan ID with a training image, so the scores are likely optimistic for brand-new patients. A patient-level split would be a more honest test.
- **Small test set** — with 246 images, one wrong prediction moves accuracy by about 0.4 points, so small differences between models are not meaningful.
- **Scan-style shortcuts** — class differences in brightness and style mean a model may partly learn the scan appearance instead of the tumour.
- **Not clinically validated** — the models were not tested on scans from other hospitals or scanners.

### Future Work

Grad-CAM heatmaps for explainability, patient-level splitting, external validation, and ensembling the best models.

---

## Tech Stack

- **Deep learning:** TensorFlow / Keras (custom CNN, transfer learning with MobileNetV2, ResNet50V2, InceptionV3, EfficientNetB0)
- **Data / ML utilities:** pandas, numpy, scikit-learn (metrics, class weights), Pillow
- **Visualization:** matplotlib, seaborn
- **App:** Streamlit
- **Model format:** HDF5 (`.h5`)

---

# Author

**MOHIT SINGH RAJPUT — AI/ML Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/13msrajput)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/13msrajput)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/13msrajput)
[![LeetCode](https://img.shields.io/badge/LeetCode-181717?style=flat-square&logo=leetcode&logoColor=FFA116)](https://leetcode.com/u/13msrajput)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghrajput1307@gmail.com)

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>