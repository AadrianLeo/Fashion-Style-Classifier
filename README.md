
# Fashion Style Classifier

End-to-end machine learning project for classifying fashion images into five style categories using Jupyter notebooks as the primary deliverable. The project compares classical ML baselines, CNN transfer learning (EfficientNet, ResNet), and Vision Transformer models (ViT, Swin-T).

> **Full project plan → [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)**

---

## Quick Start

```bash
git clone https://github.com/AadrianLeo/Fashion-Style-Classifier.git
cd Fashion-Style-Classifier
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order (see [Notebook Sequence](#notebook-sequence) below).

---

## Style Classes

| Class | Description |
|---|---|
| **casual** | Everyday comfortable clothing — T-shirts, jeans, sneakers |
| **formal** | Office or occasion wear — suits, blazers, dress shoes |
| **sporty** | Athletic and activewear — leggings, jerseys, gym shoes |
| **streetwear** | Urban, brand-centric style — hoodies, oversized tees, trainers |
| **vintage** | Retro or thrift-store aesthetic — faded colors, decade-specific cuts |

---

## Repository Structure

```text
Fashion-Style-Classifier/
├── data/
│   ├── raw/               # Scraped images & metadata (gitignored if large)
│   ├── processed/         # Cleaned, resized, split images (train/val/test)
│   └── labels/            # Annotation CSVs (full + per-split)
├── Dataset/               # Legacy Label Studio export (classes.txt, labels_file.csv)
├── notebooks/             # ← PRIMARY DELIVERABLE (run in order 00–10)
├── src/                   # Reusable Python modules (data_utils, models, evaluate, …)
├── outputs/
│   ├── models/            # Saved checkpoints (.pth, .pkl)
│   ├── figures/           # Plots, confusion matrices, Grad-CAM overlays
│   └── metrics/           # JSON/CSV metric logs
├── app/                   # Optional Gradio demo
├── requirements.txt
├── README.md
└── PROJECT_OUTLINE.md     # Detailed end-to-end project plan
```

---

## Notebook Sequence

| # | Notebook | Purpose |
|---|---|---|
| 00 | `notebooks/00_project_setup.ipynb` | Environment check, seed, folder creation |
| 01 | `notebooks/01_data_acquisition.ipynb` | Reddit scraping, image download, metadata |
| 02 | `notebooks/02_eda_and_preprocessing.ipynb` | EDA, QC, resize, normalization |
| 03 | `notebooks/03_dataset_splits.ipynb` | Stratified 70/15/15 split, leakage audit |
| 04 | `notebooks/04_baseline_models.ipynb` | SVM, Random Forest, MobileNetV2+KNN baselines |
| 05 | `notebooks/05_cnn_transfer_learning.ipynb` | EfficientNet-B0 & ResNet-50 fine-tuning |
| 06 | `notebooks/06_vision_transformer.ipynb` | ViT-B/16 & Swin-T fine-tuning |
| 07 | `notebooks/07_experiment_comparison.ipynb` | Side-by-side model comparison, stat tests |
| 08 | `notebooks/08_interpretability.ipynb` | Grad-CAM, attention maps, SHAP |
| 09 | `notebooks/09_error_analysis.ipynb` | Confusion deep-dive, hardest examples |
| 10 | `notebooks/10_final_report.ipynb` | Narrative report — export to HTML/PDF |

---

## Models & Baselines

### Baselines (Notebook 04)
- **Majority-class** — always predicts the most frequent class
- **Color histogram + SVM** — HSV histogram features with RBF-SVM
- **HOG + Random Forest** — gradient orientation features with RF (n=200)
- **MobileNetV2 + KNN** — frozen deep features → PCA → KNN

### Model Family 1 — CNNs (Notebook 05)
- **EfficientNet-B0** (PyTorch + timm, 5.3M params)
- **ResNet-50** (torchvision, 25M params)

### Model Family 2 — Vision Transformers (Notebook 06)
- **ViT-B/16** (timm, ImageNet-21k pretrained, 86M params)
- **Swin-T** (timm, ImageNet-1k pretrained, 28M params)

---

## Evaluation

- **Primary metric:** Weighted F1-score (target ≥ 0.55)
- **Secondary metrics:** Macro F1, per-class precision/recall, accuracy, confusion matrix
- **Protocol:** All tuning on validation set; test set evaluated once in `10_final_report.ipynb`
- **Confidence intervals:** Bootstrap (n=1000) on test metrics
- **Experiment tracking:** MLflow (local; run `mlflow ui` to browse runs)

---

## Key Results (Existing Notebooks)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| KNN (MobileNetV2 features) | ~41% | ~0.37 |
| EfficientNet-B0 (CNN) | ~41% | ~0.38 |

> Best F1 per class: Formal & Streetwear (~0.55); Sporty (~0.00, class imbalance)

---

## Experimental Setup

- **Data:** ~500 images from Reddit (r/streetwear, r/fashion, r/femalefashionadvice), manually labeled with Label Studio.
- **Preprocessing:** Corrupt-file removal, class-balance check, stratified split, ImageNet normalization.
- **Augmentation (train only):** Random flip, rotation ±15°, color jitter, random crop.
- **Class imbalance handling:** Weighted loss, `WeightedRandomSampler`, optional oversampling.
- **Seed:** `SEED = 42` throughout.

---

## Challenges & Literature Context

### Challenges
- **Class imbalance** — Sporty and Vintage classes are underrepresented.
- **Noisy real-world data** — varied quality, backgrounds, and styles from Reddit.
- **Limited labeled data** — manual annotation constrains dataset size.
- **Fuzzy boundaries** — casual vs. streetwear are visually similar.

### Relevant Literature
| Paper | Relevance |
|---|---|
| [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) | CNN backbone |
| [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) | Transformer backbone |
| [Swin-T (Liu et al., 2021)](https://arxiv.org/abs/2103.14030) | Hierarchical ViT |
| [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002) | Class imbalance |
| [Grad-CAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391) | Interpretability |

---

## Risks & Ethics

See [PROJECT_OUTLINE.md § 16](PROJECT_OUTLINE.md#16-risks-limitations--ethics) for the full discussion.

**Key points:**
- Reddit images are public but individuals did not explicitly consent to ML use; faces should be blurred before sharing.
- Model may reflect demographic biases present in Reddit data (skews young, Western, urban).
- Classifier should not be used for surveillance or any non-opt-in application.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `timm`, `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `mlflow`, `shap`, `gradio`, `praw`, `jupyter`.

---

## References

- Tan, M. & Le, Q. (2019). EfficientNet. https://arxiv.org/abs/1905.11946
- Dosovitskiy, A. et al. (2020). ViT. https://arxiv.org/abs/2010.11929
- Liu, Z. et al. (2021). Swin Transformer. https://arxiv.org/abs/2103.14030
- Lin, T.-Y. et al. (2017). Focal Loss. https://arxiv.org/abs/1708.02002
- Selvaraju, R. R. et al. (2017). Grad-CAM. https://arxiv.org/abs/1610.02391
- Howard, A. et al. (2018). MobileNetV2. https://arxiv.org/abs/1801.04381
- [Label Studio](https://labelstud.io/) | [MLflow](https://mlflow.org/) | [timm](https://github.com/huggingface/pytorch-image-models)

---

For questions or contributions, please open an issue or submit a pull request.