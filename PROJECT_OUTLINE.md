# Fashion Style Classifier — Complete Project Outline

> **Primary deliverable:** Jupyter notebooks  
> **Language/runtime:** Python 3.9+  
> **ML focus:** Multi-class image classification of fashion styles (casual, formal, sporty, streetwear, vintage)

---

## Table of Contents

1. [Project Goals & Scope](#1-project-goals--scope)
2. [Repository Structure](#2-repository-structure)
3. [Notebook Sequence](#3-notebook-sequence)
4. [Data Acquisition & Structure](#4-data-acquisition--structure)
5. [Labeling Strategy](#5-labeling-strategy)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Dataset Splits & Leakage Prevention](#7-dataset-splits--leakage-prevention)
8. [Baselines & Model Families](#8-baselines--model-families)
9. [Model Training](#9-model-training)
10. [Evaluation Methodology & Metrics](#10-evaluation-methodology--metrics)
11. [Experiment Tracking](#11-experiment-tracking)
12. [Interpretability & Explainability](#12-interpretability--explainability)
13. [Error Analysis](#13-error-analysis)
14. [Deployment / Notebook Report Outputs](#14-deployment--notebook-report-outputs)
15. [Milestones & Timeline](#15-milestones--timeline)
16. [Risks, Limitations & Ethics](#16-risks-limitations--ethics)
17. [References](#17-references)

---

## 1. Project Goals & Scope

### Problem Statement
Given a single fashion image (scraped from Reddit or similar social media), automatically assign one of five style labels:
- **casual** — everyday comfortable clothing
- **formal** — office, business, or occasion wear
- **sporty** — athletic or activewear
- **streetwear** — urban, hype, brand-centric style
- **vintage** — retro or thrift-store aesthetic

### Success Criteria
| Metric | Minimum Target |
|---|---|
| Weighted F1-score | ≥ 0.55 |
| Macro F1-score | ≥ 0.45 |
| Per-class recall for minority classes | ≥ 0.40 |

### Scope
- **In scope:** image-only classification, transfer learning, classical ML baseline, interpretability
- **Out of scope:** multi-label classification, video, outfit composition (multiple garments)

---

## 2. Repository Structure

```text
Fashion-Style-Classifier/
│
├── data/
│   ├── raw/                        # Original scraped images (not versioned if large)
│   │   ├── images/                 # All raw downloaded images
│   │   └── metadata/               # Scraper output (URLs, subreddit, timestamps)
│   ├── processed/                  # Cleaned, resized, split-ready images
│   │   ├── train/
│   │   │   ├── casual/
│   │   │   ├── formal/
│   │   │   ├── sporty/
│   │   │   ├── streetwear/
│   │   │   └── vintage/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── labels_file.csv         # Full annotation export (Label Studio)
│       ├── train_labels.csv        # Train split manifest
│       ├── val_labels.csv          # Val split manifest
│       └── test_labels.csv         # Test split manifest (held out until final eval)
│
├── Dataset/                        # Legacy label export (classes.txt, notes.json)
│
├── notebooks/                      # ← PRIMARY DELIVERABLE
│   ├── 00_project_setup.ipynb
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda_and_preprocessing.ipynb
│   ├── 03_dataset_splits.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_cnn_transfer_learning.ipynb
│   ├── 06_vision_transformer.ipynb
│   ├── 07_experiment_comparison.ipynb
│   ├── 08_interpretability.ipynb
│   ├── 09_error_analysis.ipynb
│   └── 10_final_report.ipynb
│
├── src/                            # Reusable Python modules (imported by notebooks)
│   ├── __init__.py
│   ├── data_utils.py               # Dataset loading, splitting helpers
│   ├── augmentation.py             # Augmentation pipelines (Albumentations / torchvision)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py                  # EfficientNet / ResNet wrappers
│   │   ├── vit.py                  # Vision Transformer wrapper
│   │   └── classical.py            # KNN, SVM, Random Forest pipelines
│   ├── train.py                    # Training loop helpers
│   ├── evaluate.py                 # Metrics, plots, report generation
│   └── explainability.py           # Grad-CAM, SHAP helpers
│
├── outputs/
│   ├── models/                     # Saved model checkpoints (.pth, .pkl)
│   ├── figures/                    # Plots, confusion matrices, Grad-CAM overlays
│   ├── metrics/                    # JSON/CSV metric logs per experiment
│   └── reports/                    # Auto-generated HTML/PDF reports from notebooks
│
├── app/                            # (Optional) Gradio / Streamlit demo
│   └── app.py
│
├── Scrape_Reddit_Fashion_Dataset.ipynb   # Data collection notebook (existing)
├── fashion_image_classification_CNN.ipynb         # Existing CNN notebook
├── fashion_image_classification_CNN_Improved.ipynb
├── fashion_image_classification_knn.ipynb         # Existing KNN notebook
├── requirements.txt
├── README.md
└── PROJECT_OUTLINE.md              # This file
```

---

## 3. Notebook Sequence

Run notebooks in order for a reproducible end-to-end pipeline:

| # | Notebook | Purpose | Key Outputs |
|---|---|---|---|
| 00 | `00_project_setup.ipynb` | Environment check, directory setup, seed configuration | Verified environment, folder skeleton |
| 01 | `01_data_acquisition.ipynb` | Reddit scraping (PRAW), download images, save metadata | `data/raw/images/`, `data/raw/metadata/` |
| 02 | `02_eda_and_preprocessing.ipynb` | Class distribution, duplicate detection, image QC, resize, normalize | `data/processed/`, EDA figures |
| 03 | `03_dataset_splits.ipynb` | Stratified train/val/test split, leakage audit | `data/labels/{train,val,test}_labels.csv` |
| 04 | `04_baseline_models.ipynb` | Pixel-histogram features + SVM/RF/KNN baselines | Baseline metric table |
| 05 | `05_cnn_transfer_learning.ipynb` | EfficientNet-B0 & ResNet-50 fine-tuning | Best CNN checkpoint, metrics |
| 06 | `06_vision_transformer.ipynb` | ViT-B/16 or Swin-T fine-tuning, compare with CNN | ViT checkpoint, metrics |
| 07 | `07_experiment_comparison.ipynb` | Side-by-side comparison of all models, statistical tests | Comparison table, charts |
| 08 | `08_interpretability.ipynb` | Grad-CAM (CNNs), attention maps (ViT), SHAP (baselines) | Saliency figures |
| 09 | `09_error_analysis.ipynb` | Confusion deep-dive, hardest examples, inter-class confusion | Error analysis figures |
| 10 | `10_final_report.ipynb` | Narrative summary, all results, conclusions, future work | Self-contained report |

---

## 4. Data Acquisition & Structure

### 4.1 Sources
| Source | Subreddits / Collections | Expected Volume |
|---|---|---|
| Reddit (PRAW) | r/streetwear, r/fashion, r/femalefashionadvice, r/malefashionadvice, r/malelivingspace | ~2 000 raw images |
| Kaggle Fashion datasets | DeepFashion, iMaterialist (supplemental) | Optional augmentation |

### 4.2 Scraping Protocol (`01_data_acquisition.ipynb`)
1. Authenticate with PRAW (credentials stored in `.env` — never committed).
2. Scrape top/hot/new posts; keep only direct image URLs (`.jpg`, `.png`, `.webp`).
3. Save each image to `data/raw/images/<subreddit>/<post_id>.<ext>`.
4. Write a metadata CSV to `data/raw/metadata/scrape_log.csv` with columns:
   `post_id, subreddit, url, downloaded_at, file_path, width, height`.
5. Deduplicate by perceptual hash (pHash, threshold = 10).

### 4.3 Data File Conventions
- All images resized to **224×224** during preprocessing (not raw).
- Raw files preserved in `data/raw/` (large; add to `.gitignore` or use Git LFS).
- Processed files in `data/processed/{train,val,test}/<class>/`.
- Label manifests are CSVs with at minimum: `filename, label, split`.

---

## 5. Labeling Strategy

### 5.1 Tool
[Label Studio](https://labelstud.io/) (open-source, self-hosted).

### 5.2 Annotation Guidelines (summary)
| Class | Key Visual Cues |
|---|---|
| casual | T-shirts, jeans, sneakers; relaxed fit; muted or everyday colors |
| formal | Suits, blazers, dress shoes, button-down shirts; structured silhouette |
| sporty | Athletic wear, leggings, gym shoes, jerseys; technical fabrics |
| streetwear | Hoodies, oversized tees, trainers, branded items; urban aesthetic |
| vintage | Retro cuts, thrift-store finds, faded colors, decades-specific patterns |

Full annotation guide lives in `data/labels/annotation_guidelines.md`.

### 5.3 Quality Control
- Single-annotator for initial pass; flag ambiguous images for review.
- Inter-annotator agreement (Cohen's κ) computed on a 10% sample.
- Ambiguous examples (confidence < 0.7 during model training) flagged for re-annotation.
- Label export from Label Studio in CSV format; columns aligned to `labels_file.csv` schema.

---

## 6. Preprocessing Pipeline

Implemented in `src/data_utils.py` and exercised in `02_eda_and_preprocessing.ipynb`.

### Steps
1. **QC filtering** — remove corrupt/unreadable files, images < 64px on any side.
2. **Duplicate removal** — pHash-based dedup (keep one representative per cluster).
3. **Resize** — 224×224 (bicubic interpolation); preserve aspect ratio with padding for inference.
4. **Normalization** — ImageNet mean/std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
5. **Class-balance visualization** — bar chart of class counts before/after cleaning.
6. **Data augmentation** (train split only, defined in `src/augmentation.py`):
   - Random horizontal flip (p=0.5)
   - Random rotation (±15°)
   - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
   - Random crop → resize back to 224×224
   - (Optional) MixUp / CutMix for minority classes

---

## 7. Dataset Splits & Leakage Prevention

### 7.1 Split Ratios
| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final, held-out evaluation only |

### 7.2 Stratification
- Splits are stratified by class label to preserve the original class distribution.
- Implemented via `sklearn.model_selection.train_test_split(..., stratify=y)`.

### 7.3 Leakage Prevention Checklist
- [ ] All augmentation applied **only to train split** after the split is defined.
- [ ] Normalization statistics (mean, std) computed **only on train split**.
- [ ] Duplicate removal performed **before** splitting; no identical images across splits.
- [ ] Test set labels are **never used** during model selection or hyperparameter tuning.
- [ ] `test_labels.csv` is read only in `10_final_report.ipynb` (final evaluation cell).
- [ ] Random seed fixed globally (`SEED = 42`) for reproducible splits.

### 7.4 Class Imbalance Handling
- Report class weights per split.
- Use `class_weight='balanced'` in scikit-learn classifiers.
- Use `WeightedRandomSampler` in PyTorch DataLoader.
- Consider oversampling minority classes (SMOTE on features, or image duplication + augmentation).

---

## 8. Baselines & Model Families

### 8.1 Strong Baselines (Notebook 04)

| Baseline | Description | Library |
|---|---|---|
| **Majority-class** | Predict most frequent class for all samples | — |
| **Color histogram + SVM** | Flatten HSV histogram (bins=32) → RBF-SVM | scikit-learn |
| **HOG features + Random Forest** | HOG descriptor from grayscale image → RF (n=200) | scikit-learn, skimage |
| **MobileNetV2 features + KNN** | Frozen MobileNetV2 embeddings → PCA → KNN | Keras/TF, scikit-learn |

All baselines include GridSearchCV tuned on the validation set.

### 8.2 Model Family 1 — Convolutional Neural Networks (Notebook 05)

| Model | Backbone | Params (approx.) | Pretrained |
|---|---|---|---|
| EfficientNet-B0 | EfficientNet | 5.3M | ImageNet (timm) |
| ResNet-50 | ResNet | 25M | ImageNet (torchvision) |

- Framework: **PyTorch + timm**
- Strategy: Freeze backbone → train head (5 epochs) → unfreeze top layers → fine-tune (15 epochs)
- Optimizer: AdamW, lr=1e-4; cosine LR scheduler
- Loss: CrossEntropyLoss with class weights

### 8.3 Model Family 2 — Vision Transformers (Notebook 06)

| Model | Backbone | Params (approx.) | Pretrained |
|---|---|---|---|
| ViT-B/16 | Vision Transformer | 86M | ImageNet-21k (timm) |
| Swin-T | Swin Transformer | 28M | ImageNet-1k (timm) |

- Framework: **PyTorch + timm**
- Strategy: Same two-phase fine-tuning as CNN
- Extra: linear probe (frozen backbone) as an intermediate comparison point

### 8.4 Comparison Summary Table

| Model | Architecture type | Parameters | Expected val F1 (est.) |
|---|---|---|---|
| Majority class | Baseline | 0 | ~0.10 |
| Color hist + SVM | Classical | <1K | ~0.25 |
| HOG + RF | Classical | <1K | ~0.30 |
| MobileNetV2 + KNN | Classical + deep features | ~3M (frozen) | ~0.40 |
| EfficientNet-B0 | CNN | 5.3M | ~0.55 |
| ResNet-50 | CNN | 25M | ~0.55 |
| Swin-T | ViT family | 28M | ~0.58 |
| ViT-B/16 | ViT | 86M | ~0.58 |

---

## 9. Model Training

### 9.1 Training Configuration

```python
# Shared config (src/train.py)
SEED        = 42
BATCH_SIZE  = 32
NUM_EPOCHS  = 20        # CNNs; ViTs may need fewer
LR          = 1e-4
WEIGHT_DECAY = 1e-2
IMG_SIZE    = 224
NUM_CLASSES = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
```

### 9.2 Training Loop Features
- **Early stopping** — patience=5 on validation loss.
- **Checkpoint saving** — save best val-F1 checkpoint to `outputs/models/<run_name>_best.pth`.
- **LR scheduling** — cosine annealing over training epochs.
- **Mixed precision** — `torch.cuda.amp.autocast()` when CUDA available (reduces VRAM, speeds up).
- **Gradient clipping** — `max_norm=1.0` for ViTs.

### 9.3 Hyperparameter Tuning
- For baselines: `GridSearchCV` with 3-fold CV on train set.
- For deep models: manual sweep over `{lr, batch_size, weight_decay}` logged via MLflow/wandb.

---

## 10. Evaluation Methodology & Metrics

### 10.1 Primary Metrics
| Metric | Why |
|---|---|
| **Weighted F1-score** | Accounts for class imbalance; primary ranking metric |
| **Macro F1-score** | Equal weight to all classes; tracks minority class performance |
| **Per-class F1, Precision, Recall** | Diagnose per-category strengths/weaknesses |
| **Accuracy** | Reported for context but not the primary ranking metric |

### 10.2 Secondary Metrics
| Metric | Purpose |
|---|---|
| **Confusion matrix** (normalized) | Visualize inter-class confusion |
| **Top-2 accuracy** | Useful when boundary between similar classes is ambiguous |
| **AUC-ROC (one-vs-rest)** | Probabilistic calibration check |
| **Inference latency (ms/image)** | Deployment relevance |

### 10.3 Evaluation Protocol
1. All hyperparameter decisions made on **validation set only**.
2. **Final metrics reported on test set once**, in `10_final_report.ipynb`.
3. 95% confidence intervals via bootstrap (n=1000 resamples) on test set metrics.
4. McNemar's test for pairwise model significance (best CNN vs. best ViT).
5. All metric computations in `src/evaluate.py`; notebooks call functions, not raw sklearn code.

---

## 11. Experiment Tracking

### 11.1 Tool
**MLflow** (local tracking server, no cloud account required).

```bash
mlflow ui  # launch at http://localhost:5000
```

Tracking URI stored in `outputs/mlruns/`.

### 11.2 What Is Logged Per Run
- **Parameters:** model name, backbone, lr, batch_size, weight_decay, augmentation config, seed
- **Metrics (per epoch):** train_loss, val_loss, val_acc, val_f1_weighted, val_f1_macro
- **Artifacts:** best checkpoint path, confusion matrix PNG, classification report JSON

### 11.3 Naming Convention
Run names follow: `<model_family>_<backbone>_<date>_<short_hash>`.  
Example: `cnn_efficientnet_b0_20250801_a3f2`.

---

## 12. Interpretability & Explainability

Implemented in `08_interpretability.ipynb` and `src/explainability.py`.

### 12.1 Grad-CAM (CNN models)
- Target layer: last convolutional block of EfficientNet/ResNet.
- Overlay heatmap on 20 random test images per class.
- Identify spurious correlations (e.g., background bias, clothing tags).

### 12.2 Attention Maps (ViT models)
- Average attention weights from the last attention head across 20 test images.
- Visualize which patches drive classification.

### 12.3 SHAP (Classical baselines)
- `shap.KernelExplainer` for SVM/RF on flattened feature vectors.
- Identify which color histogram bins or HOG regions are most predictive.

### 12.4 Global Feature Importance
- For classical models: `feature_importances_` (RF) or `coef_` magnitude (SVM).
- For deep models: class activation maps averaged over 50 images per class.

---

## 13. Error Analysis

Implemented in `09_error_analysis.ipynb`.

### 13.1 Confusion Deep-Dive
- Normalized confusion matrix for the best model.
- Identify top-3 most confused class pairs (e.g., casual ↔ streetwear).

### 13.2 Hardest Examples
- Extract images with highest cross-entropy loss on the test set.
- Display 10 hardest examples per class with predicted vs. true label.

### 13.3 Slice Analysis
- Breakdown of performance by subreddit source (if metadata available).
- Breakdown by image quality (blur score, brightness extremes).

### 13.4 Annotation Quality Issues
- Images near decision boundaries flagged for potential re-annotation.
- Compute label agreement on ambiguous examples.

---

## 14. Deployment / Notebook Report Outputs

### 14.1 Notebook as Self-Contained Report
`10_final_report.ipynb` is designed to be exported as:
- **HTML** (`jupyter nbconvert --to html`) — primary share format.
- **PDF** (via `nbconvert --to pdf` with LaTeX, or via browser print).

All figures, tables, and narrative are rendered inline.

### 14.2 Interactive Demo (Optional)
`app/app.py` — a Gradio demo:
```python
import gradio as gr
# Load best checkpoint, predict on uploaded image, show label + Grad-CAM overlay
```
Run locally: `python app/app.py` or deploy to Hugging Face Spaces.

### 14.3 Model Export
- PyTorch: save via `torch.save(model.state_dict(), "outputs/models/best_model.pth")`
- ONNX export for lightweight inference: `torch.onnx.export(...)`
- Scikit-learn pipelines: `joblib.dump(pipeline, "outputs/models/baseline_svm.pkl")`

---

## 15. Milestones & Timeline

| Milestone | Deliverable | Target |
|---|---|---|
| **M1 — Data & Labels** | `01_data_acquisition.ipynb` complete; ≥ 500 labeled images | Week 1 |
| **M2 — EDA & Splits** | `02_eda_and_preprocessing.ipynb`, `03_dataset_splits.ipynb` done | Week 2 |
| **M3 — Baselines** | `04_baseline_models.ipynb` with ≥ 3 baselines evaluated | Week 3 |
| **M4 — CNN Models** | `05_cnn_transfer_learning.ipynb`; EfficientNet & ResNet trained | Week 4–5 |
| **M5 — ViT Models** | `06_vision_transformer.ipynb`; ViT-B/16 or Swin-T trained | Week 5–6 |
| **M6 — Comparison** | `07_experiment_comparison.ipynb`; all models vs. baselines | Week 6 |
| **M7 — Interpretability** | `08_interpretability.ipynb`; Grad-CAM + SHAP | Week 7 |
| **M8 — Error Analysis** | `09_error_analysis.ipynb`; confusion analysis, hardest examples | Week 7 |
| **M9 — Final Report** | `10_final_report.ipynb`; exported HTML report | Week 8 |

---

## 16. Risks, Limitations & Ethics

### 16.1 Technical Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Class imbalance degrades minority-class F1 | High | High | Weighted loss, oversampling, augmentation |
| Small dataset (< 1K samples) causes overfitting | Medium | High | Strong augmentation, regularization, pretrained backbones |
| ViT fine-tuning requires GPU; may time out in Colab | Medium | Medium | Use Swin-T (smaller) or linear probe only |
| Reddit API rate limits interrupt scraping | Low | Medium | Add sleep/retry logic; cache metadata |
| Label quality: ambiguous images (casual vs. streetwear) | High | Medium | Annotation guidelines, κ audits, re-annotation loop |

### 16.2 Limitations
- **Single-annotator labels** — subjective fashion categories; ground truth is inherently fuzzy.
- **Reddit demographic bias** — dataset skews toward certain demographics (likely young, Western, urban); model may not generalize to other cultural contexts.
- **Static dataset** — fashion trends evolve; a 2025 model may underperform on 2027 images.
- **No multi-label support** — an outfit can span multiple styles (e.g., casual-sporty); single-label forced choice loses nuance.
- **Image context ignored** — background, occasion, and accessories not explicitly modeled.

### 16.3 Ethical Considerations
- **Consent and privacy:** Reddit images are public posts, but individuals did not explicitly consent to ML training use. Scraped data should not be republished. Face detection and blurring should be applied before sharing the dataset.
- **Stereotype reinforcement:** Fashion classifiers can reinforce gender, race, and body-type stereotypes if training data is unrepresentative. Regularly audit class performance across demographic slices.
- **Misuse potential:** A deployed style classifier could be used for invasive surveillance (e.g., tracking individuals by style). Usage should be limited to opt-in, privacy-preserving applications.
- **Bias amplification:** Models trained on majority-class data (streetwear, casual) may amplify existing over-representation, making minority styles (sporty, vintage) harder to classify.
- **Data minimization:** Only image URL, subreddit, and timestamp metadata are retained from scraping. No PII (usernames, post content) should be stored.

---

## 17. References

- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*. https://arxiv.org/abs/1905.11946
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*. https://arxiv.org/abs/1512.03385
- Dosovitskiy, A. et al. (2020). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. https://arxiv.org/abs/2010.11929
- Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. https://arxiv.org/abs/2103.14030
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*. https://arxiv.org/abs/1708.02002
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*. https://arxiv.org/abs/1610.02391
- Howard, A. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*. https://arxiv.org/abs/1801.04381
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Label Studio Documentation: https://labelstud.io/
- timm library: https://github.com/huggingface/pytorch-image-models
