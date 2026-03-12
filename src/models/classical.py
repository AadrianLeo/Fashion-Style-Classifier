"""
models/classical.py — Classical ML pipelines (SVM, Random Forest, KNN).

Usage (from notebook):
    from src.models.classical import build_svm_pipeline, build_rf_pipeline, build_knn_pipeline
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(pca_components: int = 128, seed: int = 42) -> Pipeline:
    """
    Color-histogram / flattened feature → StandardScaler → PCA → RBF-SVM.

    Hyperparameter search space (GridSearchCV):
        C: [0.1, 1, 10, 100]
        gamma: ['scale', 'auto']
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=seed)),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
    ])


def build_rf_pipeline(pca_components: int = 128, n_estimators: int = 200, seed: int = 42) -> Pipeline:
    """
    HOG / deep feature vector → StandardScaler → PCA → Random Forest.

    Hyperparameter search space (GridSearchCV):
        n_estimators: [100, 200, 400]
        max_depth: [None, 20, 40]
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=seed)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def build_knn_pipeline(pca_components: int = 64, n_neighbors: int = 5, seed: int = 42) -> Pipeline:
    """
    Deep feature vector (e.g., MobileNetV2 embeddings) → StandardScaler → PCA → KNN.

    Hyperparameter search space (GridSearchCV):
        n_neighbors: [3, 5, 7, 11]
        weights: ['uniform', 'distance']
        metric: ['euclidean', 'cosine']
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=seed)),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)),
    ])
