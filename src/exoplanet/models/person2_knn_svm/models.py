"""KNN and SVM estimators for Person 2 (Venkata Sai Bhavana Yarrapragada).

Every factory returns a fresh, unfitted scikit-learn Pipeline:
median imputation -> (optional) StandardScaler -> classifier.
Preprocessing comes from the shared ``make_preprocess_pipeline`` so it is fit
inside each training fold only.

Hyperparameters were selected on the saved development folds with grouped
5-fold cross-validation and macro-F1 (see
``experiments/person2_knn_svm/KNN_SVM_Yarrapragada.ipynb``). The test
partition was never used.
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from exoplanet.preprocess import make_preprocess_pipeline

# Final hyperparameters chosen on development folds.
# Configuration A = all 76 proposed features (primary scientific set).
# Configuration B = 56 features without the 20 provisional centroid offsets.
KNN_DEFAULT = {"n_neighbors": 5}
KNN_TUNED_A = {"n_neighbors": 7, "metric": "euclidean", "weights": "distance"}
KNN_TUNED_B = {"n_neighbors": 5, "metric": "euclidean", "weights": "uniform"}

LINEAR_SVM_DEFAULT = {"C": 1.0}
LINEAR_SVM_TUNED = {"C": 10.0, "class_weight": "balanced"}

RBF_SVM_DEFAULT = {"C": 1.0, "gamma": "scale"}
# C = 100 scored 0.0009 higher on configuration A, far inside the fold-to-fold
# standard deviation (about 0.01). C = 10 is kept because a smaller C means
# stronger regularization, so it is the simpler model at equal performance.
RBF_SVM_TUNED_A = {"C": 10.0, "gamma": "scale", "class_weight": "balanced"}
RBF_SVM_TUNED_B = {"C": 30.0, "gamma": 0.01, "class_weight": "balanced"}


def _pipeline(model, *, scale: bool) -> Pipeline:
    return Pipeline([*make_preprocess_pipeline(scale=scale).steps, ("model", model)])


def make_knn(*, scale: bool = True, **params) -> Pipeline:
    """K-nearest neighbours. Use scale=True except for the scaling experiment."""
    return _pipeline(KNeighborsClassifier(**params), scale=scale)


def make_linear_svm(*, scale: bool = True, **params) -> Pipeline:
    """Linear support vector machine (one-vs-rest)."""
    params.setdefault("max_iter", 10000)
    return _pipeline(LinearSVC(**params), scale=scale)


def make_rbf_svm(*, scale: bool = True, **params) -> Pipeline:
    """Support vector machine with an RBF kernel."""
    return _pipeline(SVC(kernel="rbf", **params), scale=scale)
