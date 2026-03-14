"""
models.py — Trains and evaluates ML models for indoor positioning.

Default task is classification (predict location classes). If the dataset
has mostly unique classes (one sample per class), it automatically falls back
to coordinate regression and reports distance error metrics.

Models used:
    1. K-Nearest Neighbors (KNN)
    2. Support Vector Machine (SVM)
    3. Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error


def _should_use_regression(db: pd.DataFrame) -> bool:
    """
    Heuristic: if most samples are unique classes, classification is ill-posed.
    In that case, switch to coordinate regression.
    """
    n_samples = len(db)
    if n_samples == 0:
        return False
    n_classes = db["location"].nunique()
    return (n_classes / n_samples) > 0.5


def _extract_features_and_targets(
    db: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts feature arrays (X) and target arrays (y) from the DataFrames.

    task: "classification" uses integer labels, "regression" uses (x, y).
    """
    X_train = db[feature_cols].values
    X_test = test[feature_cols].values

    if task == "classification":
        y_train = db["label"].values
        y_test = test["label"].values
    else:
        y_train = db[["x", "y"]].values
        y_test = test[["x", "y"]].values

    return X_train, X_test, y_train, y_test


def train_and_evaluate_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """
    Trains KNN, SVM, and Random Forest on the given data.

    Parameters
    ----------
    X_train, X_test : feature arrays
    y_train, y_test : integer label arrays
    label_names     : optional list of class names for the confusion matrix

    Returns
    -------
    results : dict with structure:
        {
            'KNN':           {'accuracy': float, 'cm': ndarray, 'model': fitted model},
            'SVM':           {'accuracy': float, 'cm': ndarray, 'model': fitted model},
            'Random Forest': {'accuracy': float, 'cm': ndarray, 'model': fitted model},
        }
    """
    # ── Define the three classifiers ──
    models = {
        "KNN": KNeighborsClassifier(
            n_neighbors=3,     # use the 3 closest fingerprints to decide
            weights="distance", # closer points get more influence
            metric="euclidean",
        ),
        "SVM": SVC(
            kernel="rbf",   # radial basis function — good for non-linear boundaries
            C=10,            # regularization — higher = fit training data more closely
            gamma="scale",   # auto-scale based on feature variance
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,  # use 100 decision trees
            max_depth=None,    # let trees grow fully
            random_state=42,   # reproducible results
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  🏋️ Training {name}…")

        # Train the model on the database fingerprints
        model.fit(X_train, y_train)

        # Predict locations for the test points
        y_pred = model.predict(X_test)

        # Calculate accuracy (% of test points correctly located)
        acc = accuracy_score(y_test, y_pred)

        # Build confusion matrix (shows which locations get confused with which)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "model": model,
            "y_pred": y_pred,
        }

        print(f"    ✓ {name} accuracy: {acc:.2%}")

    return results


def train_and_evaluate_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Trains regressors to predict (x, y) coordinates and reports distance errors.

    Returns
    -------
    results : dict with structure:
        {
            'KNN': {'mean_error': float, 'rmse': float, 'model': fitted model, ...},
            'SVR': {'mean_error': float, 'rmse': float, 'model': fitted model, ...},
            'Random Forest': {'mean_error': float, 'rmse': float, 'model': fitted model, ...},
        }
    """
    models = {
        "KNN": KNeighborsRegressor(
            n_neighbors=3,
            weights="distance",
            metric="euclidean",
        ),
        "SVR": MultiOutputRegressor(
            SVR(kernel="rbf", C=10, gamma="scale")
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  🏋️ Training {name}…")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors = np.linalg.norm(y_test - y_pred, axis=1)
        mean_error = float(np.mean(errors)) if len(errors) else float("nan")
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred))) if len(y_test) else float("nan")

        results[name] = {
            "mean_error": mean_error,
            "rmse": rmse,
            "errors": errors,
            "y_pred": y_pred,
            "model": model,
        }

        print(f"    ✓ {name} mean error: {mean_error:.3f} m")

    return results


def run_full_evaluation(
    data: dict,
    dataset_name: str = "Dataset",
) -> dict:
    """
    End-to-end evaluation: extracts features from a data dict (as returned
    by fusion.py) and trains/evaluates all three models.

    Parameters
    ----------
    data : dict with keys 'db', 'test', 'encoder', 'features'
    dataset_name : str for display

    Returns
    -------
    results : dict of results per model
    """
    print(f"\n{'='*50}")
    print(f"  📊 Evaluating: {dataset_name}")
    print(f"{'='*50}")

    task = "regression" if _should_use_regression(data["db"]) else "classification"

    X_train, X_test, y_train, y_test = _extract_features_and_targets(
        data["db"], data["test"], data["features"], task
    )

    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"  Task type:    {task}")

    if task == "classification":
        label_names = list(data["encoder"].classes_)
        results = train_and_evaluate_classification(X_train, X_test, y_train, y_test, label_names)
        return {"task": task, "results": results, "label_names": label_names}

    results = train_and_evaluate_regression(X_train, X_test, y_train, y_test)
    return {"task": task, "results": results, "label_names": None}


def compare_results(wifi_eval: dict, fused_eval: dict) -> pd.DataFrame:
    """
    Creates a comparison DataFrame for WiFi-only vs WiFi+BLE fused approaches.

    For classification: compares accuracy.
    For regression: compares mean distance error.
    """
    task = wifi_eval.get("task")
    if task != fused_eval.get("task"):
        raise ValueError("WiFi and Fused evaluations must use the same task type.")

    wifi_results = wifi_eval["results"]
    fused_results = fused_eval["results"]

    rows = []
    for model_name in wifi_results:
        if task == "classification":
            rows.append({
                "Model": model_name,
                "WiFi-Only Accuracy":      wifi_results[model_name]["accuracy"],
                "WiFi+BLE Fused Accuracy": fused_results[model_name]["accuracy"],
                "Improvement": (
                    fused_results[model_name]["accuracy"]
                    - wifi_results[model_name]["accuracy"]
                ),
            })
        else:
            rows.append({
                "Model": model_name,
                "WiFi Mean Error (m)":  wifi_results[model_name]["mean_error"],
                "Fused Mean Error (m)": fused_results[model_name]["mean_error"],
                "Improvement (m)": (
                    wifi_results[model_name]["mean_error"]
                    - fused_results[model_name]["mean_error"]
                ),
            })

    df = pd.DataFrame(rows)
    return df
