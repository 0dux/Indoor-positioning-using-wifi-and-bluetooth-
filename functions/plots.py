"""
plots.py — Creates and saves publication-quality visualizations.

Generates:
    1. Confusion matrix heatmaps for each model (classification)
    2. Accuracy comparison bar chart (classification)
    3. Error comparison bar chart (regression)

All plots are saved to the outputs/ folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Use a clean, modern style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_dir: str = "outputs",
    label_names: list[str] | None = None,
    ax=None,
) -> str:
    """
    Creates a heatmap of the confusion matrix for one model.

    The confusion matrix shows:
    - Rows = actual locations
    - Columns = predicted locations
    - Diagonal = correct predictions (we want these to be high)
    - Off-diagonal = misclassifications

    Parameters
    ----------
    cm           : confusion matrix (numpy array)
    model_name   : name of the model (e.g., "KNN")
    dataset_name : "WiFi-Only" or "WiFi+BLE Fused"
    output_dir   : directory to save the plot
    label_names  : optional list of location labels
    ax           : optional matplotlib axis (for embedding in subplots)

    Returns
    -------
    filepath : path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure if no axis provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    # Draw the heatmap with numbers in each cell
    sns.heatmap(
        cm,
        annot=True,          # show numbers in cells
        fmt="d",             # integer format
        cmap="Blues",        # blue color scheme
        square=True,         # make cells square
        cbar_kws={"shrink": 0.8},
        xticklabels=label_names if label_names else "auto",
        yticklabels=label_names if label_names else "auto",
        ax=ax,
    )

    ax.set_xlabel("Predicted Location", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Location", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} — {dataset_name}", fontsize=14, fontweight="bold")

    # Rotate tick labels for readability
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # Save
    safe_name = f"cm_{model_name.replace(' ', '_')}_{dataset_name.replace(' ', '_').replace('+', '_')}.png"
    filepath = os.path.join(output_dir, safe_name)

    if created_fig:
        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  💾 Saved: {filepath}")
    return filepath


def plot_all_confusion_matrices(
    results: dict,
    dataset_name: str,
    output_dir: str = "outputs",
    encoder=None,
) -> list[str]:
    """
    Plots confusion matrices for all three models side by side.

    Returns list of saved file paths.
    """
    label_names = list(encoder.classes_) if encoder else None

    # Create a figure with 3 subplots (one per model)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Confusion Matrices — {dataset_name}",
        fontsize=16, fontweight="bold", y=1.02,
    )

    saved_files = []
    for ax, (model_name, res) in zip(axes, results.items()):
        plot_confusion_matrix(
            cm=res["confusion_matrix"],
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            label_names=label_names,
            ax=ax,
        )

    plt.tight_layout()
    combo_path = os.path.join(output_dir, f"cm_all_{dataset_name.replace(' ', '_').replace('+', '_')}.png")
    fig.savefig(combo_path, dpi=150, bbox_inches="tight")
    saved_files.append(combo_path)
    plt.close(fig)
    print(f"  💾 Saved combined: {combo_path}")

    return saved_files


def plot_accuracy_comparison(
    wifi_results: dict,
    fused_results: dict,
    output_dir: str = "outputs",
) -> str:
    """
    Creates a grouped bar chart comparing WiFi-only vs Fused accuracy
    for each of the three models.

    This is the key visualization showing that adding BLE data
    improves (or doesn't improve) prediction accuracy.

    Returns
    -------
    filepath : path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(wifi_results.keys())
    wifi_accs  = [wifi_results[m]["accuracy"] * 100 for m in model_names]
    fused_accs = [fused_results[m]["accuracy"] * 100 for m in model_names]

    x = np.arange(len(model_names))  # bar positions
    width = 0.35                      # bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw two sets of bars side by side
    bars1 = ax.bar(x - width / 2, wifi_accs,  width, label="WiFi-Only",
                   color="#4A90D9", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, fused_accs, width, label="WiFi + BLE Fused",
                   color="#E8744F", edgecolor="white", linewidth=0.8)

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("WiFi-Only vs WiFi+BLE Fused — Accuracy Comparison",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim(0, 110)  # leave room for value labels
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "accuracy_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  💾 Saved: {filepath}")
    return filepath


def plot_error_comparison(
    wifi_results: dict,
    fused_results: dict,
    output_dir: str = "outputs",
    metric: str = "mean_error",
) -> str:
    """
    Creates a grouped bar chart comparing WiFi-only vs Fused errors
    for each model (regression task).

    Returns
    -------
    filepath : path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(wifi_results.keys())
    wifi_errs = [wifi_results[m][metric] for m in model_names]
    fused_errs = [fused_results[m][metric] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, wifi_errs, width, label="WiFi-Only",
                   color="#4A90D9", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, fused_errs, width, label="WiFi + BLE Fused",
                   color="#E8744F", edgecolor="white", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Distance Error (m)", fontsize=13, fontweight="bold")
    ax.set_title("WiFi-Only vs WiFi+BLE Fused — Error Comparison",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "error_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  💾 Saved: {filepath}")
    return filepath
