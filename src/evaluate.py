"""Model evaluation with precision-recall focused metrics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def evaluate_all(
    results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path | None = None,
) -> dict:
    test_metrics = {}
    for name, result in results.items():
        model = result["model"]
        metrics = evaluate_model(model, X_test, y_test)
        test_metrics[name] = metrics
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    if output_dir:
        output_dir = Path(output_dir)
        _plot_pr_curves(results, X_test, y_test, output_dir / "pr_curves.png")
        _plot_confusion_matrices(test_metrics, output_dir / "confusion_matrices.png")

        serializable = {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in test_metrics.items()
        }
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(serializable, f, indent=2)

    return test_metrics


def _plot_pr_curves(
    results: dict, X_test: np.ndarray, y_test: np.ndarray, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, result in results.items():
        y_prob = result["model"].predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(metrics: dict, save_path: Path) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, metrics.items()):
        cm = np.array(m["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
