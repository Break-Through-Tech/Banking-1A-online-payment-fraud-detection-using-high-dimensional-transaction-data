import matplotlib.pyplot as plt
from pathlib import Path


def plot_precision_recall(y_true, y_score, output_dir="reports/figures"):
    from sklearn.metrics import PrecisionRecallDisplay

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display = PrecisionRecallDisplay.from_predictions(y_true, y_score)
    display.ax_.set_title("Precision-Recall Curve")
    display.figure_.savefig(output_dir / "precision_recall_curve.png")
    plt.close(display.figure_)


def plot_confusion_matrix(y_true, y_pred, output_dir="reports/figures"):
    from sklearn.metrics import ConfusionMatrixDisplay

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    display.ax_.set_title("Confusion Matrix")
    display.figure_.savefig(output_dir / "confusion_matrix.png")
    plt.close(display.figure_)
