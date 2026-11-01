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


def plot_dollar_capture_table(dollar_capture, output_dir="reports/figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "Review Rate",
        "Fraud Dollars Captured",
        "Dollar Capture Rate",
        "Fraud Cases Captured",
        "Case Capture Rate",
    ]

    rows = []

    for key, values in dollar_capture.items():
        rows.append([
            key.replace("top_", "").replace("pct", "%"),
            f"${values['fraud_dollars_captured']:,.2f}",
            f"{values['fraud_dollar_capture_rate']:.1%}",
            values["fraud_cases_captured"],
            f"{values['fraud_case_capture_rate']:.1%}",
        ])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    plt.title("Fraud Dollar Capture by Review Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "dollar_capture_table.png", dpi=300)
    plt.close(fig)

