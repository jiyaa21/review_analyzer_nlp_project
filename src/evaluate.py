"""Metrics (accuracy, precision, recall, F1, confusion matrix) - no sklearn needed."""
LABELS = ["positive", "negative", "neutral"]


def evaluate(y_true: list, y_pred: list) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and of equal length")
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    per_class = {}
    for lab in LABELS:
        tp = matrix[lab][lab]
        fp = sum(matrix[o][lab] for o in LABELS if o != lab)
        fn = sum(matrix[lab][o] for o in LABELS if o != lab)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[lab] = {"precision": round(precision, 3), "recall": round(recall, 3),
                          "f1": round(f1, 3)}
    correct = sum(matrix[l][l] for l in LABELS)
    return {
        "accuracy": round(correct / len(y_true), 3),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(LABELS), 3),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
