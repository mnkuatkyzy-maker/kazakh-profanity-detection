"""
Step 3: ONE evaluation on the held-out TEST split with the saved model and the
threshold chosen on validation.  Nothing is tuned here.

Saves to results/:
  test_metrics.json        precision / recall / F1 for class 1, confusion matrix, counts
  confusion_matrix.png     the confusion matrix as an image
  test_predictions.csv     every test comment with its probability and prediction
  test_errors.csv          only the mistakes (false positives and false negatives)

Usage:  python src/evaluate.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")                       # draw images without a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from text_utils import load_config


def main():
    cfg = load_config()
    res = cfg["paths"]["results_dir"]
    model = joblib.load(cfg["paths"]["model"])
    with open(cfg["paths"]["model_info"], encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    test = pd.read_csv(cfg["data"]["processed"]).query("split == 'test'").copy()
    test["prob_profanity"] = model.predict_proba(test["text"])[:, 1]
    test["pred"] = (test["prob_profanity"] >= threshold).astype(int)
    y, p = test["label"], test["pred"]

    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    metrics = {
        "n_test": int(len(test)),
        "n_test_class_1": int(y.sum()), "n_test_class_0": int((y == 0).sum()),
        "threshold": threshold,
        "precision_class_1": precision_score(y, p),
        "recall_class_1": recall_score(y, p),
        "f1_class_1": f1_score(y, p),
        "accuracy": accuracy_score(y, p),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                             "false_negative": int(fn), "true_positive": int(tp)},
        "f1_class_1_at_default_0.5": f1_score(y, test["prob_profanity"] >= 0.5),
    }
    # accuracy per domain (app store vs product reviews ...)
    metrics["per_domain"] = {
        d: {"n": int(len(g)), "n_class_1": int(g["label"].sum()),
            "accuracy": float((g["label"] == g["pred"]).mean())}
        for d, g in test.groupby("domain")
    }
    with open(os.path.join(res, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # ---------- confusion matrix image ----------
    cm = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    ax.set_xticks([0, 1], ["0 clean", "1 profanity"])
    ax.set_yticks([0, 1], ["0 clean", "1 profanity"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Test set (n={len(test)}), threshold={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(res, "confusion_matrix.png"), dpi=150)

    # ---------- predictions and errors for analysis ----------
    cols = ["id", "domain", "label", "pred", "prob_profanity", "label_source", "text"]
    test[cols].sort_values("prob_profanity", ascending=False).to_csv(
        os.path.join(res, "test_predictions.csv"), index=False)
    errors = test[test["label"] != test["pred"]].copy()
    errors["error_type"] = errors["label"].map({0: "false_positive", 1: "false_negative"})
    errors[["error_type"] + cols].sort_values(["error_type", "prob_profanity"]).to_csv(
        os.path.join(res, "test_errors.csv"), index=False)
    print(f"saved results to {res}/  ({len(errors)} errors)")


if __name__ == "__main__":
    main()
