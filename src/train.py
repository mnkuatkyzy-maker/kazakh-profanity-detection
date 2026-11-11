"""
Step 2: train the classifier and choose settings + threshold on VALIDATION data.

Model = Pipeline(
    TfidfVectorizer(char_wb n-grams)   # text -> sparse vector of character n-gram weights
    LogisticRegression                  # vector -> probability of class 1 (profanity)
)
The vectorizer vocabulary and IDF weights are fitted on the TRAINING split only
(pipeline.fit(train)).  Validation is used to pick n-gram range, C and threshold.
The test split is not touched here.

Usage:  python src/train.py
"""
import functools
import itertools
import json
import os
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from text_utils import clean_text, load_config


def build_pipeline(cfg, ngram_range, C):
    m = cfg["model"]
    vectorizer = TfidfVectorizer(
        analyzer=m["analyzer"],              # character n-grams inside word boundaries
        ngram_range=tuple(ngram_range),      # e.g. (2, 5): all 2-,3-,4-,5-letter pieces
        preprocessor=functools.partial(clean_text, max_chars=cfg["preprocessing"]["max_chars"]),
        lowercase=False,                     # clean_text already lower-cases
        min_df=m["min_df"],                  # ignore n-grams seen in < 2 training comments
        max_features=m["max_features"],
        sublinear_tf=True,                   # 1 + log(tf): repeating a word 10x is not 10x stronger
    )
    classifier = LogisticRegression(
        C=C, class_weight=m["class_weight"], solver="liblinear",
        max_iter=2000, random_state=cfg["seed"],
    )
    return Pipeline([("tfidf", vectorizer), ("logreg", classifier)])


def best_threshold(y_true, prob):
    """Threshold that gives the highest F1 for class 1 on validation."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.argmax(f1[:-1]))              # last point has no threshold
    return float(thresholds[i]), float(f1[i]), float(precision[i]), float(recall[i])


def main():
    cfg = load_config()
    np.random.seed(cfg["seed"])
    df = pd.read_csv(cfg["data"]["processed"])
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    print(f"train={len(train)}  val={len(val)}")

    # ---------- 1) grid search on validation ----------
    rows = []
    for ngram_range, C in itertools.product(cfg["model"]["ngram_ranges"], cfg["model"]["C_values"]):
        pipe = build_pipeline(cfg, ngram_range, C).fit(train["text"], train["label"])
        prob = pipe.predict_proba(val["text"])[:, 1]
        rows.append({
            "ngram_range": str(ngram_range), "C": C,
            "val_average_precision": average_precision_score(val["label"], prob),
            "val_f1_at_0.5": f1_score(val["label"], prob >= 0.5),
            "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
        })
        print(rows[-1])
    search = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    search.to_csv(os.path.join(cfg["paths"]["results_dir"], "validation_search.csv"), index=False)

    # ---------- 2) best setting -> final model (trained on train only) ----------
    best = search.iloc[0]
    ngram_range, C = json.loads(best["ngram_range"].replace("(", "[").replace(")", "]")), float(best["C"])
    model = build_pipeline(cfg, ngram_range, C).fit(train["text"], train["label"])
    val_prob = model.predict_proba(val["text"])[:, 1]
    thr, f1, p, r = best_threshold(val["label"].to_numpy(), val_prob)
    print(f"best: ngram_range={ngram_range} C={C}  threshold={thr:.3f}  val F1={f1:.3f} P={p:.3f} R={r:.3f}")

    # ---------- 3) save model + everything needed for inference ----------
    os.makedirs(os.path.dirname(cfg["paths"]["model"]), exist_ok=True)
    joblib.dump(model, cfg["paths"]["model"])
    info = {
        "model": "TF-IDF character n-grams (char_wb) + Logistic Regression",
        "ngram_range": ngram_range, "C": C,
        "threshold": thr,
        "threshold_rule": "maximise F1 of class 1 on the validation split",
        "validation": {"f1": f1, "precision": p, "recall": r,
                       "average_precision": float(average_precision_score(val["label"], val_prob))},
        "n_features": len(model.named_steps["tfidf"].vocabulary_),
        "train_size": len(train), "val_size": len(val), "seed": cfg["seed"],
        "max_chars": cfg["preprocessing"]["max_chars"],
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    with open(cfg["paths"]["model_info"], "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # ---------- 4) most important n-grams (useful to explain the model) ----------
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["logreg"].coef_[0]
    order = np.argsort(coef)
    with open(os.path.join(cfg["paths"]["results_dir"], "top_ngrams.txt"), "w", encoding="utf-8") as f:
        f.write("Top 40 n-grams pushing towards PROFANITY (class 1):\n")
        f.writelines(f"  {coef[i]:+.2f}  '{vocab[i]}'\n" for i in order[::-1][:40])
        f.write("\nTop 40 n-grams pushing towards CLEAN (class 0):\n")
        f.writelines(f"  {coef[i]:+.2f}  '{vocab[i]}'\n" for i in order[:40])
    print("saved", cfg["paths"]["model"], "and", cfg["paths"]["model_info"])


if __name__ == "__main__":
    main()
