"""
Step 4: predict on your own Kazakh comment with the SAVED model (no retraining).

Usage:
  python src/predict.py "Тауар керемет, рахмет!"          # one or more comments
  python src/predict.py                                     # interactive: type, Enter; empty line = exit

In Python / a notebook:
  import sys; sys.path.insert(0, "src")
  from predict import load_model, predict
  model, threshold = load_model()
  predict(["Тауар керемет"], model, threshold)
"""
import json
import os
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository folder
sys.path.insert(0, os.path.join(ROOT, "src"))                          # so joblib finds text_utils


def load_model(model_path=os.path.join(ROOT, "models", "profanity_model.joblib"),
               info_path=os.path.join(ROOT, "models", "model_info.json")):
    model = joblib.load(model_path)       # TF-IDF vocabulary + logistic regression weights
    with open(info_path, encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold


def predict(texts, model, threshold):
    """Return a list of (label, probability) - label 1 = Kazakh profanity present."""
    probs = model.predict_proba(texts)[:, 1]
    return [(int(p >= threshold), float(p)) for p in probs]


def show(text, label, prob, threshold):
    name = "1 = PROFANITY" if label else "0 = clean"
    print(f"{name:14s}  P(profanity) = {prob:.3f}  (threshold {threshold:.3f})  | {text}")


def main():
    model, threshold = load_model()
    if len(sys.argv) > 1:
        texts = sys.argv[1:]
        for t, (label, prob) in zip(texts, predict(texts, model, threshold)):
            show(t, label, prob, threshold)
        return
    print("Type a comment and press Enter (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        label, prob = predict([text], model, threshold)[0]
        show(text, label, prob, threshold)


if __name__ == "__main__":
    main()
