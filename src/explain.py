"""
Trace ONE comment from raw text to prediction (for the defense).

Prints every step:  raw text -> cleaned text -> n-grams -> TF-IDF vector
-> weighted sum (logit) -> sigmoid -> probability -> threshold -> label.

Usage:  python src/explain.py "Тауар бір аптада сынып қалды қотақбас сатушы"
"""
import sys

import numpy as np

from predict import load_model
from text_utils import clean_text


def main():
    text = " ".join(sys.argv[1:]) or "Курьер кешікті, тауар сынық келді. Ақшамды қайтарыңдар!"
    model, threshold = load_model()
    tfidf, logreg = model.named_steps["tfidf"], model.named_steps["logreg"]

    print("1) raw text      :", text)
    cleaned = tfidf.preprocessor(text)            # same clean_text used in training
    print("2) cleaned text  :", cleaned)

    analyzer = tfidf.build_analyzer()             # cleaning + cutting into char n-grams
    grams = analyzer(text)
    unique = set(grams)
    known = [g for g in unique if g in tfidf.vocabulary_]
    print(f"3) n-grams       : {len(grams)} pieces ({len(unique)} different), e.g. {grams[20:32]}")
    print(f"   in vocabulary : {len(known)} of {len(unique)}  (unknown n-grams are simply ignored)")

    x = tfidf.transform([text])                   # sparse TF-IDF vector, L2-normalised
    print(f"4) TF-IDF vector : {x.shape[1]} dimensions, {x.nnz} non-zero")

    w, b = logreg.coef_[0], logreg.intercept_[0]
    contrib = x.multiply(w).tocsr()               # weight * tfidf value for each n-gram
    idx, vals = contrib.indices, contrib.data
    order = np.argsort(vals)
    vocab = tfidf.get_feature_names_out()
    print("5) strongest n-grams (weight x tfidf):")
    for i in list(order[::-1][:6]) + list(order[:3]):
        print(f"     {vals[i]:+.3f}  '{vocab[idx[i]]}'")

    logit = float((x @ w)[0] + b)
    prob = 1 / (1 + np.exp(-logit))
    print(f"6) logit = sum + bias = {vals.sum():+.3f} + ({b:+.3f}) = {logit:+.3f}")
    print(f"7) probability = sigmoid(logit) = 1/(1+e^-logit) = {prob:.3f}")
    print(f"8) threshold (chosen on validation) = {threshold:.3f}")
    print(f"9) prediction  = {int(prob >= threshold)}  ->",
          "Kazakh profanity present" if prob >= threshold else "no Kazakh profanity")


if __name__ == "__main__":
    main()
