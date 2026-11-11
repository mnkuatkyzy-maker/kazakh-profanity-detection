"""
Step 1: build the labeled dataset and the train / validation / test split.

  raw KazSAnDRA reviews
    -> remove personal data (links, e-mails, phones, @mentions)
    -> remove exact duplicates
    -> tag every review with the labeling rules (strict / review / insult / none)
    -> select a pool: all strict + review + insult reviews, and a random sample of "none" reviews
    -> add the hand-written synthetic examples (train only)
    -> group near-duplicates (cosine similarity >= 0.80) so a group never crosses splits
    -> split groups 70 / 15 / 15 with a fixed seed
    -> final label: rules, overridden by data/labels/manual_labels.csv
    -> save data/processed/dataset.csv

Usage:  python src/prepare_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from labeling_rules import matched_words
from text_utils import dedup_key, load_config, remove_personal_data


def load_raw(cfg):
    frames = [pd.read_csv(os.path.join(cfg["data"]["raw_dir"], f)) for f in cfg["data"]["raw_files"]]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"custom_id": "id", "label": "rating"})
    df["rating"] = df["rating"] + 1                       # KazSAnDRA stores 0..4, we show 1..5 stars
    df["text"] = df["text"].astype(str).map(remove_personal_data).str.strip()
    df = df[df["text"].str.len() > 0]
    return df[["id", "text", "rating", "domain"]]


def select_pool(df, cfg, rng):
    """Keep every rule hit + a random sample of clean reviews (see config)."""
    hits = df[df["rule_tag"] != "none"]
    clean = df[df["rule_tag"] == "none"]
    parts = [hits]
    for domain, n in cfg["data"]["negatives_per_domain"].items():
        d = clean[clean["domain"] == domain]
        if domain == "appstore":
            # half of the app-store negatives are 1-2 star reviews: angry but not profane
            n_low = int(n * cfg["data"]["low_rating_share_appstore"])
            low, rest = d[d["rating"] <= 2], d[d["rating"] > 2]
            parts.append(low.sample(n=min(n_low, len(low)), random_state=int(rng.integers(1e9))))
            parts.append(rest.sample(n=min(n - n_low, len(rest)), random_state=int(rng.integers(1e9))))
        else:
            parts.append(d.sample(n=min(n, len(d)), random_state=int(rng.integers(1e9))))
    return pd.concat(parts, ignore_index=True)


def near_duplicate_groups(texts, threshold):
    """Union-find over pairs of comments with cosine similarity >= threshold.
    Character 3-5-grams make "керемет ойын!!" and "керемет ойын" similar."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(radius=1 - threshold, metric="cosine").fit(X)
    _, neighbours = nn.radius_neighbors(X)
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(texts))])


def split_groups(df, cfg, rng):
    """Assign whole groups to train/val/test.  Stratified by 'is the group a
    profanity candidate', so all three splits get a similar share of class 1."""
    frac = cfg["data"]["split"]
    df["split"] = ""
    # synthetic examples (and any real comment near-identical to one) -> train only
    syn_groups = df.loc[df["is_synthetic"] == 1, "group_id"].unique()
    df.loc[df["group_id"].isin(syn_groups), "split"] = "train"
    real = df[df["split"] == ""]
    group_has_candidate = real.groupby("group_id")["rule_tag"].apply(lambda s: s.isin(["strict", "review"]).any())
    for flag in [True, False]:
        groups = group_has_candidate[group_has_candidate == flag].index.to_numpy()
        groups = rng.permutation(groups)
        sizes = real.groupby("group_id").size().loc[groups].to_numpy()
        cum = np.cumsum(sizes) / sizes.sum()
        names = np.where(cum <= frac["train"], "train",
                         np.where(cum <= frac["train"] + frac["val"], "val", "test"))
        mapping = dict(zip(groups, names))
        mask = (df["split"] == "") & df["group_id"].isin(groups)
        df.loc[mask, "split"] = df.loc[mask, "group_id"].map(mapping)
    return df


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])

    df = load_raw(cfg)
    n_raw = len(df)
    df["key"] = df["text"].map(dedup_key)
    df = df[df["key"].str.len() > 0].drop_duplicates("key")          # exact duplicates
    n_dedup = len(df)

    tags = df["text"].map(matched_words)
    df["rule_tag"] = tags.map(lambda t: t[0])
    df["rule_hits"] = tags.map(lambda t: " | ".join(t[1]))
    df["is_synthetic"] = 0
    pool = select_pool(df, cfg, rng)

    # Hand-written synthetic examples (clearly marked, used only for training).
    syn = pd.read_csv(cfg["data"]["synthetic"])
    syn["text"] = syn["text"].map(remove_personal_data)
    syn["key"] = syn["text"].map(dedup_key)
    syn = syn[~syn["key"].isin(df["key"])]                             # never duplicate a real review
    syn["rating"], syn["is_synthetic"] = np.nan, 1
    tags = syn["text"].map(matched_words)
    syn["rule_tag"] = tags.map(lambda t: t[0])
    syn["rule_hits"] = tags.map(lambda t: " | ".join(t[1]))
    pool = pd.concat([pool, syn], ignore_index=True).drop_duplicates("key")

    pool["group_id"] = near_duplicate_groups(pool["key"].tolist(), cfg["data"]["near_duplicate_cosine"])
    pool = split_groups(pool, cfg, rng)

    # ---------------- labels ----------------
    pool["label"] = pool["rule_tag"].map({"strict": 1, "insult": 0, "none": 0, "review": -1})
    pool["label_source"] = "rule:" + pool["rule_tag"]
    is_syn = pool["is_synthetic"] == 1
    pool.loc[is_syn, "label"] = pool.loc[is_syn, "syn_label"]
    pool.loc[is_syn, "label_source"] = "synthetic"

    manual = pd.read_csv(cfg["data"]["manual_labels"]) if os.path.exists(cfg["data"]["manual_labels"]) \
        else pd.DataFrame(columns=["id", "label", "note"])
    manual = manual.set_index("id")
    # label "drop" = remove the comment (e.g. it contains a person's full name)
    dropped_ids = manual.index[manual["label"].astype(str) == "drop"]
    pool = pool[~pool["id"].isin(dropped_ids)].copy()
    manual = manual[manual["label"].astype(str) != "drop"]
    manual["label"] = manual["label"].astype(int)
    known = pool["id"].isin(manual.index)
    pool.loc[known, "label"] = pool.loc[known, "id"].map(manual["label"])
    pool.loc[known, "label_source"] = "manual"

    missing = pool[pool["label"] == -1]
    if len(missing):
        out = "data/labels/needs_review.csv"
        missing[["id", "split", "rule_hits", "text"]].to_csv(out, index=False)
        sys.exit(f"{len(missing)} comments need a manual label -> see {out}, add them to manual_labels.csv")

    pool["label"] = pool["label"].astype(int)
    cols = ["id", "text", "label", "split", "domain", "rating", "is_synthetic",
            "rule_tag", "rule_hits", "label_source", "group_id"]
    pool = pool[cols].sort_values(["split", "id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(cfg["data"]["processed"]), exist_ok=True)
    pool.to_csv(cfg["data"]["processed"], index=False)

    # ---------------- statistics for the README ----------------
    stats = {
        "raw_reviews": int(n_raw),
        "after_exact_dedup": int(n_dedup),
        "rule_tags_in_full_corpus": df["rule_tag"].value_counts().to_dict(),
        "dropped_for_privacy": int(len(dropped_ids)),
        "final_dataset_size": int(len(pool)),
        "synthetic_examples": int(pool["is_synthetic"].sum()),
        "near_duplicate_groups_with_2plus": int((pool.groupby("group_id").size() > 1).sum()),
        "split_sizes": pool["split"].value_counts().to_dict(),
        "class_distribution": {s: pool[pool["split"] == s]["label"].value_counts().sort_index().to_dict()
                               for s in ["train", "val", "test"]},
        "class_distribution_total": pool["label"].value_counts().sort_index().to_dict(),
        "label_source": pool["label_source"].value_counts().to_dict(),
        "domain_by_split": pd.crosstab(pool["domain"].fillna("synthetic"), pool["split"]).to_dict(),
        "seed": cfg["seed"],
    }
    # sanity check: a near-duplicate group must live in exactly one split
    assert (pool.groupby("group_id")["split"].nunique() == 1).all()
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["results_dir"], "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=int)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=int))


if __name__ == "__main__":
    main()
