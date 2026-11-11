"""
Small helpers shared by all scripts.

clean_text() is a FIXED rule (nothing is learned from data), so it is safe to
apply it to train, validation, test and new user input in exactly the same way.
"""
import re

import yaml

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{8,}\d)")
MENTION_RE = re.compile(r"@\w+")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def remove_personal_data(text: str) -> str:
    """Replace links, e-mails, phone numbers and @mentions with placeholders."""
    text = URL_RE.sub(" <url> ", str(text))
    text = EMAIL_RE.sub(" <email> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    return text


def clean_text(text: str, max_chars: int = 1000) -> str:
    """Text -> normalised text that the model sees.

    1. remove personal data (links, e-mails, phones, @mentions)
    2. lower-case                      "ҚОТАҚ" -> "қотақ"
    3. "ё" -> "е"                       one spelling for the same letter
    4. letters repeated 3+ times -> 2   "сукаааа" -> "сукаа"
    5. collapse whitespace
    6. keep only the first max_chars characters (long comments)
    Punctuation, digits, emoji and the mask symbol "*" are KEPT: "б***" is a signal.
    """
    t = remove_personal_data(text).lower().replace("ё", "е")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def dedup_key(text: str) -> str:
    """Key for EXACT duplicates: only letters, no repeats.
    "Керемет!!!" and "кереемет" get the same key."""
    t = re.sub(r"[^a-zа-яәіңғүұқөһ]", "", str(text).lower().replace("ё", "е"))
    return re.sub(r"(.)\1+", r"\1", t)
