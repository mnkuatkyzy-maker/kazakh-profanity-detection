"""
Labeling rules used ONLY to BUILD the dataset (weak labels + manual review).

IMPORTANT: the trained model never sees these rules. They are used in
prepare_data.py to (1) find comments that probably contain profanity and
(2) decide which comments a human must check by hand.

Every comment gets one of three "rule tags":
  "strict"  -> contains an unambiguous obscene word        -> label 1
  "review"  -> contains a word that MAY be obscene         -> checked by hand
  "insult"  -> rude / insulting but NOT obscene (мал, ақымақ, лох ...)  -> label 0
  "none"    -> nothing found                               -> label 0
Manual decisions in data/labels/manual_labels.csv always override the rules.
"""
import re

# Latin letters that look like Cyrillic ones (people mix keyboards).
LATIN_TO_CYR = str.maketrans({"a": "а", "o": "о", "e": "е", "c": "с", "k": "к",
                              "p": "р", "x": "х", "y": "у", "i": "і", "t": "т"})


def rule_normalize(text: str) -> str:
    """Lower-case, map Latin look-alikes to Cyrillic and collapse repeated
    letters ("сукаааа" -> "сука", "қотаққа" -> "қотақа").  Used only for matching."""
    t = str(text).lower().translate(LATIN_TO_CYR)
    t = re.sub(r"(.)\1+", r"\1", t)          # any run of the same char -> one char
    return t


# Tokens = runs of letters and the mask symbol "*".
TOKEN_RE = re.compile(r"[a-zа-яёәіңғүұқөһ*]+")

# ---------------------------------------------------------------------------
# 1) STRICT: obscene stems (Kazakh + Russian mat used inside Kazakh comments).
#    A token that STARTS with one of these stems is obscene.
#    (Patterns are written for the collapsed form, e.g. "қотақ" not "қотаққ".)
# ---------------------------------------------------------------------------
STRICT_TOKEN_PATTERNS = [
    # Kazakh: "қотақ" (penis) and derived insults "қотақбас"
    r"^[қк]от[аао]?[қкгғ]", r"^[қк]ат[аа][қк](бас|пас)", r"^каток(бас|пас)",
    # Kazakh: "сік-" (to f***) in its many spellings: сігейін, сгин, сиктир...
    r"^с[іиы]?[гк](ей|еи|ем|ин|иін|ын|ыйн|ийн|іл|ил|ыл|іс|ыс|іп|ып|у$|сп)",
    r"^с[іиы][гк](е|і|т|п|ке|ен|ін|сін|тір|тир)",
    r"^с[іи]к$",
    # Kazakh: "ам" (vulgar word for female genitals) in inflected forms
    r"^ам(ы|ың|ына|ыңа|ын|ынды|ыңды|ымды|ым|дар|дары|ды|ди|шық|шык|шелек|ыс|ысы|ысы)",
    # Kazakh: "жалап" (whore), "ташақ" (testicles, vulgar)
    r"^д?жалап", r"^т[аоа]ша[қк]",
    # Russian mat inside Kazakh comments
    r"^бл[яә]", r"^бл[эи]а?т", r"^блият", r"^блия",
    r"^ху[йиеяюн]", r"^на?х+у", r"^нах$",
    r"^п[иеі]зд", r"^пизд",
    r"^еб[аоеб]", r"^ёб", r"^ёп",
    r"^сук(а|и|і|ал|ас|ан|ый|$)",
    r"^к?га[нн]?дон", r"^гондон",
    r"^пид[оа]р", r"^пидр",
    r"^д[ао]лба[её]?[йиюя]?[оеёюаъ]?[пб]", r"^д[ао]лба[ео]в", r"^долбаю",
    r"^шлюх", r"^мудак", r"^за[еий]?[еёи]?бал", r"^за[ий]бал",
]

# Multi-word / glued phrases: "шешеңнің амы", "әкеңнің аузын сігейін", ...
SIK = r"с[іиы]?[гк](ей|еи|ем|ін|ин|иін|ын|ыйн|ийн|іт|тір|тир|іп|ип|у|сін)"
STRICT_PHRASE_PATTERNS = [
    r"(шеше[ңн]|шешел|анаң|ана[ңн]ы|әке[ңн]|аке[ңн])\S*\s?(амы|аузы|ауз|ұрай|" + SIK + ")",
    r"(аузы[нң]?|амы[нң]?|амына)\s?" + SIK,
]

# ---------------------------------------------------------------------------
# 2) REVIEW: may be obscene, may be innocent -> a human decides.
#    e.g. "көт" (ass) vs "көтеру" (to lift), bare "ам", masked words "б***".
# ---------------------------------------------------------------------------
REVIEW_TOKEN_PATTERNS = [
    r"^ам$",
    r"^көт(?!ер|ір|ереж)",            # "көт" (ass) but not "көтеру" (to lift)
    r"^д[ао]лба[нс]",                  # далбан / далбаса (fool?) - context needed
    r"^шешен$", r"^шешен(ді|ің|нің|ін)$",
    r"^скин$", r"^сыкп", r"^бле[дт]",
    r"\*",                              # any masked token, e.g. "б***", "қ*тақ"
]

# ---------------------------------------------------------------------------
# 3) INSULT: rude but NOT obscene.  By our rules these stay label 0.
#    They are added on purpose as "hard negatives".
# ---------------------------------------------------------------------------
INSULT_TOKEN_PATTERNS = [
    r"^мал$", r"^малдар", r"^ақ?ымақ", r"^акмак", r"^оңбаған", r"^онбаган",
    r"^шошқа", r"^лох", r"^чмо", r"^урод", r"^дебил", r"^идиот", r"^блин",
    r"^мақұлық", r"^есек", r"^ешак", r"^жынды",
]

_strict = [re.compile(p) for p in STRICT_TOKEN_PATTERNS]
_phrase = [re.compile(p) for p in STRICT_PHRASE_PATTERNS]
_review = [re.compile(p) for p in REVIEW_TOKEN_PATTERNS]
_insult = [re.compile(p) for p in INSULT_TOKEN_PATTERNS]

# Innocent words that the strict stems would otherwise catch.
EXCEPTIONS = re.compile(
    r"^(скач|скид|скан|скор|скрин|скри|скот|скуч|склад|скайп|скейт|скам|скат|скол|ски$|скинд|скины|"
    r"сиг(нал|ма|ар)|сикрет|сикунд|сикл|сикер|сикир|сыктыв|сигил|сигу|"
    r"аман|амал|амин|амир|амыр|амбар|амер|ампул|амбул|"
    r"блесн|блеск|блест|блез|блем|блен|блек|блэк|бли[жзнц]|блин|"
    r"ебі|ебін|ебебі|"
    r"скин|скеи|скеит|хунд|"
    r"сук[іи]лер|хуин)"
)


def matched_words(text: str):
    """Return (tag, list_of_matching_tokens) for one comment."""
    norm = rule_normalize(text)
    tokens = TOKEN_RE.findall(norm)
    strict, review, insult = [], [], []
    for tok in tokens:
        is_exception = bool(EXCEPTIONS.match(tok))   # e.g. "скачать", "аман"
        if not is_exception and any(p.search(tok) for p in _strict):
            strict.append(tok)
        elif any(p.search(tok) for p in _review) and re.search(r"[а-яёәіңғүұқөһ]", tok):
            review.append(tok)
        elif any(p.search(tok) for p in _insult):
            insult.append(tok)
    for p in _phrase:
        strict += [m.group(0) for m in p.finditer(norm)]
    if strict:
        return "strict", strict
    if review:
        return "review", review
    if insult:
        return "insult", insult
    return "none", []
