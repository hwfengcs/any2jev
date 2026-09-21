"""Training data: labelled System One requests, one JSON object per line.

A record is exactly a ``/v1/systemone`` request whose questions carry a ``label``::

    {"state": "...", "questions": {"dept": {"type": "choice", "instructions": "...",
        "criteria": {"billing": "...", "shipping": "..."}, "label": "billing"},
        "urgent": {"type": "noul", "instructions": "...", "label": true},
        "anger": {"type": "score", "instructions": "...", "criteria": ["calm", "annoyed", "furious"], "label": 2}}}

``label``: choice -> option key, noul -> bool, score -> level index. The same bytes a server would
receive are what the model trains on, so there is no train/serve prompt drift.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import QuestionSpec, SystemOneRequest, labels_of, render, to_specs

Record = dict[str, Any]


def load_jsonl(path: str | Path) -> list[Record]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                materialize(rec)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{n}: invalid record: {e}") from e
            out.append(rec)
    return out


def save_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def materialize(rec: Record) -> tuple[str, list[QuestionSpec]]:
    """Labelled request -> (state text, question specs with label indices)."""
    req = SystemOneRequest.model_validate(rec)
    specs = to_specs(req, labels_of(rec["questions"]))
    for s in specs:
        if s.label is None:
            raise ValueError(f"question {s.qid!r} has no label")
    return render(req.state), specs


def shuffle_choices(specs: list[QuestionSpec], rng: random.Random) -> list[QuestionSpec]:
    """Permute the options of every Choice question with >= 3 options (order-robustness augmentation)."""
    out = []
    for s in specs:
        if s.qtype == "choice" and s.n_options >= 3:
            perm = list(range(s.n_options))
            rng.shuffle(perm)
            out.append(s.permuted(perm))
        else:
            out.append(s)
    return out


def split_records(records: list[Record], val_frac: float = 0.1, seed: int = 0) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = int(round(len(records) * val_frac))
    val = set(idx[:n_val])
    return [r for i, r in enumerate(records) if i not in val], [r for i, r in enumerate(records) if i in val]


# ---------------------------------------------------------------------------------------------------
# Synthetic support-ticket generator: no download, labels follow known rules, enough phrasing variety
# that a model has to read the text. Used by the smoke tests and the quick-start.

DEPARTMENTS: dict[str, tuple[str, list[str]]] = {
    "billing": ("Charges, invoices, refunds and payment problems", [
        "I was charged twice for my last order", "my invoice shows the wrong amount",
        "the payment keeps failing at checkout", "there is a charge on my card I don't recognise",
        "I need a proper receipt for my purchase", "my subscription renewed even though I cancelled it",
        "the discount code was not applied to my total", "you billed me in the wrong currency"]),
    "shipping": ("Delivery status, delays and lost packages", [
        "my package has not arrived yet", "the tracking number shows no updates for a week",
        "the courier left my parcel at the wrong address", "my order says delivered but I have nothing",
        "can you change the delivery address on my order", "the delivery is three weeks late",
        "the box arrived but half the items are missing", "I never got a shipping confirmation"]),
    "returns": ("Exchanges, returns, wrong or damaged items", [
        "the shoes arrived in the wrong size", "the item came broken in the box",
        "I want to return a jacket I bought last week", "you sent me the wrong colour",
        "how do I exchange this for a different model", "the product does not match the description",
        "the screen was cracked when I opened it", "I'd like to send back the second pair I ordered"]),
    "technical": ("Bugs, outages, login and integration issues", [
        "the app crashes every time I open it", "I cannot log in to my account",
        "the API has returned 500 errors since this morning", "two-factor codes never arrive",
        "the dashboard is showing stale data", "the Stripe integration keeps failing",
        "the export button does nothing", "your webhook is sending duplicate events"]),
}
URGENCY: list[list[str]] = [
    ["whenever you get a chance", "no rush on this", "just letting you know", "it can wait a bit"],
    ["I'd appreciate a reply this week", "please look into it soon", "this is starting to become a problem"],
    ["I need this fixed today", "this is urgent, I am losing sales", "ASAP please, my whole team is blocked",
     "URGENT: I have a launch in two hours"],
]
URGENCY_LEVELS = ["Can wait", "Needs attention this week", "Needs attention today"]
TONE: dict[str, list[str]] = {
    "calm": ["Hi there,", "Hello,", "Good morning,", "Hi team,"],
    "frustrated": ["This is the second time I am writing about this.", "I'm quite frustrated by this.",
                   "Honestly, this is disappointing.", "I expected better."],
    "angry": ["This is unacceptable!!", "I am furious.", "What kind of service is this?!", "Absolutely ridiculous."],
}
REFUND = ["I would like a refund.", "Please refund me.", "Can I get my money back?", "I want my money back now."]
SIGNOFF = ["Thanks.", "Regards.", "Thank you.", "", "Best,", "Cheers."]
NONE_OPTIONS = [("other", "None of the above"), ("other", "Something else"), ("none", "None of these"),
                ("not_listed", "Not listed here"), ("other", None)]


def synthetic_record(rng: random.Random) -> Record:
    dept = rng.choice(list(DEPARTMENTS))
    issue = rng.choice(DEPARTMENTS[dept][1])
    urgency = rng.choices([0, 1, 2], weights=[0.35, 0.35, 0.30])[0]
    tone = rng.choices(["calm", "frustrated", "angry"], weights=[0.5, 0.3, 0.2])[0]
    refund = rng.random() < (0.5 if dept in ("billing", "returns") else 0.12)
    parts = [rng.choice(TONE[tone]), issue[0].upper() + issue[1:] + "."]
    if refund:
        parts.append(rng.choice(REFUND))
    u = rng.choice(URGENCY[urgency])
    if u:
        parts.append(u[0].upper() + u[1:] + ("." if not u.endswith(".") else ""))
    parts.append(rng.choice(SIGNOFF))
    text = " ".join(p for p in parts if p)
    r = rng.random()
    if r < 0.15:
        state: Any = {"ticket": {"channel": rng.choice(["email", "chat", "web form"]), "body": text}}
    elif r < 0.25:
        state = [{"role": "customer", "content": text}]
    elif r < 0.32:
        state = {"message": text, "customer_tier": rng.choice(["free", "pro", "enterprise"])}
    else:
        state = text

    questions: dict[str, Any] = {}
    # department (choice), sometimes with the true department removed and a "none of the above" option
    keys = list(DEPARTMENTS)
    rng.shuffle(keys)
    criteria: dict[str, Any] = {k: (DEPARTMENTS[k][0] if rng.random() < 0.7 else None) for k in keys}
    label = dept
    if rng.random() < 0.12:
        criteria.pop(dept)
        none_key, none_desc = rng.choice(NONE_OPTIONS)
        criteria[none_key] = none_desc
        label = none_key
    elif rng.random() < 0.15:
        none_key, none_desc = rng.choice(NONE_OPTIONS)
        criteria[none_key] = none_desc
    questions["department"] = {"type": "choice", "criteria": criteria, "label": label,
                               "instructions": rng.choice(["Which team should handle this ticket?",
                                                           "Route this message to the right department.",
                                                           {"question": "Which team should handle this?",
                                                            "focus": "Pick the single best fit."}])}
    questions["urgency"] = {"type": "score", "criteria": list(URGENCY_LEVELS), "label": urgency,
                            "instructions": rng.choice(["How urgent is this ticket?", "How soon does this need a reply?"])}
    questions["refund"] = {"type": "noul", "label": refund,
                           "instructions": rng.choice(["Does the customer ask for a refund?",
                                                       "Is the customer requesting their money back?"])}
    if rng.random() < 0.5:
        questions["refund"]["criteria"] = {"true": "Explicitly asks for money back", "false": "No refund mentioned"}
    tone_keys = ["calm", "frustrated", "angry"]
    rng.shuffle(tone_keys)
    questions["tone"] = {"type": "choice", "criteria": {k: None for k in tone_keys}, "label": tone,
                         "instructions": "What is the customer's tone?"}
    # keep a random non-empty subset so records have 1..4 questions
    keep = [q for q in questions if rng.random() < 0.75] or [rng.choice(list(questions))]
    return {"state": state, "questions": {q: questions[q] for q in keep}}


def synthetic_records(n: int, seed: int = 0) -> list[Record]:
    rng = random.Random(seed)
    return [synthetic_record(rng) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------
# Public datasets (optional dependency: ``datasets``). Each builder returns labelled requests.

AG_NEWS = {"world": "World news: politics, international affairs, conflicts",
           "sports": "Sports: games, athletes, teams, results",
           "business": "Business: companies, markets, economy, finance",
           "scitech": "Science and technology: research, gadgets, software, space"}
YELP_LEVELS = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
MNLI = {"entailment": "The hypothesis follows from the premise",
        "neutral": "The hypothesis may or may not be true given the premise",
        "contradiction": "The hypothesis contradicts the premise"}


def _load(repo: str, split: str, config: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("public dataset builders need `pip install any2jev[data]`") from e
    try:
        return load_dataset(repo, config, split=split)
    except Exception:  # noqa: BLE001 - script-based repos: read the Hub's parquet conversion
        return load_dataset(repo, config, split=split, revision="refs/convert/parquet")


def _take(ds, n: int, rng: random.Random):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [ds[i] for i in idx[:n]]


def build_boolq(split: str, n: int, rng: random.Random) -> list[Record]:
    ds = _load("google/boolq", "validation" if split == "test" else "train")
    return [{"state": {"passage": r["passage"]},
             "questions": {"answer": {"type": "noul", "instructions": r["question"].strip().rstrip("?") + "?",
                                      "label": bool(r["answer"])}}} for r in _take(ds, n, rng)]


def build_ag_news(split: str, n: int, rng: random.Random) -> list[Record]:
    ds = _load("fancyzhx/ag_news", "test" if split == "test" else "train")
    keys = list(AG_NEWS)
    out = []
    for r in _take(ds, n, rng):
        order = list(keys)
        rng.shuffle(order)
        out.append({"state": r["text"], "questions": {"topic": {
            "type": "choice", "instructions": "What is the topic of this news article?",
            "criteria": {k: AG_NEWS[k] for k in order}, "label": keys[int(r["label"])]}}})
    return out


def build_yelp(split: str, n: int, rng: random.Random) -> list[Record]:
    ds = _load("Yelp/yelp_review_full", "test" if split == "test" else "train")
    return [{"state": {"review": r["text"]},
             "questions": {"rating": {"type": "score", "instructions": "How many stars did the reviewer give?",
                                      "criteria": list(YELP_LEVELS), "label": int(r["label"])}}}
            for r in _take(ds, n, rng)]


SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def build_sst5(split: str, n: int, rng: random.Random) -> list[Record]:
    ds = _load("SetFit/sst5", "test" if split == "test" else "train")
    return [{"state": r["text"],
             "questions": {"sentiment": {"type": "score", "instructions": "How positive is this movie review sentence?",
                                         "criteria": list(SST5_LEVELS), "label": int(r["label"])}}}
            for r in _take(ds, n, rng)]


def build_mnli(split: str, n: int, rng: random.Random) -> list[Record]:
    ds = _load("nyu-mll/multi_nli", "validation_matched" if split == "test" else "train")
    keys = list(MNLI)
    out = []
    for r in _take(ds, n, rng):
        if r["label"] not in (0, 1, 2):
            continue
        out.append({"state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                    "questions": {"relation": {"type": "choice", "criteria": dict(MNLI),
                                               "instructions": "How does the hypothesis relate to the premise?",
                                               "label": keys[int(r["label"])]}}})
    return out


def build_banking77(split: str, n: int, rng: random.Random, k_options: tuple[int, int] = (4, 12)) -> list[Record]:
    ds = _load("legacy-datasets/banking77", "test" if split == "test" else "train")
    names = ds.features["label"].names
    out = []
    for r in _take(ds, n, rng):
        gold = names[int(r["label"])]
        k = rng.randint(*k_options)
        others = rng.sample([x for x in names if x != gold], k - 1)
        opts = others + [gold]
        rng.shuffle(opts)
        out.append({"state": r["text"], "questions": {"intent": {
            "type": "choice", "instructions": "Which banking intent does this message express?",
            "criteria": {o: o.replace("_", " ") for o in opts}, "label": gold}}})
    return out


PUBLIC_SOURCES = {"boolq": build_boolq, "ag_news": build_ag_news, "yelp": build_yelp, "sst5": build_sst5,
                  "mnli": build_mnli, "banking77": build_banking77}


def build_public(sources: Iterable[str], split: str, n_per_source: int, seed: int = 0) -> list[Record]:
    rng = random.Random(seed)
    out: list[Record] = []
    for s in sources:
        if s not in PUBLIC_SOURCES:
            raise ValueError(f"unknown source {s!r}; choose from {sorted(PUBLIC_SOURCES)}")
        recs = PUBLIC_SOURCES[s](split, n_per_source, rng)
        for r in recs:
            for q in r["questions"].values():
                q["source"] = s
        out += recs
    rng.shuffle(out)
    return out
