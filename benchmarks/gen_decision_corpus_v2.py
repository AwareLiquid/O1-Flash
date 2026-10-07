"""真实数据族决策语料 v2 — JevBench 形状（治合成族过拟合）。

families（全真实数据）:
  intent      banking77 (已用) + CLINC150 (real utterances, 150 intents)
  ordinal     FinancialPhraseBank (real sentiment sentences, 3 levels)
  choice_nli  ContractNLI (real contract clauses + hypotheses, 3-way)

输出 JSONL: {family, state, instructions, criteria, labels, expected}
JevBench public 题不参与（官方评测保留）。

    python gen_decision_corpus_v2.py <out.jsonl>
"""
import argparse
import json
import random


def try_load(ids):
    from datasets import load_dataset
    for kwargs in ids:
        try:
            ds = load_dataset(**kwargs)
            print("  loaded:", kwargs)
            return ds
        except Exception as ex:
            print("  failed:", kwargs, type(ex).__name__, str(ex)[:80])
    return None


def gen_clinc(rng: random.Random, limit: int) -> list[dict]:
    ds = try_load([
        {"path": "clinc/clinc_oos", "name": "plus", "split": "train"},
        {"path": "clinc/clinc_oos", "name": "small", "split": "train"},
        {"path": "clinc_oos", "name": "plus", "split": "train"},
    ])
    if ds is None:
        return []
    names = ds.features["intent"].names if hasattr(
        ds.features["intent"], "names") else None
    if not names:
        return []
    real_names = [n for n in names if n != "oos"]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx[:limit]:
        item = ds[int(i)]
        intent = names[item["intent"]]
        if intent == "oos":
            continue
        distractors = rng.sample([n for n in real_names if n != intent], 4)
        labels = [intent] + distractors
        rng.shuffle(labels)
        rows.append({
            "family": "intent",
            "state": item["text"],
            "instructions": "Which intent does the user's message express?",
            "criteria": {lab: lab.replace("_", " ") for lab in labels},
            "labels": labels,
            "expected": intent,
        })
    return rows


def gen_fpb(rng: random.Random, limit: int) -> list[dict]:
    ds = try_load([
        {"path": "takala/financial_phrasebank", "name": "sentences_50agree",
         "split": "train"},
        {"path": "nyu-mll/glue", "name": "sst2", "split": "train"},
        {"path": "stanfordnlp/sst2", "split": "train"},
    ])
    if ds is None:
        return []
    print("  columns:", ds.column_names)
    tcol = "sentence" if "sentence" in ds.column_names else "text"
    lcol = "label" if "label" in ds.column_names else ds.column_names[-1]
    label_names = getattr(ds.features[lcol], "names", None)
    full = {0: "strongly negative", 1: "negative", 2: "neutral",
            3: "positive", 4: "strongly positive"}
    rows = []
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx[:limit]:
        item = ds[int(i)]
        lab = int(item[lcol])
        name = label_names[lab] if label_names else str(lab)
        # map to the 0-4 scale
        level = {"negative": 0, "positive": 4, "neutral": 2}.get(
            str(name).lower(), {0: 0, 1: 4}.get(lab, 2))
        labels = [str(x) for x in range(5)]
        rows.append({
            "family": "ordinal",
            "state": str(item[tcol]),
            "instructions": "Rate the sentiment of the statement "
                            "on a 0-4 scale (0 = worst, 4 = best).",
            "criteria": {l: full[int(l)] for l in labels},
            "labels": labels,
            "expected": str(level),
        })
    return rows


def gen_contract_nli(rng: random.Random, limit: int) -> list[dict]:
    ds = try_load([
        {"path": "stanfordnlp/contract_nli", "split": "train"},
        {"path": "kiddothe2b/contract-nli", "split": "train"},
        {"path": "nyu-mll/glue", "name": "mnli", "split": "train"},
    ])
    if ds is None:
        return []
    cols = ds.column_names
    pcol = "premise" if "premise" in cols else cols[0]
    hcol = "hypothesis" if "hypothesis" in cols else cols[1]
    lcol = "label" if "label" in cols else cols[-1]
    label_names = getattr(ds.features[lcol], "names", None) or [
        "entailment", "neutral", "contradiction"]
    rows = []
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx[:limit]:
        item = ds[int(i)]
        premise = str(item[pcol])[:1200]
        hyp = str(item[hcol])
        exp = label_names[int(item[lcol])].lower()
        if exp == "not_mentioned":
            exp = "not_mentioned"
        rows.append({
            "family": "choice_nli",
            "state": premise + "\nHypothesis: " + hyp,
            "instructions": "What is the relationship between the text and "
                            "the hypothesis?",
            "criteria": {"entailment": "The text entails the hypothesis.",
                         "contradiction": "The text contradicts it.",
                         "neutral": "Neither entailed nor contradicted."},
            "labels": ["contradiction", "entailment", "neutral"],
            "expected": exp,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit-clinc", type=int, default=10000)
    ap.add_argument("--limit-fpb", type=int, default=4000)
    ap.add_argument("--limit-cnli", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    print("CLINC150 ...")
    rows += gen_clinc(rng, args.limit_clinc)
    print("FinancialPhraseBank ...")
    rows += gen_fpb(rng, args.limit_fpb)
    print("ContractNLI ...")
    rows += gen_contract_nli(rng, args.limit_cnli)
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {args.out}: {len(rows)} items")
    print("families:", dict(Counter(r["family"] for r in rows)))


if __name__ == "__main__":
    main()
