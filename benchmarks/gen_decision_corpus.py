"""决策型训练语料生成器 — JevBench 形状（非 JevBench 污染）。

families:
  intent      banking77 train 重排（5 选项采样，10k 条真实意图）
  fact        $\mu$合成算术/事实 判断与选择（选项=数值题）
  noul        $\mu$合成规则场景（policy 是/否）
  routing     $\mu$合成工具路由（4 选项）
  extraction  $\mu$合成字段抽取（价格/日期/数量）
  ordinal     $\mu$合成 0-4 评分（5 档）

输出 JSONL: {family, state, instructions, criteria(可选), labels, expected}
JevBench public 题不参与生成（留作官方评测）。

    python gen_decision_corpus.py <out.jsonl> [--per-family N]
"""
import argparse
import json
import os
import random
import re


def load_banking77(rng: random.Random, per: int) -> list[dict]:
    import sys

    sys.path.insert(0, r"E:\O1-Flash")
    from benchmarks.real_data_bench import load_banking77 as _load
    train, _test, cats = _load()
    rows = []
    pool = list(range(len(train)))
    rng.shuffle(pool)
    for i in pool[:per]:
        text, idx = train[i]
        intent = cats[idx]
        distractors = rng.sample([c for c in cats if c != intent], 4)
        labels = [intent] + distractors
        rng.shuffle(labels)
        rows.append({
            "family": "intent",
            "state": text,
            "instructions": "Which intent does the user's message express?",
            "criteria": {lab: lab.replace("_", " ") for lab in labels},
            "labels": labels,
            "expected": intent,
        })
    return rows


ARITH_OPS = ["+", "-", "*"]


def gen_fact(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.6:
            a, b = rng.randint(11, 99), rng.randint(11, 99)
            op = rng.choice(ARITH_OPS)
            val = {"+": a + b, "-": a - b, "*": a * b}[op]
            others = set()
            while len(others) < 3:
                d = val + rng.randint(-12, 12)
                if d != val:
                    others.add(d)
            labels = [str(x) for x in [val] + sorted(others)]
            rng.shuffle(labels)
            rows.append({
                "family": "fact",
                "state": f"{a} {op} {b}",
                "instructions": "Compute the result. The state is an arithmetic expression.",
                "criteria": None,
                "labels": labels,
                "expected": str(val),
            })
        else:
            # simple true/false statements
            a = rng.randint(2, 40)
            b = rng.randint(2, 40)
            true_stmt = f"{a} + {b} equals {a + b}"
            false_stmt = f"{a} + {b} equals {a + b + rng.choice([-3, -2, -1, 1, 2, 3])}"
            stmt, exp = (true_stmt, "yes") if rng.random() < 0.5 else (false_stmt, "no")
            rows.append({
                "family": "fact",
                "state": stmt,
                "instructions": "Is the statement true?",
                "criteria": {"no": "The statement is false.",
                             "yes": "The statement is true."},
                "labels": ["no", "yes"],
                "expected": exp,
            })
    return rows


RULES = [
    ("The discount applies if the order total is at least {A} and the coupon code is valid.",
     "order total is at least {A}", "coupon code is valid"),
    ("Access is granted if the badge is active and the visitor is registered.",
     "badge is active", "visitor is registered"),
    ("Reimbursement is paid if the receipt is attached and the manager approved the request.",
     "receipt is attached", "manager approved the request"),
    ("Overtime is approved if the shift exceeds {A} hours and the request was filed in advance.",
     "shift exceeds {A} hours", "request was filed in advance"),
    ("The warranty covers the repair if the device is under {A} months old and damage is not accidental.",
     "device is under {A} months old", "damage is not accidental"),
]


def gen_noul(rng: random.Random, n: int) -> list[dict]:
    rows = []
    tpls = [
        ("approve if {A} and {B}", "and"),
        ("approve if {A} or {B}", "or"),
        ("deny if {A} and {B}", "deny_and"),
    ]
    for _ in range(n):
        tpl, mode = rng.choice(tpls)
        rule_body, cond_a, cond_b = rng.choice([
            (tpl, "the order total is at least {v}", "the coupon code is valid"),
            (tpl, "the badge is active", "the visitor is registered"),
            (tpl, "the receipt is attached", "the manager approved the request"),
            (tpl, "the device is under {v} months old", "the damage is not accidental"),
        ])
        v = rng.choice([50, 100, 200, 500])
        ca = cond_a.format(v=v)
        rule = "Approval rule: " + rule_body.format(A=ca, B=cond_b) + "."
        a_ok, b_ok = rng.random() < 0.55, rng.random() < 0.55

        def clause(cond: str, ok: bool) -> str:
            return cond if ok else f"the claim that {cond} is not satisfied"

        scenario = ("Scenario: In this case, " + clause(ca, a_ok) + " and "
                    + clause(cond_b, b_ok) + ".")
        if mode == "and":
            expected = "yes" if (a_ok and b_ok) else "no"
        elif mode == "or":
            expected = "yes" if (a_ok or b_ok) else "no"
        else:  # deny if both
            expected = "no" if (a_ok and b_ok) else "yes"
        rows.append({
            "family": "noul",
            "state": rule + "\n" + scenario,
            "instructions": ("Under the stated rule, is approval granted? "
                             "Treat unproved conditions as not satisfied."),
            "criteria": {"no": "Approval is not granted under the rule.",
                         "yes": "Approval is granted under the rule."},
            "labels": ["no", "yes"],
            "expected": expected,
        })
    return rows


TOOLS = {
    "search": "look up general information on the web",
    "calculator": "compute a numeric result",
    "calendar": "schedule or check an appointment",
    "email": "send or read a message",
}
TOOL_TMPL = [
    ("What is the population of Peru?", "search"),
    ("Multiply 148 by 27.", "calculator"),
    ("Move my 3pm meeting to Friday.", "calendar"),
    ("Send Anna the report draft.", "email"),
    ("Who won the 2018 world cup?", "search"),
    ("What is 15% of 240?", "calculator"),
    ("Book a dentist appointment for next Tuesday.", "calendar"),
    ("Read the latest message from the supplier.", "email"),
]


def gen_routing(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        query, truth = rng.choice(TOOL_TMPL)
        if rng.random() < 0.7:
            num = rng.randint(10, 400)
            mult = rng.randint(3, 90)
            query, truth = f"What is {num} times {mult}?", "calculator"
        labels = list(TOOLS)
        rows.append({
            "family": "routing",
            "state": query,
            "instructions": "Which tool should handle the request?",
            "criteria": {k: v for k, v in TOOLS.items()},
            "labels": labels,
            "expected": truth,
        })
    return rows


_RNG: random.Random | None = None


FIELDS = [
    ("price", r"[$](\d+(?:\.\d{2})?)", lambda: f"${_RNG.randint(3, 900)}.{_RNG.randint(0,99):02d}"),
    ("quantity", r"(\d+) (?:units|pieces|boxes)", lambda: f"{_RNG.randint(2, 400)} units"),
    ("date", r"(20\d\d-\d\d-\d\d)", lambda: f"20{_RNG.randint(20,29)}-{_RNG.randint(1,12):02d}-{_RNG.randint(1,28):02d}"),
]


def gen_extraction(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        name, pat, maker = rng.choice(FIELDS)
        truth = maker()
        filler = rng.choice([
            "The shipment arrives on Tuesday.",
            "Please file the document before noon.",
            "The vendor confirmed the order.",
        ])
        state = f"Invoice note: {filler} {name.capitalize()}: {truth}. Contact billing@example.com."
        candidates = {truth}
        while len(candidates) < 4:
            candidates.add(maker())
        labels = sorted(candidates)
        rows.append({
            "family": "extraction",
            "state": state,
            "instructions": f"Extract the {name} from the text.",
            "criteria": None,
            "labels": labels,
            "expected": truth,
        })
    return rows


ORDINAL_LABELS = ["0", "1", "2", "3", "4"]
ORDINAL_WORDS = {
    0: ["terrible", "awful", "horrible"],
    1: ["poor", "bad", "disappointing"],
    2: ["okay", "mediocre", "average"],
    3: ["good", "pleasant", "nice"],
    4: ["excellent", "outstanding", "fantastic"],
}


def gen_ordinal(rng: random.Random, n: int) -> list[dict]:
    rows = []
    subjects = ["the service", "the hotel", "the meal", "the delivery", "the support team",
                "the app", "the flight", "the workshop"]
    for _ in range(n):
        level = rng.randint(0, 4)
        word = rng.choice(ORDINAL_WORDS[level])
        subj = rng.choice(subjects)
        state = f"Review: {subj.capitalize()} was {word}."
        if rng.random() < 0.5:
            state = f"Review: I found {subj} {word}."
        rows.append({
            "family": "ordinal",
            "state": state,
            "instructions": "Rate the sentiment on a 0-4 scale (0 = worst, 4 = best).",
            "criteria": {l: f"Level {l}" for l in ORDINAL_LABELS},
            "labels": ORDINAL_LABELS,
            "expected": str(level),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-family", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global _RNG
    _RNG = rng = random.Random(args.seed)
    n = args.per_family

    rows = []
    rows += load_banking77(rng, per=min(n, 10000))
    rows += gen_fact(rng, n)
    rows += gen_noul(rng, n)
    rows += gen_routing(rng, n // 2)
    rows += gen_extraction(rng, n // 2)
    rows += gen_ordinal(rng, n // 2)
    rng.shuffle(rows)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {args.out}: {len(rows)} items")
    print("families:", dict(Counter(r["family"] for r in rows)))


if __name__ == "__main__":
    main()
