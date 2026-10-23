#!/usr/bin/env python3
"""
eval.py — Compare predictions.json against gold.json
Usage:
    python eval.py
Reads:
    outputs/predictions.json
    data/gold.json
Prints:
    Accuracy, confusion matrix, false-supported count
"""

import json
import pathlib
import sys
from collections import defaultdict

PREDICTIONS_FILE = pathlib.Path("outputs/predictions.json")
GOLD_FILE = pathlib.Path("data/gold.json")

LABELS = ["supported", "overstated", "cannot_verify"]


def load_json(path: pathlib.Path) -> list:
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalise_verdict(v: str) -> str:
    v = v.lower().strip()
    if v in {"supported", "overstated", "cannot_verify", "cannot verify"}:
        return v.replace(" ", "_")
    return "cannot_verify"


def main():
    preds_list = load_json(PREDICTIONS_FILE)
    gold_list = load_json(GOLD_FILE)

    # Index by claim_id
    preds = {p["claim_id"]: p for p in preds_list}
    gold = {g["claim_id"]: g for g in gold_list}

    all_ids = sorted(set(preds.keys()) | set(gold.keys()))

    # ── Per-claim comparison ──
    results = []
    for cid in all_ids:
        g = gold.get(cid)
        p = preds.get(cid)
        gold_v = normalise_verdict(g["verdict"]) if g else "missing"
        pred_v = normalise_verdict(p["verdict"]) if p else "missing"
        correct = gold_v == pred_v
        is_false_supported = (gold_v != "supported") and (pred_v == "supported")
        results.append(
            {
                "claim_id": cid,
                "gold": gold_v,
                "predicted": pred_v,
                "correct": correct,
                "false_supported": is_false_supported,
            }
        )

    # ── Metrics ──
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / n if n > 0 else 0.0
    false_supported = [r for r in results if r["false_supported"]]

    # ── Confusion matrix ──
    # rows = gold, cols = predicted
    labels_used = LABELS + ["missing"]
    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r["gold"]][r["predicted"]] += 1

    # ─────────────── Print report ───────────────
    print("=" * 60)
    print("  FRACTIVA CLAIM VERIFIER — EVALUATION REPORT")
    print("=" * 60)
    print(f"\nClaims evaluated : {n}")
    print(f"Correct          : {n_correct}")
    print(f"Accuracy         : {accuracy:.1%}")
    print(f"False-supported  : {len(false_supported)}  <- worst error type")

    print("\n-- Per-claim results --")
    header = f"{'ID':<6} {'Gold':<17} {'Predicted':<17} {'OK':>4} {'False-supp':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        flag = "OK" if r["correct"] else "XX"
        fs = "*** FALSE SUPPORTED ***" if r["false_supported"] else ""
        print(
            f"{r['claim_id']:<6} {r['gold']:<17} {r['predicted']:<17} {flag:>4}  {fs}"
        )

    print("\n-- Confusion matrix (rows=gold, cols=predicted) --")
    col_labels = [l[:10] for l in LABELS]
    print(f"{'':>17}", end="")
    for cl in col_labels:
        print(f"  {cl:>12}", end="")
    print()
    for gold_l in LABELS:
        print(f"{gold_l:>17}", end="")
        for pred_l in LABELS:
            cnt = matrix[gold_l][pred_l]
            print(f"  {cnt:>12}", end="")
        print()

    if false_supported:
        print("\n-- False-supported detail --")
        for r in false_supported:
            pred_obj = preds.get(r["claim_id"], {})
            print(f"  {r['claim_id']}: predicted=supported  reason='{pred_obj.get('reason','')}'")

    print("\n" + "=" * 60)
    return 0 if len(false_supported) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
