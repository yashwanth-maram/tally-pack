"""Tally Pack - PackBench evaluation: SAM 3.1 tracks -> ledger -> verifier vs ground truth.

Outcome classes (what the system would tell a packer):
  PASS          no violations: the box matches the order and the evidence is consistent
  FAIL          C1 order violation: the ledger proves the box does not match the order
  NEEDS_REVIEW  no order violation, but integrity checks (C2-C4) say the evidence is unreliable
                -> this is what the investigator agent takes over

The metric that matters most is the false-clearance rate: a wrongly packed box marked PASS.
"""
import json
import os

from . import ledger as L

CONTAINER = "cardboard box"


def order_by_prompt(gt):
    """Order expressed per SAM prompt, with 0 for every SKU present on the table but not ordered."""
    return {gt["prompts"][s]: gt["order"].get(s, 0) for s in gt["prompts"]}


def outcome(report):
    checks = {v["check"] for v in report["violations"]}
    if "C1_order" in checks:
        return "FAIL"
    return "NEEDS_REVIEW" if checks else "PASS"


def score_clip(tracks, gt, params=None):
    prompts = sorted(set(gt["prompts"].values()))
    led = L.build_ledger(tracks, CONTAINER, prompts, params)
    rep = L.verify(led, order_by_prompt(gt), params)
    pred = outcome(rep)
    counts = {p: rep["counts"][p]["in_container"] for p in prompts}
    truth = {gt["prompts"][s]: n for s, n in gt["in_box"].items()}
    count_ok = all(counts.get(p, 0) == truth.get(p, 0) for p in prompts)
    return {"name": gt["name"], "scenario": gt["scenario"], "expected": gt["expected"], "pred": pred,
            "count_exact": count_ok, "counts": counts, "truth": truth,
            "violations": [f"{v['check']}:{v['detail']}" for v in rep["violations"]], "model_calls": rep["model_calls"]}


def summarize(results):
    n = len(results)
    fail_gt = [r for r in results if r["expected"] == "FAIL"]
    pass_gt = [r for r in results if r["expected"] == "PASS"]
    false_clear = [r for r in fail_gt if r["pred"] == "PASS"]
    caught = [r for r in fail_gt if r["pred"] == "FAIL"]
    cleared = [r for r in pass_gt if r["pred"] == "PASS"]
    review = [r for r in results if r["pred"] == "NEEDS_REVIEW"]
    return {
        "clips": n,
        "errors_caught": f"{len(caught)}/{len(fail_gt)}",
        "false_clearances": f"{len(false_clear)}/{len(fail_gt)}",
        "correct_clears": f"{len(cleared)}/{len(pass_gt)}",
        "needs_review": f"{len(review)}/{n}",
        "count_exact": f"{sum(r['count_exact'] for r in results)}/{n}",
        "model_calls": sum(r["model_calls"] for r in results),
    }


def run(frames_root, gts, tracks_dir, prompts_for=None, track_fn=None, load_fn=None, save_fn=None):
    """Track (or load cached tracks for) every clip, then score. Returns (results, summary)."""
    os.makedirs(tracks_dir, exist_ok=True)
    results = []
    for gt in gts:
        path = os.path.join(tracks_dir, f"{gt['name']}.json")
        if os.path.exists(path) and load_fn:
            tracks = load_fn(path)
        else:
            prompts = [CONTAINER] + sorted(set(gt["prompts"].values()))
            tracks = track_fn(os.path.join(frames_root, gt["name"]), prompts)
            save_fn(tracks, path)
        results.append(score_clip(tracks, gt))
    return results, summarize(results)
